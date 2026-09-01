from __future__ import annotations

from gymnasium import Env
from omegaconf import DictConfig  # noqa: F401
import torch

import collections
import json
import os

import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace as bp  # noqa: F401

from typing import Dict, Optional, Union
from pathlib import Path

from src.behavior.base import Actor
from src.behavior.base import (
    model_requires_skill_input,
    model_uses_grasp,
    model_uses_grasp_colored,
    model_uses_guidance_point,
    model_uses_guidance_point_colored,
)
from src.common.skills import batch_skills_to_onehot_tensor
from src.common.image_annotations import resize_guidance_point_for_image
from src.visualization.render_mp4 import create_in_memory_mp4
from src.common.context import suppress_all_output
from src.common.tasks import task2idx
from src.common.eepose import (
    ROBOT_BASE,
    SIM_LOCAL,
    resolve_eepose_frame,
    select_policy_eepose,
)
from src.common.geometry import quaternion_xyzw_to_matrix
from src.common.gripper import (
    binarize_robot_state_gripper_width,
    normalizer_expects_binary_gripper_width,
)
from src.common.files import get_processed_path, trajectory_save_dir
from src.data_collection.io import save_raw_rollout
from src.data_processing.utils import filter_and_concat_robot_state
from src.data_processing.utils import resize, resize_crop
from tensordict import TensorDict

from copy import deepcopy

import wandb
import zarr
from datetime import datetime

from src.eval.skill_annotation_util import (
    draw_grasp_annotation_on_image,
    draw_guidance_point_on_image,
    draw_skill_on_image,
    get_annotation_bundle_all_envs,
    reset_skill_annotator,
)
from src.eval.annotation_noise import AnnotationNoiseConfig, write_guidance_shuffle_bank
from src.eval.perturb_util import PerturbContext, PerturbRunner
from src.eval.progress_schema import (
    append_tracking_annotation_histories,
    accumulate_episode_skill_stats,
    accumulate_tracking_error_records,
    build_tracking_error_summary,
    build_tracking_workspace_filter_summary,
    compute_episode_tracking_errors,
    compute_success_rates,
    get_task_progress_labels,
    new_tracking_workspace_counts,
    normalize_progress_counts,
    tracking_histories_are_complete,
)
from src.eval.vlm_guidance import (
    VLMGuidanceClient,
    VLMPrediction,
    policy_bundles_from_vlm,
    state_info_for_env,
)
from src.eval.vlm_point_metrics import (
    DEFAULT_MONTE_CARLO_SAMPLES_PER_PAIR,
    build_vlm_point_error_summary,
    make_point_error_record,
    merge_vlm_point_error_summaries,
)


RolloutStats = collections.namedtuple(
    "RolloutStats",
    [
        "success_rate",
        "n_success",
        "n_rollouts",
        "epoch_idx",
        "rollout_max_steps",
        "total_return",
        "total_reward",
        "state_counts",
        "skill_completion_counts",
        "skill_success_rates",
        "step_counts",
        "step_completion_counts",
        "step_success_rates",
        "tracking_error",
        "vlm_point_error",
        "vlm_model_revision",
        "n_saved_rollouts",
    ],
)

RolloutSaveValues = collections.namedtuple(
    "RolloutSaveValues",
    [
        "robot_states",
        "imgs1",
        "imgs2",
        "actions",
        "rewards",
        "parts_poses",
        "point_clouds",
        "depth_image1",
        "depth_image2",
        "skills",
        "skill_states",
        "assembly_steps",
        "guidance_points",
        "guidance_points_clean",
        "guidance_poses",
        "guidance_poses_clean",
        "guidance_gripper_widths",
        "guidance_points_2d",
        "grasp_annotations_2d",
        "camera_infos",
        "oracle_skills",
        "oracle_guidance_points_2d",
        "vlm_annotations",
        "vlm_point_error_records",
    ],
)


def _guidance_bank_records_for_episode(
    *,
    task: str,
    source_episode: int,
    skill_states,
    skills,
    guidance_points,
    guidance_poses,
    guidance_gripper_widths,
) -> list[dict]:
    sequences = [
        [] if sequence is None else sequence
        for sequence in (skill_states, guidance_points, guidance_poses)
    ]
    n_frames = min(len(sequence) for sequence in sequences)
    records = []
    previous_state = None
    visit_counts: dict[str, int] = {}
    for frame_idx in range(n_frames):
        raw_state = skill_states[frame_idx]
        skill_state = None if raw_state is None else str(raw_state)
        if skill_state == previous_state:
            continue
        previous_state = skill_state
        if skill_state is None:
            continue
        point = guidance_points[frame_idx]
        pose = guidance_poses[frame_idx]
        if point is None and pose is None:
            continue
        visit_idx = visit_counts.get(skill_state, 0)
        visit_counts[skill_state] = visit_idx + 1
        skill = (
            skills[frame_idx]
            if skills is not None and frame_idx < len(skills)
            else None
        )
        width = (
            guidance_gripper_widths[frame_idx]
            if guidance_gripper_widths is not None
            and frame_idx < len(guidance_gripper_widths)
            else None
        )
        records.append(
            {
                "task": task,
                "skill_state": skill_state,
                "skill_type": str(skill or skill_state.rsplit("-", 1)[-1]),
                "source_episode": int(source_episode),
                "visit_idx": int(visit_idx),
                "guidance_point": (
                    None
                    if point is None
                    else np.asarray(point, dtype=np.float32).reshape(3).tolist()
                ),
                "guidance_pose": (
                    None
                    if pose is None
                    else np.asarray(pose, dtype=np.float32).reshape(4, 4).tolist()
                ),
                "guidance_gripper_width": (
                    None if width is None else float(np.asarray(width).reshape(-1)[0])
                ),
                "guidance_frame": ROBOT_BASE,
            }
        )
    return records


def _add_sim_local_ee_pose_to_robot_state(env: Env, robot_state):
    if not isinstance(robot_state, dict):
        return robot_state
    robot_state = dict(robot_state)
    if "ee_pos_sim" not in robot_state or "ee_quat_sim" not in robot_state:
        if not all(
            hasattr(env, attr) for attr in ("rb_states", "ee_idxs", "base_idxs")
        ):
            return robot_state

        device = env.rb_states.device
        ee_idxs = torch.as_tensor(env.ee_idxs, device=device, dtype=torch.long)
        base_idxs = torch.as_tensor(env.base_idxs, device=device, dtype=torch.long)
        ee_pos_global = env.rb_states[ee_idxs, :3].clone()
        ee_quat = env.rb_states[ee_idxs, 3:7].clone()
        base_pos_global = env.rb_states[base_idxs, :3].clone()

        if hasattr(env, "franka_from_origin_mat"):
            franka_origin = torch.as_tensor(
                np.asarray(env.franka_from_origin_mat, dtype=np.float32)[:3, 3],
                device=device,
                dtype=ee_pos_global.dtype,
            )
            env_offset = base_pos_global - franka_origin
            ee_pos_sim = ee_pos_global - env_offset
        else:
            ee_pos_sim = ee_pos_global
        robot_state["ee_pos_sim"] = ee_pos_sim
        robot_state["ee_quat_sim"] = ee_quat

    def pose_matrix(pos, quat):
        pose = torch.zeros(
            (*pos.shape[:-1], 4, 4), dtype=pos.dtype, device=pos.device
        )
        pose[..., :3, :3] = quaternion_xyzw_to_matrix(quat)
        pose[..., :3, 3] = pos
        pose[..., 3, 3] = 1.0
        return pose

    if "ee_pose" not in robot_state:
        robot_state["ee_pose"] = pose_matrix(
            robot_state["ee_pos"], robot_state["ee_quat"]
        )
    if "ee_pose_sim" not in robot_state:
        robot_state["ee_pose_sim"] = pose_matrix(
            robot_state["ee_pos_sim"], robot_state["ee_quat_sim"]
        )
    robot_state.setdefault("ee_pos_original", robot_state["ee_pos_sim"])
    robot_state.setdefault("ee_quat_original", robot_state["ee_quat_sim"])
    robot_state.setdefault("ee_pose_original", robot_state["ee_pose_sim"])
    return robot_state


def _policy_action_to_robot_base(env: Env, action, robot_state, eepose_frame):
    """Convert legacy sim-local absolute targets to canonical robot-base.

    The two sim representations differ only by a translation, so delta actions
    need no conversion.  Absolute position actions must remove that offset.
    """

    resolved_frame = resolve_eepose_frame(
        eepose_frame, original_frame=SIM_LOCAL
    )
    if resolved_frame == ROBOT_BASE or getattr(env, "action_type", None) != "pos":
        return action
    if resolved_frame != SIM_LOCAL:
        raise ValueError(
            f"eepose frame {resolved_frame!r} is not available in simulation"
        )
    missing = [
        key for key in ("ee_pos", "ee_pos_sim") if key not in robot_state
    ]
    if missing:
        raise KeyError(
            "cannot convert sim-local absolute action; missing robot-state fields: "
            + ", ".join(missing)
        )
    action_robot_base = action.clone()
    sim_local_offset = robot_state["ee_pos_sim"] - robot_state["ee_pos"]
    action_robot_base[..., :3] = action[..., :3] - sim_local_offset
    return action_robot_base


def resize_image(obs, key):
    try:
        obs[key] = resize(obs[key])
    except KeyError:
        pass

def resize_depth(obs, key):
    try:
        if obs.get(key) is None:
            return
        # key : [B, H, W]
        depth_image = obs[key].unsqueeze(-1)  # [B, H, W, C]
        obs[key] = resize(depth_image).squeeze(-1)
    except KeyError:
        pass

def resize_crop_image(obs, key):
    try:
        obs[key] = resize_crop(obs[key])
    except KeyError:
        pass

def resize_crop_depth(obs, key):
    try:
        if obs.get(key) is None:
            return
        # key : [B, H, W]
        depth_image = obs[key].unsqueeze(-1)  # [B, H, W, C]
        obs[key] = resize_crop(depth_image).squeeze(-1)
    except KeyError:
        pass

def squeeze_and_numpy(d: Dict[str, Union[torch.Tensor, np.ndarray, float, int, None]]):
    """
    Recursively squeeze and convert tensors to numpy arrays
    Convert scalars to floats
    Leave NoneTypes alone
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = squeeze_and_numpy(v)

        elif v is None:
            continue

        elif isinstance(v, (torch.Tensor, np.ndarray)):
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            d[k] = v.squeeze()

        else:
            raise ValueError(f"Unsupported type: {type(v)}")

    return d


def tensordict_to_list_of_dicts(tensordict):
    list_of_dicts = []
    keys = list(tensordict.keys())
    num_elements = tensordict[keys[0]].shape[0]

    for i in range(num_elements):
        dict_element = {}
        for key in keys:
            dict_element[key] = tensordict[key][i].cpu().numpy()
        list_of_dicts.append(dict_element)

    return list_of_dicts


def _resize_guidance_point_for_image(
    guidance_point_2d,
    bundle,
    image_key: str,
    image_shape,
):
    camera_info = bundle.get("camera_info", {}).get(image_key)
    if not camera_info:
        return guidance_point_2d
    return resize_guidance_point_for_image(
        guidance_point_2d,
        image_key=image_key,
        source_image_size=camera_info["image_size"],
        image_shape=image_shape,
    )


def _resize_grasp_annotation_for_image(
    grasp_annotation_2d,
    bundle,
    image_key: str,
    image_shape,
):
    if not grasp_annotation_2d:
        return None

    camera_info = bundle.get("camera_info", {}).get(image_key)
    if not camera_info:
        return grasp_annotation_2d

    source_width, source_height = [int(v) for v in camera_info["image_size"]]
    target_height, target_width = image_shape[:2]
    if source_width == target_width and source_height == target_height:
        return grasp_annotation_2d

    corners = np.asarray(grasp_annotation_2d.get("corners"), dtype=np.float32)
    center = np.asarray(grasp_annotation_2d.get("center"), dtype=np.float32)

    def _resize_points(points: np.ndarray) -> Optional[np.ndarray]:
        pts = points.copy()
        if image_key == "color_image1":
            sx = target_width / max(source_width, 1)
            sy = target_height / max(source_height, 1)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
        elif image_key == "color_image2":
            aspect_ratio = source_width / max(source_height, 1)
            resized_width = int(target_height * aspect_ratio)
            crop_size = max(0, (resized_width - target_width) // 2)
            sx = resized_width / max(source_width, 1)
            sy = target_height / max(source_height, 1)
            pts[:, 0] = pts[:, 0] * sx - crop_size
            pts[:, 1] *= sy
        else:
            return points

        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= target_width):
            return None
        if np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= target_height):
            return None
        return pts.astype(np.float32)

    resized_corners = _resize_points(corners)
    if resized_corners is None:
        return None
    resized_center = _resize_points(center[None, :])
    if resized_center is None:
        return None

    resized = dict(grasp_annotation_2d)
    resized["corners"] = resized_corners
    resized["center"] = resized_center[0]
    return resized


def _draw_guidance_points_for_all_envs(
    video_obs, annotation_bundles, annotate_wrist_camera: bool,
    guidance_point_colored: bool = False,
):
    image_keys = ["color_image2"]
    if annotate_wrist_camera:
        image_keys.append("color_image1")

    for image_key in image_keys:
        if image_key not in video_obs:
            continue
        image_batch = video_obs[image_key].cpu().numpy()
        annotated_batch = image_batch.copy()
        for env_idx, bundle in enumerate(annotation_bundles):
            guidance = _resize_guidance_point_for_image(
                bundle.get("guidance_point_2d", {}).get(image_key),
                bundle,
                image_key,
                annotated_batch[env_idx].shape,
            )
            annotated_batch[env_idx] = draw_guidance_point_on_image(
                annotated_batch[env_idx],
                guidance,
                skill=bundle.get("skill"),
                use_skill_color=guidance_point_colored,
            )
        video_obs[image_key] = torch.from_numpy(annotated_batch).to(video_obs[image_key].device)


def _draw_grasp_annotations_for_all_envs(
    video_obs, annotation_bundles, annotate_wrist_camera: bool,
    grasp_annotation_colored: bool = False,
):
    image_keys = ["color_image2"]
    if annotate_wrist_camera:
        image_keys.append("color_image1")

    for image_key in image_keys:
        if image_key not in video_obs:
            continue
        image_batch = video_obs[image_key].cpu().numpy()
        annotated_batch = image_batch.copy()
        for env_idx, bundle in enumerate(annotation_bundles):
            grasp_annotation = _resize_grasp_annotation_for_image(
                bundle.get("grasp_annotation_2d", {}).get(image_key),
                bundle,
                image_key,
                annotated_batch[env_idx].shape,
            )
            annotated_batch[env_idx] = draw_grasp_annotation_on_image(
                annotated_batch[env_idx],
                grasp_annotation,
                skill=bundle.get("skill"),
                use_skill_color=grasp_annotation_colored,
            )
        video_obs[image_key] = torch.from_numpy(annotated_batch).to(video_obs[image_key].device)


def _draw_grasp_part_annotations_for_all_envs(
    video_obs,
    annotation_bundles,
    annotate_wrist_camera: bool,
    guidance_point_colored: bool = False,
    grasp_annotation_colored: bool = False,
):
    image_keys = ["color_image2"]
    if annotate_wrist_camera:
        image_keys.append("color_image1")

    grasp_skills = {"pick", "place"}
    for image_key in image_keys:
        if image_key not in video_obs:
            continue
        image_batch = video_obs[image_key].cpu().numpy()
        annotated_batch = image_batch.copy()
        for env_idx, bundle in enumerate(annotation_bundles):
            skill = bundle.get("skill")
            frame = annotated_batch[env_idx]
            if skill in grasp_skills:
                grasp_annotation = _resize_grasp_annotation_for_image(
                    bundle.get("grasp_annotation_2d", {}).get(image_key),
                    bundle,
                    image_key,
                    frame.shape,
                )
                annotated_batch[env_idx] = draw_grasp_annotation_on_image(
                    frame,
                    grasp_annotation,
                    skill=skill,
                    use_skill_color=grasp_annotation_colored,
                )
            else:
                guidance = _resize_guidance_point_for_image(
                    bundle.get("guidance_point_2d", {}).get(image_key),
                    bundle,
                    image_key,
                    frame.shape,
                )
                annotated_batch[env_idx] = draw_guidance_point_on_image(
                    frame,
                    guidance,
                    skill=skill,
                    use_skill_color=guidance_point_colored,
                )
        video_obs[image_key] = torch.from_numpy(annotated_batch).to(video_obs[image_key].device)


def _apply_policy_visual_annotations(
    obs,
    annotation_bundles,
    *,
    annotate_wrist_camera: bool,
    annotate_guidance_point: bool,
    annotate_grasp: bool,
    grasp_part_annotate: bool,
    guidance_point_colored: bool,
    grasp_annotation_colored: bool,
):
    if grasp_part_annotate:
        _draw_grasp_part_annotations_for_all_envs(
            obs,
            annotation_bundles,
            annotate_wrist_camera=annotate_wrist_camera,
            guidance_point_colored=guidance_point_colored,
            grasp_annotation_colored=grasp_annotation_colored,
        )
        return

    if annotate_guidance_point:
        _draw_guidance_points_for_all_envs(
            obs,
            annotation_bundles,
            annotate_wrist_camera=annotate_wrist_camera,
            guidance_point_colored=guidance_point_colored,
        )
    if annotate_grasp:
        _draw_grasp_annotations_for_all_envs(
            obs,
            annotation_bundles,
            annotate_wrist_camera=annotate_wrist_camera,
            grasp_annotation_colored=grasp_annotation_colored,
        )


def _saved_pickle_image_annotation_mode(
    *,
    guidance_point_on_image: bool,
    grasp_annotation_on_image: bool,
    grasp_part_annotate: bool,
    guidance_point_colored: bool,
    grasp_annotation_colored: bool,
    skill_on_image: bool,
) -> str:
    """Describe pixels persisted by the rollout collector."""
    if grasp_part_annotate:
        return "grasp-part-colored" if grasp_annotation_colored else "grasp-part"
    if guidance_point_on_image:
        return "guidance-point-colored" if guidance_point_colored else "guidance-point"
    if grasp_annotation_on_image:
        return "grasp-colored" if grasp_annotation_colored else "grasp"
    if skill_on_image:
        return "skill"
    return "none"


def _transpose_step_env_annotations(values, num_envs: int):
    if not values:
        return [[] for _ in range(num_envs)]
    return [[values[step_idx][env_idx] for step_idx in range(len(values))] for env_idx in range(num_envs)]


def _build_rollout_progress_summary(rollout_stats: RolloutStats) -> dict:
    return {
        "n_success": int(rollout_stats.n_success),
        "n_rollouts": int(rollout_stats.n_rollouts),
        "success_rate": float(rollout_stats.success_rate),
        "rollout_max_steps": int(rollout_stats.rollout_max_steps),
        "total_return": float(rollout_stats.total_return),
        "total_reward": float(rollout_stats.total_reward),
        "skill_state_counts": dict(rollout_stats.state_counts),
        "skill_completion_counts": dict(rollout_stats.skill_completion_counts),
        "skill_success_rates": dict(rollout_stats.skill_success_rates),
        "assembly_step_counts": dict(rollout_stats.step_counts),
        "assembly_step_completion_counts": dict(rollout_stats.step_completion_counts),
        "assembly_step_success_rates": dict(rollout_stats.step_success_rates),
        "tracking_error": rollout_stats.tracking_error,
        "vlm_point_error": rollout_stats.vlm_point_error,
        "vlm_model_revision": rollout_stats.vlm_model_revision,
    }


def _flatten_rollout_path_for_log_name(rollout_path_hint: Optional[Path]) -> str:
    if rollout_path_hint is None:
        return "rollout"

    raw_root_env = os.environ.get("DATA_DIR_RAW")
    candidate_path = rollout_path_hint.expanduser()
    relative_path = candidate_path

    if raw_root_env:
        try:
            raw_root = (Path(raw_root_env).expanduser().resolve() / "raw").resolve()
            relative_path = candidate_path.resolve().relative_to(raw_root)
        except (OSError, RuntimeError, ValueError):
            relative_path = candidate_path

    flattened_parts = [part for part in relative_path.parts if part not in ("", ".", "..")]
    if not flattened_parts:
        return "rollout"
    return "__".join(flattened_parts)


def write_rollout_progress_log(
    log_dir: Path,
    rollout_stats: RolloutStats,
    epoch_idx: int,
    task_name: str,
    rollout_path_hint: Optional[Path] = None,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    rollout_name = _flatten_rollout_path_for_log_name(rollout_path_hint)
    log_path = log_dir / f"{rollout_name}__epoch_{epoch_idx:06d}__{timestamp}.json"

    payload = {
        "task": task_name,
        "epoch": int(epoch_idx),
        "timestamp": timestamp,
        "rollout_path_hint": str(rollout_path_hint) if rollout_path_hint is not None else None,
        **_build_rollout_progress_summary(rollout_stats),
    }

    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Saved rollout progress log to: {log_path}")
    return log_path


class SuccessTqdm(tqdm):
    def __init__(
        self,
        num_envs: int,
        n_rollouts: int,
        task_name: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_envs = num_envs
        self.n_rollouts = n_rollouts
        self.task_name = task_name
        self.round = 0
        self.success_in_prev_rounds = 0
        self.target_only = False

    def pbar_desc(self, n_success: int):
        total = self.round * self.num_envs
        n_success += self.success_in_prev_rounds
        success_rate = n_success / total if total > 0 else 0
        if self.target_only:
            desc = (
                f"Performing rollouts ({self.task_name}): "
                f"round {self.round}, "
                f"success: {n_success}/{total} ({success_rate:.1%})"
            )
        else:
            desc = (
                f"Performing rollouts ({self.task_name}): "
                f"round {self.round}/{self.n_rollouts//self.num_envs}, "
                f"success: {n_success}/{total} ({success_rate:.1%})"
            )
        self.set_description(desc)

    def before_round(self, n_success: int):
        self.success_in_prev_rounds = n_success
        self.round += 1

        self.pbar_desc(0)


def _attach_skill_tensor_to_obs(obs, actor: Actor, skills):
    if not getattr(actor, "requires_skill_input", False):
        return

    obs["skill"] = batch_skills_to_onehot_tensor(
        skills,
        skill_dim=getattr(actor, "skill_dim", 0),
        device=actor.device,
        dtype=torch.float32,
    )


def _query_vlm_annotations(
    *,
    client: VLMGuidanceClient,
    env: Env,
    obs,
    oracle_bundles,
    step_idx: int,
) -> tuple[list[dict], list[VLMPrediction]]:
    if "color_image1" not in obs or "color_image2" not in obs:
        raise ValueError("VLM guidance requires both wrist and front RGB images")
    robot_state = _add_sim_local_ee_pose_to_robot_state(env, obs.get("robot_state"))
    if not isinstance(robot_state, dict):
        raise ValueError("VLM guidance requires dictionary robot_state")
    predictions, _ = client.predict(
        task=env.furniture_name,
        front_images=[obs["color_image2"][idx] for idx in range(env.num_envs)],
        wrist_images=[obs["color_image1"][idx] for idx in range(env.num_envs)],
        state_infos=[
            state_info_for_env(robot_state, idx) for idx in range(env.num_envs)
        ],
        step_idx=step_idx,
    )
    return policy_bundles_from_vlm(
        oracle_bundles, predictions, step_idx=step_idx
    ), predictions


def _record_vlm_point_errors(
    records_per_env,
    oracle_bundles,
    predictions,
    active_mask,
    *,
    step_idx: int,
    episode_offset: int = 0,
    noise_projection_samples: int = DEFAULT_MONTE_CARLO_SAMPLES_PER_PAIR,
) -> None:
    for env_idx, (oracle, prediction) in enumerate(zip(oracle_bundles, predictions)):
        if not active_mask[env_idx]:
            continue
        oracle_point = oracle.get("guidance_point_2d", {}).get("color_image2")
        oracle_point_3d = oracle.get("guidance_point")
        front_camera_info = oracle.get("camera_info", {}).get("color_image2")
        noise_seed = (
            int(episode_offset + env_idx) * 1_000_003
            + int(step_idx) * 97_409
            + int(prediction.query_step) * 9_176
        )
        records_per_env[env_idx].append(
            make_point_error_record(
                step_idx=step_idx,
                oracle_skill=oracle.get("skill"),
                vlm_skill=prediction.skill,
                oracle_point=oracle_point,
                vlm_point=prediction.point_px,
                query_step=prediction.query_step,
                oracle_point_3d=oracle_point_3d,
                camera_info=front_camera_info,
                noise_seed=noise_seed,
                noise_projection_samples=noise_projection_samples,
            )
        )


def rollout(
    env: Env,
    actor: Actor,
    rollout_max_steps: int,
    pbar: SuccessTqdm = None,
    resize_video: bool = True,
    n_parts_assemble: int = 1,
    save_rollouts: bool = False,
    pc_generator = None,
    annotate_skill: bool = False,
    annotate_guidance_point: bool = False,
    annotate_grasp: bool = False,
    guidance_point_on_image: bool = False,
    grasp_annotation_on_image: bool = False,
    grasp_part_annotate: bool = False,
    guidance_point_colored: bool = False,
    grasp_annotation_colored: bool = False,
    model_guidance_point_colored: bool = False,
    model_grasp_annotation_colored: bool = False,
    skill_on_image: bool = False,
    annotate_wrist_camera: bool = False,
    provide_skill_input: bool = False,
    collect_skill_stats: bool = False,
    enable_annotation_verify: bool = False,
    annotation_noise_config: Optional[AnnotationNoiseConfig] = None,
    rollout_after_success: int = 0,
    full_length_rollout: bool = False,
    perturb_runner: Optional[PerturbRunner] = None,
    init_states: Optional[List[dict]] = None,
    annotation_source: str = "scripted",
    vlm_client: Optional[VLMGuidanceClient] = None,
    vlm_query_interval: Optional[int] = None,
    vlm_metric_episode_offset: int = 0,
    vlm_noise_projection_samples: int = DEFAULT_MONTE_CARLO_SAMPLES_PER_PAIR,
    eepose_frame: str = ROBOT_BASE,
) -> Optional[RolloutSaveValues]:
    use_vlm = annotation_source == "vlm"
    if annotation_source not in {"scripted", "vlm"}:
        raise ValueError(f"unsupported annotation_source: {annotation_source}")
    if use_vlm and vlm_client is None:
        raise ValueError("annotation_source=vlm requires a VLM client")
    query_interval = None
    if use_vlm:
        query_interval = int(vlm_query_interval or actor.action_horizon)
        if query_interval <= 0:
            raise ValueError("vlm_query_interval must be positive")
    # get first observation
    with suppress_all_output(False):
        if init_states is not None:
            # Use provided init states instead of random reset.
            # init_states should already have exactly env.num_envs entries.
            env.reset_to(init_states)
            env.refresh()
            obs = env.get_observation()
        else:
            obs = env.reset()
        actor.reset()
    use_binary_gripper_width = (
        not getattr(actor, "expects_raw_robot_state", False)
        and normalizer_expects_binary_gripper_width(actor.normalizer)
    )
    collect_skill_annotations = (
        use_vlm
        or annotate_skill
        or annotate_guidance_point
        or guidance_point_on_image
        or grasp_annotation_on_image
        or grasp_part_annotate
        or provide_skill_input
        or collect_skill_stats
        or (
            perturb_runner is not None
            and perturb_runner.enabled
            and perturb_runner.requires_skill_annotations
        )
    )
    if collect_skill_annotations:
        reset_skill_annotator(env)

    video_obs = deepcopy(obs)
    previous_skills = [None] * env.num_envs
    oracle_initial_annotations = (
        get_annotation_bundle_all_envs(
            env,
            previous_skills=previous_skills,
            annotate_wrist_camera=annotate_wrist_camera,
            resize_images=(resize_video or use_vlm),
            enable_verify=enable_annotation_verify,
            annotation_noise_config=(None if use_vlm else annotation_noise_config),
        )
        if collect_skill_annotations
        else [{} for _ in range(env.num_envs)]
    )
    initial_annotations = oracle_initial_annotations
    initial_skills = [bundle.get("skill") for bundle in initial_annotations]
    initial_skill_states = [bundle.get("skill_state") for bundle in initial_annotations]
    initial_assembly_steps = [bundle.get("assembly_step") for bundle in initial_annotations]
    initial_guidance_points = [bundle.get("guidance_point") for bundle in initial_annotations]
    initial_guidance_points_clean = [
        bundle.get("guidance_point_clean") for bundle in initial_annotations
    ]
    initial_guidance_poses = [bundle.get("guidance_pose") for bundle in initial_annotations]
    initial_guidance_poses_clean = [
        bundle.get("guidance_pose_clean") for bundle in initial_annotations
    ]
    initial_guidance_gripper_widths = [
        bundle.get("guidance_gripper_width") for bundle in initial_annotations
    ]
    initial_guidance_points_2d = [
        bundle.get("guidance_point_2d", {}) for bundle in initial_annotations
    ]
    initial_grasp_annotations_2d = [
        bundle.get("grasp_annotation_2d", {}) for bundle in initial_annotations
    ]
    for env_idx, skill in enumerate(initial_skills):
        if skill is not None:
            previous_skills[env_idx] = skill
    if provide_skill_input and all(skill is None for skill in initial_skills):
        raise ValueError(
            f"Actor requires skill one-hot input, but no skill labels were produced for task `{env.furniture_name}`."
        )
    # Resize the images in the observation if they exist
    resize_image(obs, "color_image1")
    resize_crop_image(obs, "color_image2")
    # Resize the depth image
    resize_depth(obs, "depth_image1")
    resize_crop_depth(obs, "depth_image2")
    vlm_predictions: list[VLMPrediction] = []
    if use_vlm:
        initial_annotations, vlm_predictions = _query_vlm_annotations(
            client=vlm_client,
            env=env,
            obs=obs,
            oracle_bundles=oracle_initial_annotations,
            step_idx=0,
        )
        initial_skills = [bundle.get("skill") for bundle in initial_annotations]
        initial_guidance_points_2d = [
            bundle.get("guidance_point_2d", {}) for bundle in initial_annotations
        ]
    _apply_policy_visual_annotations(
        obs,
        initial_annotations,
        annotate_wrist_camera=annotate_wrist_camera,
        annotate_guidance_point=annotate_guidance_point,
        annotate_grasp=annotate_grasp,
        grasp_part_annotate=grasp_part_annotate,
        guidance_point_colored=model_guidance_point_colored,
        grasp_annotation_colored=model_grasp_annotation_colored,
    )
    _attach_skill_tensor_to_obs(obs, actor, initial_skills)

    if resize_video:
        resize_image(video_obs, "color_image1")
        resize_crop_image(video_obs, "color_image2")
        resize_depth(video_obs, "depth_image1")
        resize_crop_depth(video_obs, "depth_image2")

    if grasp_part_annotate:
        _draw_grasp_part_annotations_for_all_envs(
            video_obs,
            initial_annotations,
            annotate_wrist_camera=annotate_wrist_camera,
            guidance_point_colored=guidance_point_colored,
            grasp_annotation_colored=grasp_annotation_colored,
        )
    elif guidance_point_on_image:
        _draw_guidance_points_for_all_envs(
            video_obs, initial_annotations, annotate_wrist_camera=annotate_wrist_camera,
            guidance_point_colored=guidance_point_colored,
        )
    if not grasp_part_annotate and grasp_annotation_on_image:
        _draw_grasp_annotations_for_all_envs(
            video_obs, initial_annotations, annotate_wrist_camera=annotate_wrist_camera,
            grasp_annotation_colored=grasp_annotation_colored,
        )

    # save initial visualization and rewards
    video_obs["robot_state"] = _add_sim_local_ee_pose_to_robot_state(
        env, video_obs["robot_state"]
    )
    robot_states = [TensorDict(video_obs["robot_state"], batch_size=env.num_envs)]
    imgs1 = [] if "color_image1" not in video_obs else [video_obs["color_image1"].cpu()]
    imgs2 = [] if "color_image2" not in video_obs else [video_obs["color_image2"].cpu()]
    depth_image1 = [] if video_obs.get("depth_image1") is None else [video_obs["depth_image1"]]
    depth_image2 = [] if video_obs.get("depth_image2") is None else [video_obs["depth_image2"]]
    parts_poses = [video_obs["parts_poses"].cpu()]
    skills = [initial_skills]
    skill_states = [initial_skill_states]
    assembly_steps = [initial_assembly_steps]
    guidance_points = [initial_guidance_points]
    guidance_points_clean = [initial_guidance_points_clean]
    guidance_poses = [initial_guidance_poses]
    guidance_poses_clean = [initial_guidance_poses_clean]
    guidance_gripper_widths = [initial_guidance_gripper_widths]
    guidance_points_2d = [initial_guidance_points_2d]
    grasp_annotations_2d = [initial_grasp_annotations_2d]
    camera_infos = [[bundle.get("camera_info", {}) for bundle in initial_annotations]]
    oracle_skills = [
        [bundle.get("skill") for bundle in oracle_initial_annotations]
    ]
    oracle_guidance_points_2d = [
        [bundle.get("guidance_point_2d", {}) for bundle in oracle_initial_annotations]
    ]
    vlm_annotations = [
        [bundle.get("vlm_annotation") for bundle in initial_annotations]
    ]
    vlm_point_error_records = [[] for _ in range(env.num_envs)]
    vlm_metric_active = [True] * env.num_envs
    current_oracle_annotations = oracle_initial_annotations
    current_annotations = initial_annotations
    active_skill_states = initial_skill_states

    # Verify history for summary at end of rollout
    from src.eval.skill_annotation_verify import VerifyHistory, verify_and_record
    _verify_history = VerifyHistory(furniture_name=getattr(env, "furniture_name", ""))
    for bundle in initial_annotations:
        verify_and_record(
            bundle, _verify_history,
            step_idx=0,
            assembly_step=bundle.get("assembly_step", ""),
            skill=bundle.get("skill", ""),
        )
    actions = list()
    rewards = torch.zeros((env.num_envs, rollout_max_steps), dtype=torch.float32)
    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")
    success_stop_step = torch.full(
        (env.num_envs, 1), -1, dtype=torch.int64, device="cuda"
    )
    
    # Collect point clouds if pc_generator is provided
    point_clouds = []  # List of lists: [[env0_step0, env1_step0, ...], [env0_step1, ...]]
    if pc_generator is not None:
        pcs_step = pc_generator.generate_transformed_cropped_point_cloud_for_all_env()
        # Add point cloud to obs for actor
        if len(pcs_step) > 0:
            obs["point_cloud"] = torch.stack(pcs_step)
            
        pcs_step_np = []
        for env_idx, pc in enumerate(pcs_step):
            pc_np = pc.detach().cpu().numpy()
            pcs_step_np.append(pc_np)
        point_clouds.append(pcs_step_np)

    step_idx = 0
    apply_ee_force = None
    perturb_device = None
    if perturb_runner is not None and perturb_runner.enabled:
        perturb_device = getattr(env, "device", actor.device)
        if not isinstance(perturb_device, torch.device):
            perturb_device = torch.device(perturb_device)
        if perturb_runner.applies_force:
            apply_ee_force = getattr(env, "apply_end_effector_force", None)
            if not callable(apply_ee_force):
                raise ValueError(
                    f"Perturb mode `{perturb_runner.mode}` requires an environment with "
                    "`apply_end_effector_force`. This is currently supported for "
                    "FurnitureRLSimEnv only."
                )
        perturb_runner.reset_episode(env.num_envs, perturb_device)

    # TODO - figure out how to fix this
    actor.normalizer = actor.normalizer.to(actor.device)
    actor.model = actor.model.to(actor.device)

    while True:
        if use_vlm:
            _record_vlm_point_errors(
                vlm_point_error_records,
                current_oracle_annotations,
                vlm_predictions,
                vlm_metric_active,
                step_idx=step_idx,
                episode_offset=vlm_metric_episode_offset,
                noise_projection_samples=vlm_noise_projection_samples,
            )
        raw_robot_state = obs.get("robot_state") if isinstance(obs, dict) else None
        if isinstance(raw_robot_state, dict):
            raw_robot_state = _add_sim_local_ee_pose_to_robot_state(
                env, raw_robot_state
            )
        ee_pos_vel = (
            raw_robot_state.get("ee_pos_vel")
            if isinstance(raw_robot_state, dict)
            else None
        )

        # Keep the policy tensor layout stable while selecting either the new
        # robot-base pose or a legacy EE representation.
        if isinstance(raw_robot_state, dict):
            policy_robot_state = select_policy_eepose(
                raw_robot_state,
                eepose_frame,
                original_frame=SIM_LOCAL,
            )
            if use_binary_gripper_width:
                policy_robot_state = binarize_robot_state_gripper_width(
                    policy_robot_state
                )
            obs["robot_state"] = policy_robot_state

        # Convert from robot state dict to robot state tensor
        if not getattr(actor, "expects_raw_robot_state", False):
            obs["robot_state"] = env.filter_and_concat_robot_state(obs["robot_state"])

        if perturb_runner is not None and perturb_runner.enabled:
            perturb_ctx = PerturbContext(
                step_idx=step_idx,
                num_envs=env.num_envs,
                device=perturb_device,
                furniture_name=getattr(env, "furniture_name", None),
                task_name=getattr(env, "task_name", None),
                skill_states=active_skill_states,
                ee_pos_vel=ee_pos_vel,
            )
            if perturb_runner.subdivides_action:
                actor.subdivide_ratio = perturb_runner.get_subdivide_ratio(perturb_ctx)
            if perturb_runner.applies_force:
                assert apply_ee_force is not None
                perturb_forces = perturb_runner.compute_force(perturb_ctx)
                apply_ee_force(perturb_forces)
        else:
            actor.subdivide_ratio = 1.0

        # Get the next actions from the actor
        action_pred = actor.action(obs)
        if isinstance(raw_robot_state, dict):
            action_pred = _policy_action_to_robot_base(
                env,
                action_pred,
                raw_robot_state,
                eepose_frame,
            )

        # Mutate the action vector for action-modifying perturb modes
        # (e.g. place_drop forces the gripper open during the place skill).
        if perturb_runner is not None and perturb_runner.modifies_action:
            assert perturb_ctx is not None
            action_pred = perturb_runner.modify_action(action_pred, perturb_ctx)

        obs, reward, done, _ = env.step(action_pred, sample_perturbations=False)

        # Generate point clouds for the new observation
        if pc_generator is not None:
            pcs_step = pc_generator.generate_transformed_cropped_point_cloud_for_all_env()
            if len(pcs_step) > 0:
                obs["point_cloud"] = torch.stack(pcs_step)
        else:
            pcs_step = None

        video_obs = deepcopy(obs)
        current_oracle_annotations = (
            get_annotation_bundle_all_envs(
                env,
                previous_skills=previous_skills,
                annotate_wrist_camera=annotate_wrist_camera,
                resize_images=(resize_video or use_vlm),
                enable_verify=enable_annotation_verify,
                annotation_noise_config=(None if use_vlm else annotation_noise_config),
            )
            if collect_skill_annotations
            else [{} for _ in range(env.num_envs)]
        )
        oracle_current_skills = [
            bundle.get("skill") for bundle in current_oracle_annotations
        ]
        current_skill_states = [bundle.get("skill_state") for bundle in current_oracle_annotations]
        current_assembly_steps = [bundle.get("assembly_step") for bundle in current_oracle_annotations]
        current_guidance_points = [bundle.get("guidance_point") for bundle in current_oracle_annotations]
        current_guidance_points_clean = [
            bundle.get("guidance_point_clean") for bundle in current_oracle_annotations
        ]
        current_guidance_poses = [bundle.get("guidance_pose") for bundle in current_oracle_annotations]
        current_guidance_poses_clean = [
            bundle.get("guidance_pose_clean") for bundle in current_oracle_annotations
        ]
        current_guidance_gripper_widths = [
            bundle.get("guidance_gripper_width") for bundle in current_oracle_annotations
        ]
        current_grasp_annotations_2d = [
            bundle.get("grasp_annotation_2d", {}) for bundle in current_oracle_annotations
        ]
        for env_idx, skill in enumerate(oracle_current_skills):
            if skill is not None:
                previous_skills[env_idx] = skill
        # Record verify results for end-of-rollout summary
        for bundle in current_oracle_annotations:
            verify_and_record(
                bundle, _verify_history,
                step_idx=step_idx + 1,
                assembly_step=bundle.get("assembly_step", ""),
                skill=bundle.get("skill", ""),
            )

        # Resize the images in the observation if they exist
        resize_image(obs, "color_image1")
        resize_crop_image(obs, "color_image2")
        resize_depth(obs, "depth_image1")
        resize_crop_depth(obs, "depth_image2")
        next_step_idx = step_idx + 1
        if use_vlm:
            assert query_interval is not None
            if next_step_idx % query_interval == 0:
                current_annotations, vlm_predictions = _query_vlm_annotations(
                    client=vlm_client,
                    env=env,
                    obs=obs,
                    oracle_bundles=current_oracle_annotations,
                    step_idx=next_step_idx,
                )
            else:
                current_annotations = policy_bundles_from_vlm(
                    current_oracle_annotations,
                    vlm_predictions,
                    step_idx=next_step_idx,
                )
        else:
            current_annotations = current_oracle_annotations
        current_skills = [bundle.get("skill") for bundle in current_annotations]
        current_guidance_points_2d = [
            bundle.get("guidance_point_2d", {}) for bundle in current_annotations
        ]
        _apply_policy_visual_annotations(
            obs,
            current_annotations,
            annotate_wrist_camera=annotate_wrist_camera,
            annotate_guidance_point=annotate_guidance_point,
            annotate_grasp=annotate_grasp,
            grasp_part_annotate=grasp_part_annotate,
            guidance_point_colored=model_guidance_point_colored,
            grasp_annotation_colored=model_grasp_annotation_colored,
        )
        _attach_skill_tensor_to_obs(obs, actor, current_skills)

        # Save observations for the policy
        if resize_video:
            resize_image(video_obs, "color_image1")
            resize_crop_image(video_obs, "color_image2")
            resize_depth(video_obs, "depth_image1")
            resize_crop_depth(video_obs, "depth_image2")

        if grasp_part_annotate:
            _draw_grasp_part_annotations_for_all_envs(
                video_obs,
                current_annotations,
                annotate_wrist_camera=annotate_wrist_camera,
                guidance_point_colored=guidance_point_colored,
                grasp_annotation_colored=grasp_annotation_colored,
            )
        elif guidance_point_on_image:
            _draw_guidance_points_for_all_envs(
                video_obs, current_annotations, annotate_wrist_camera=annotate_wrist_camera,
                guidance_point_colored=guidance_point_colored,
            )
        if not grasp_part_annotate and grasp_annotation_on_image:
            _draw_grasp_annotations_for_all_envs(
                video_obs, current_annotations, annotate_wrist_camera=annotate_wrist_camera,
                grasp_annotation_colored=grasp_annotation_colored,
            )

        skills.append(current_skills)
        skill_states.append(current_skill_states)
        assembly_steps.append(current_assembly_steps)
        oracle_skills.append(oracle_current_skills)
        oracle_guidance_points_2d.append(
            [
                bundle.get("guidance_point_2d", {})
                for bundle in current_oracle_annotations
            ]
        )
        vlm_annotations.append(
            [bundle.get("vlm_annotation") for bundle in current_annotations]
        )
        active_skill_states = current_skill_states

        # Store the results for visualization and logging
        if save_rollouts or collect_skill_stats:
            video_obs["robot_state"] = _add_sim_local_ee_pose_to_robot_state(
                env, video_obs["robot_state"]
            )
            robot_states.append(
                TensorDict(video_obs["robot_state"], batch_size=env.num_envs)
            )
        append_tracking_annotation_histories(
            save_rollouts=save_rollouts,
            collect_skill_stats=collect_skill_stats,
            histories=(
                guidance_points,
                guidance_points_clean,
                guidance_poses,
                guidance_poses_clean,
                guidance_gripper_widths,
            ),
            current_values=(
                current_guidance_points,
                current_guidance_points_clean,
                current_guidance_poses,
                current_guidance_poses_clean,
                current_guidance_gripper_widths,
            ),
        )
        if save_rollouts:
            if "color_image1" in video_obs:
                imgs1.append(video_obs["color_image1"].cpu())
            if "color_image2" in video_obs:
                imgs2.append(video_obs["color_image2"].cpu())
            if video_obs.get("depth_image1") is not None:
                depth_image1.append(video_obs["depth_image1"])
            if video_obs.get("depth_image2") is not None:
                depth_image2.append(video_obs["depth_image2"])
            actions.append(action_pred.cpu())
            parts_poses.append(video_obs["parts_poses"].cpu())
            guidance_points_2d.append(current_guidance_points_2d)
            grasp_annotations_2d.append(current_grasp_annotations_2d)
            camera_infos.append([bundle.get("camera_info", {}) for bundle in current_annotations])

            # Collect point clouds at each step
            if pcs_step is not None:
                pcs_step_np = []
                for env_idx, pc in enumerate(pcs_step):
                    pc_np = pc.detach().cpu().numpy()
                    pcs_step_np.append(pc_np)
                point_clouds.append(pcs_step_np)

        # Always store rewards as they are used to calculate success
        rewards[:, step_idx] = reward.squeeze().cpu()

        # update progress bar
        step_idx += 1
        if pbar is not None:
            pbar.set_postfix(step=step_idx)
            n_success = (rewards.sum(dim=1) == n_parts_assemble).sum().item()
            pbar.pbar_desc(n_success)
            pbar.update()

        if step_idx >= rollout_max_steps:
            done = torch.ones((env.num_envs, 1), dtype=torch.bool, device="cuda")

        done_for_break = done
        if rollout_after_success > 0 and not full_length_rollout:
            current_success = (
                rewards[:, :step_idx].sum(dim=1, keepdim=True) >= n_parts_assemble
            ).to(done.device)
            new_success = current_success & (success_stop_step < 0)
            success_stop_step[new_success] = step_idx + rollout_after_success
            delayed_success_done = current_success & (step_idx >= success_stop_step)
            done_for_break = torch.where(current_success, delayed_success_done, done)

        if not full_length_rollout:
            flattened_done = done_for_break.reshape(-1).detach().cpu().tolist()
            vlm_metric_active = [not bool(value) for value in flattened_done]

        if done_for_break.all() and not full_length_rollout:
            break

        if step_idx >= rollout_max_steps:
            break

    # Reorganize point_clouds from [step][env] to [env][step]
    if pc_generator is not None and point_clouds:
        # point_clouds is [[env0_s0, env1_s0, ...], [env0_s1, env1_s1, ...], ...]
        # Convert to [[env0_s0, env0_s1, ...], [env1_s0, env1_s1, ...], ...]
        num_steps = len(point_clouds)
        num_envs = len(point_clouds[0]) if point_clouds else 0
        pcs_per_env = []
        for env_idx in range(num_envs):
            pcs_per_env.append([point_clouds[step][env_idx] for step in range(num_steps)])
    else:
        pcs_per_env = None

    skills_per_env = _transpose_step_env_annotations(skills, env.num_envs)
    skill_states_per_env = _transpose_step_env_annotations(skill_states, env.num_envs)
    assembly_steps_per_env = _transpose_step_env_annotations(assembly_steps, env.num_envs)
    guidance_points_per_env = _transpose_step_env_annotations(guidance_points, env.num_envs)
    guidance_points_clean_per_env = _transpose_step_env_annotations(
        guidance_points_clean, env.num_envs
    )
    guidance_poses_per_env = _transpose_step_env_annotations(guidance_poses, env.num_envs)
    guidance_poses_clean_per_env = _transpose_step_env_annotations(
        guidance_poses_clean, env.num_envs
    )
    guidance_gripper_widths_per_env = _transpose_step_env_annotations(
        guidance_gripper_widths, env.num_envs
    )
    guidance_points_2d_per_env = _transpose_step_env_annotations(
        guidance_points_2d, env.num_envs
    )
    grasp_annotations_2d_per_env = _transpose_step_env_annotations(
        grasp_annotations_2d, env.num_envs
    )
    camera_infos_per_env = _transpose_step_env_annotations(camera_infos, env.num_envs)
    oracle_skills_per_env = _transpose_step_env_annotations(oracle_skills, env.num_envs)
    oracle_guidance_points_2d_per_env = _transpose_step_env_annotations(
        oracle_guidance_points_2d, env.num_envs
    )
    vlm_annotations_per_env = _transpose_step_env_annotations(
        vlm_annotations, env.num_envs
    )

    # --- verify summary ---
    if enable_annotation_verify:
        print(_verify_history.summary(), flush=True)

    return RolloutSaveValues(
        torch.stack(robot_states, dim=1) if robot_states else [],
        torch.stack(imgs1, dim=1) if imgs1 else [],
        torch.stack(imgs2, dim=1) if imgs2 else [],
        torch.stack(actions, dim=1) if actions else [],
        rewards,
        torch.stack(parts_poses, dim=1) if parts_poses else [],
        pcs_per_env,
        torch.stack(depth_image1, dim=1) if depth_image1 else [],
        torch.stack(depth_image2, dim=1) if depth_image2 else [],
        skills_per_env,
        skill_states_per_env,
        assembly_steps_per_env,
        guidance_points_per_env,
        guidance_points_clean_per_env,
        guidance_poses_per_env,
        guidance_poses_clean_per_env,
        guidance_gripper_widths_per_env,
        guidance_points_2d_per_env,
        grasp_annotations_2d_per_env,
        camera_infos_per_env,
        oracle_skills_per_env,
        oracle_guidance_points_2d_per_env,
        vlm_annotations_per_env,
        vlm_point_error_records,
    )


@torch.no_grad()
def calculate_success_rate(
    env: Env,
    actor: Actor,
    n_rollouts: int,
    rollout_max_steps: int,
    epoch_idx: int,
    discount: float = 0.99,
    rollout_save_dir: Optional[Path] = None,
    save_rollouts_to_wandb: bool = False,
    save_failures: bool = False,
    n_parts_assemble: Optional[int] = None,
    compress_pickles: bool = False,
    resize_video: bool = True,
    n_steps_padding: int = 30,
    break_on_n_success: bool = False,
    stop_after_n_success: int = 0,
    rollout_after_success: int = 0,
    record_first_state_only: bool = False,
    pc_generator = None,
    annotate_skill: bool = False,
    annotate_guidance_point: bool = False,
    annotate_grasp: bool = False,
    guidance_point_on_image: bool = False,
    grasp_annotation_on_image: bool = False,
    grasp_part_annotate: bool = False,
    guidance_point_colored: bool = False,
    grasp_annotation_colored: bool = False,
    model_guidance_point_colored: bool = False,
    model_grasp_annotation_colored: bool = False,
    skill_on_image: bool = False,
    annotate_wrist_camera: bool = False,
    provide_skill_input: bool = False,
    collect_skill_stats: bool = False,
    enable_annotation_verify: bool = False,
    annotation_noise_config: Optional[AnnotationNoiseConfig] = None,
    full_length_rollout: bool = False,
    output_only_pickle: bool = False,
    output_only_video: bool = False,
    perturb_runner: Optional[PerturbRunner] = None,
    target_successes: Optional[int] = None,
    init_states: Optional[List[dict]] = None,
    max_saved_rollouts: Optional[int] = None,
    guidance_bank_out: Optional[Path] = None,
    annotation_source: str = "scripted",
    vlm_client: Optional[VLMGuidanceClient] = None,
    vlm_query_interval: Optional[int] = None,
    tracking_metric_type: Optional[str] = None,
    vlm_noise_projection_samples: int = DEFAULT_MONTE_CARLO_SAMPLES_PER_PAIR,
    eepose_frame: str = ROBOT_BASE,
) -> RolloutStats:

    use_target_mode = target_successes is not None and target_successes > 0

    pbar = SuccessTqdm(
        num_envs=env.num_envs,
        n_rollouts=n_rollouts,
        task_name=env.task_name,
        total=rollout_max_steps * (n_rollouts // env.num_envs),
        desc="Performing rollouts",
        leave=True,
        unit="step",
    )
    if use_target_mode:
        pbar.target_only = True

    if n_parts_assemble is None:
        n_parts_assemble = env.n_parts_assemble

    tbl = wandb.Table(
        columns=["rollout", "success", "epoch", "reward", "return", "steps"]
    )

    n_success = 0
    n_total_rollouts = 0
    total_reward = 0
    episode_returns = []
    table_rows = []
    state_counts: dict[str, int] = {}
    skill_completion_counts: dict[str, int] = {}
    step_counts: dict[str, int] = {}
    step_completion_counts: dict[str, int] = {}
    tracking_error_records: dict[str, list[dict[str, float]]] = {}
    tracking_workspace_counts = new_tracking_workspace_counts()
    if tracking_metric_type is None:
        tracking_metric_type = (
            "pose" if grasp_part_annotate or annotate_grasp else "position"
        )
    if tracking_metric_type not in {"position", "pose"}:
        raise ValueError(f"Unsupported tracking metric type: {tracking_metric_type}")
    vlm_point_error_summaries: list[dict] = []
    vlm_model_revisions: set[str] = set()
    tracking_episode_count = 0
    tracking_incomplete_episode_count = 0
    saved_rollouts_count = 0
    guidance_bank_records: list[dict] = []

    save_rollouts = rollout_save_dir is not None or save_rollouts_to_wandb

    # For record_first_state_only
    if record_first_state_only:
        first_robot_states = []
        first_part_poses = []
        first_success = []

    pbar.pbar_desc(n_success)
    while True:
        if not use_target_mode and n_total_rollouts >= n_rollouts:
            break
        if use_target_mode and n_success >= target_successes:
            break

        # Update the progress bar
        pbar.before_round(n_success)

        # Slice init_states for this round to avoid using the same state every time
        round_init_states = None
        if init_states is not None:
            start = n_total_rollouts % len(init_states)
            end = start + env.num_envs
            round_init_states = [
                init_states[i % len(init_states)] for i in range(start, end)
            ]

        # Perform a rollout with the current model
        save_rollouts_this_round = save_rollouts and (
            max_saved_rollouts is None or saved_rollouts_count < max_saved_rollouts
        )

        rollout_data: RolloutSaveValues = rollout(
            env,
            actor,
            rollout_max_steps,
            pbar=pbar,
            resize_video=resize_video,
            n_parts_assemble=n_parts_assemble,
            save_rollouts=save_rollouts_this_round,
            pc_generator=pc_generator,
            annotate_skill=annotate_skill,
            annotate_guidance_point=annotate_guidance_point,
            annotate_grasp=annotate_grasp,
            guidance_point_on_image=guidance_point_on_image,
            grasp_annotation_on_image=grasp_annotation_on_image,
            grasp_part_annotate=grasp_part_annotate,
            guidance_point_colored=guidance_point_colored,
            grasp_annotation_colored=grasp_annotation_colored,
            model_guidance_point_colored=model_guidance_point_colored,
            model_grasp_annotation_colored=model_grasp_annotation_colored,
            skill_on_image=skill_on_image,
            annotate_wrist_camera=annotate_wrist_camera,
            provide_skill_input=provide_skill_input,
            collect_skill_stats=collect_skill_stats,
            enable_annotation_verify=enable_annotation_verify,
            annotation_noise_config=annotation_noise_config,
            rollout_after_success=rollout_after_success,
            full_length_rollout=full_length_rollout,
            perturb_runner=perturb_runner,
            init_states=round_init_states,
            annotation_source=annotation_source,
            vlm_client=vlm_client,
            vlm_query_interval=vlm_query_interval,
            vlm_metric_episode_offset=n_total_rollouts,
            vlm_noise_projection_samples=vlm_noise_projection_samples,
            eepose_frame=eepose_frame,
        )

        # Calculate the success rate
        success_flags = rollout_data.rewards.sum(dim=1) == n_parts_assemble
        for env_idx in range(env.num_envs):
            rewards_for_stats = rollout_data.rewards[env_idx].numpy()
            episode_returns.append(
                np.sum(rewards_for_stats * discount ** np.arange(len(rewards_for_stats)))
            )
            total_reward += np.sum(rewards_for_stats)
        if annotation_source == "vlm":
            for episode in rollout_data.vlm_annotations:
                for annotation in episode:
                    if isinstance(annotation, dict) and annotation.get("model_revision"):
                        vlm_model_revisions.add(str(annotation["model_revision"]))
            vlm_point_error_summaries.append(
                build_vlm_point_error_summary(
                    rollout_data.vlm_point_error_records,
                    [bool(value) for value in success_flags.tolist()],
                )
            )
        n_success += success_flags.sum().item()
        n_total_rollouts += env.num_envs

        for env_idx in range(env.num_envs):
            robot_states_for_tracking = []
            if collect_skill_stats and rollout_data.robot_states is not None:
                robot_states_for_tracking = tensordict_to_list_of_dicts(
                    rollout_data.robot_states[env_idx]
                )
            accumulate_episode_skill_stats(
                state_labels=(
                    rollout_data.skill_states[env_idx] if rollout_data.skill_states else []
                ),
                step_labels=(
                    rollout_data.assembly_steps[env_idx]
                    if rollout_data.assembly_steps
                    else []
                ),
                success=bool(success_flags[env_idx].item()),
                state_counts=state_counts,
                skill_completion_counts=skill_completion_counts,
                step_counts=step_counts,
                step_completion_counts=step_completion_counts,
            )
            if guidance_bank_out is not None:
                episode_idx = n_total_rollouts - env.num_envs + env_idx
                guidance_bank_records.extend(
                    _guidance_bank_records_for_episode(
                        task=str(
                            getattr(
                                env,
                                "furniture_name",
                                getattr(env, "task_name", ""),
                            )
                        ),
                        source_episode=episode_idx,
                        skill_states=(
                            rollout_data.skill_states[env_idx]
                            if rollout_data.skill_states
                            else []
                        ),
                        skills=(
                            rollout_data.skills[env_idx]
                            if rollout_data.skills
                            else []
                        ),
                        guidance_points=(
                            rollout_data.guidance_points_clean[env_idx]
                            if rollout_data.guidance_points_clean
                            else []
                        ),
                        guidance_poses=(
                            rollout_data.guidance_poses_clean[env_idx]
                            if rollout_data.guidance_poses_clean
                            else []
                        ),
                        guidance_gripper_widths=(
                            rollout_data.guidance_gripper_widths[env_idx]
                            if rollout_data.guidance_gripper_widths
                            else []
                        ),
                    )
                )
            if collect_skill_stats and robot_states_for_tracking:
                skill_states_for_tracking = (
                    rollout_data.skill_states[env_idx]
                    if rollout_data.skill_states
                    else []
                )
                guidance_poses_for_tracking = (
                    rollout_data.guidance_poses_clean[env_idx]
                    if rollout_data.guidance_poses_clean
                    else []
                )
                if tracking_histories_are_complete(
                    robot_states_for_tracking,
                    skill_states_for_tracking,
                    guidance_poses_for_tracking,
                ):
                    tracking_episode_count += 1
                    accumulate_tracking_error_records(
                        tracking_error_records,
                        compute_episode_tracking_errors(
                            robot_states_for_tracking,
                            skill_states_for_tracking,
                            guidance_poses_for_tracking,
                            metric_type=tracking_metric_type,
                            workspace_counts=tracking_workspace_counts,
                        ),
                    )
                else:
                    tracking_incomplete_episode_count += 1

        # Save the results from the rollout immediately
        if save_rollouts_this_round:
            have_img_obs = rollout_data.imgs1 is not None and len(rollout_data.imgs1) > 0
            have_depth_obs = rollout_data.depth_image1 is not None and len(rollout_data.depth_image1) > 0

            for env_idx in range(env.num_envs):
                robot_states = tensordict_to_list_of_dicts(rollout_data.robot_states[env_idx])
                actions = rollout_data.actions[env_idx].numpy()
                rewards = rollout_data.rewards[env_idx].numpy()
                parts_poses = rollout_data.parts_poses[env_idx].numpy()
                skills = rollout_data.skills[env_idx] if rollout_data.skills else []
                guidance_points = (
                    rollout_data.guidance_points[env_idx]
                    if rollout_data.guidance_points
                    else []
                )
                guidance_points_clean = (
                    rollout_data.guidance_points_clean[env_idx]
                    if rollout_data.guidance_points_clean
                    else []
                )
                guidance_poses = (
                    rollout_data.guidance_poses[env_idx]
                    if rollout_data.guidance_poses
                    else []
                )
                guidance_poses_clean = (
                    rollout_data.guidance_poses_clean[env_idx]
                    if rollout_data.guidance_poses_clean
                    else []
                )
                guidance_gripper_widths = (
                    rollout_data.guidance_gripper_widths[env_idx]
                    if rollout_data.guidance_gripper_widths
                    else []
                )
                guidance_points_2d = (
                    rollout_data.guidance_points_2d[env_idx]
                    if rollout_data.guidance_points_2d
                    else []
                )
                grasp_annotations_2d = (
                    rollout_data.grasp_annotations_2d[env_idx]
                    if rollout_data.grasp_annotations_2d
                    else []
                )
                camera_infos = (
                    rollout_data.camera_infos[env_idx]
                    if rollout_data.camera_infos
                    else []
                )
                oracle_skills_for_rollout = (
                    rollout_data.oracle_skills[env_idx]
                    if rollout_data.oracle_skills
                    else []
                )
                oracle_guidance_points_2d_for_rollout = (
                    rollout_data.oracle_guidance_points_2d[env_idx]
                    if rollout_data.oracle_guidance_points_2d
                    else []
                )
                vlm_annotations_for_rollout = (
                    rollout_data.vlm_annotations[env_idx]
                    if rollout_data.vlm_annotations
                    else []
                )
                vlm_point_error_records_for_rollout = (
                    rollout_data.vlm_point_error_records[env_idx]
                    if rollout_data.vlm_point_error_records
                    else []
                )
                vlm_model_revision = next(
                    (
                        annotation.get("model_revision")
                        for annotation in vlm_annotations_for_rollout
                        if isinstance(annotation, dict)
                        and annotation.get("model_revision")
                    ),
                    None,
                )
                success = success_flags[env_idx].item()
                task = env.furniture_name
                
                # Get point clouds for this env (list of arrays per step)
                pcs_for_rollout = rollout_data.point_clouds[env_idx] if rollout_data.point_clouds is not None else None

                if record_first_state_only:
                    first_robot_states.append(robot_states[0])
                    first_part_poses.append(parts_poses[0])
                    first_success.append(success)
                    continue

                video1 = (
                    rollout_data.imgs1[env_idx].numpy()
                    if have_img_obs
                    else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8)
                )
                video2 = (
                    rollout_data.imgs2[env_idx].numpy()
                    if have_img_obs
                    else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8)
                )
                video2_for_video = video2.copy()
                if annotate_skill and skill_on_image:
                    n_annotated = min(len(video2_for_video), len(skills))
                    for frame_idx in range(n_annotated):
                        skill = skills[frame_idx]
                        if skill is None:
                            continue
                        video2_for_video[frame_idx] = draw_skill_on_image(
                            video2_for_video[frame_idx], skill
                        )
                depth_video1 = (
                    rollout_data.depth_image1[env_idx].cpu().numpy()
                    if have_depth_obs
                    else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8)
                )
                depth_video2 = (
                    rollout_data.depth_image2[env_idx].cpu().numpy()
                    if have_depth_obs
                    else np.zeros((len(robot_states), 2, 2, 3), dtype=np.uint8)
                )

                # Number of steps until success
                if full_length_rollout:
                    n_steps = rollout_max_steps
                else:
                    n_steps = (
                        np.where(rewards == 1)[0][-1] + 1
                        if success
                        else rollout_max_steps
                    )
                    n_steps += n_steps_padding
                trim_start_steps = 0

                # Stack the two videos side by side
                if have_img_obs:
                    video = np.concatenate([video1, video2_for_video], axis=2)[
                        trim_start_steps:n_steps
                    ]
                    video = create_in_memory_mp4(video, fps=20)

                if save_rollouts_to_wandb and have_img_obs:
                    table_rows.append(
                        [
                            wandb.Video(video, fps=20, format="mp4"),
                            success,
                            epoch_idx,
                            np.sum(rewards),
                            episode_return,
                            n_steps,
                        ]
                    )

                should_save_rollout = (
                    rollout_save_dir is not None
                    and (save_failures or success)
                    and (
                        max_saved_rollouts is None
                        or saved_rollouts_count < max_saved_rollouts
                    )
                )
                if should_save_rollout:
                    point_error_records_to_save = None
                    if annotation_source == "vlm":
                        point_error_records_to_save = []
                        for record in vlm_point_error_records_for_rollout:
                            source_step = int(record["step_idx"])
                            if not trim_start_steps <= source_step < n_steps:
                                continue
                            source_query_step = int(record["query_step"])
                            rebased = dict(record)
                            rebased["source_step_idx"] = source_step
                            rebased["source_query_step"] = source_query_step
                            rebased["step_idx"] = source_step - trim_start_steps
                            rebased["query_step"] = (
                                source_query_step - trim_start_steps
                            )
                            point_error_records_to_save.append(rebased)
                    # Trim point clouds to match n_steps
                    pcs_trimmed = None
                    if pcs_for_rollout is not None:
                        pcs_trimmed = pcs_for_rollout[trim_start_steps : n_steps + 1]
                    save_raw_rollout(
                        robot_states=robot_states[trim_start_steps : n_steps + 1],
                        imgs1=video1[trim_start_steps : n_steps + 1],
                        imgs2=video2[trim_start_steps : n_steps + 1],
                        depth_image1=depth_video1[trim_start_steps : n_steps + 1],
                        depth_image2=depth_video2[trim_start_steps : n_steps + 1],
                        parts_poses=parts_poses[trim_start_steps : n_steps + 1],
                        skills=skills[trim_start_steps : n_steps + 1],
                        guidance_points=guidance_points[trim_start_steps : n_steps + 1],
                        guidance_points_clean=guidance_points_clean[trim_start_steps : n_steps + 1],
                        guidance_poses=guidance_poses[trim_start_steps : n_steps + 1],
                        guidance_poses_clean=guidance_poses_clean[trim_start_steps : n_steps + 1],
                        guidance_gripper_widths=guidance_gripper_widths[trim_start_steps : n_steps + 1],
                        guidance_points_2d=guidance_points_2d[trim_start_steps : n_steps + 1],
                        grasp_annotations_2d=grasp_annotations_2d[trim_start_steps : n_steps + 1],
                        camera_infos=camera_infos[trim_start_steps : n_steps + 1],
                        actions=actions[trim_start_steps:n_steps],
                        rewards=rewards[trim_start_steps:n_steps],
                        success=success,
                        task=task,
                        action_type=env.action_type,
                        rollout_save_dir=rollout_save_dir,
                        compress_pickles=compress_pickles,
                        have_img_obs=have_img_obs,
                        have_depth_obs=have_depth_obs,
                        pcs=pcs_trimmed,
                        skill_on_image=skill_on_image,
                        output_only_pickle=output_only_pickle,
                        output_only_video=output_only_video,
                        oracle_skills=(
                            oracle_skills_for_rollout[trim_start_steps : n_steps + 1]
                            if annotation_source == "vlm"
                            else None
                        ),
                        oracle_guidance_points_2d=(
                            oracle_guidance_points_2d_for_rollout[
                                trim_start_steps : n_steps + 1
                            ]
                            if annotation_source == "vlm"
                            else None
                        ),
                        vlm_annotations=(
                            vlm_annotations_for_rollout[trim_start_steps : n_steps + 1]
                            if annotation_source == "vlm"
                            else None
                        ),
                        vlm_point_error_records=point_error_records_to_save,
                        annotation_source=annotation_source,
                        image_annotation_mode=_saved_pickle_image_annotation_mode(
                            guidance_point_on_image=guidance_point_on_image,
                            grasp_annotation_on_image=grasp_annotation_on_image,
                            grasp_part_annotate=grasp_part_annotate,
                            guidance_point_colored=guidance_point_colored,
                            grasp_annotation_colored=grasp_annotation_colored,
                            skill_on_image=skill_on_image,
                        ),
                        vlm_model_revision=vlm_model_revision,
                        eepose_frame=ROBOT_BASE,
                        eepose_original_frame=SIM_LOCAL,
                        policy_eepose_frame=eepose_frame,
                        guidance_frame=ROBOT_BASE,
                    )
                    saved_rollouts_count += 1

        if break_on_n_success and n_success >= stop_after_n_success:
            print(
                f"Current number of success {n_success} greater than breaking threshold {stop_after_n_success}. Breaking"
            )
            break

    # Handle record_first_state_only after all rollouts
    if record_first_state_only and rollout_save_dir is not None:
        first_state_npz = str(rollout_save_dir / "first_states.npz")
        print(f"Saving first states to: {first_state_npz}")
        np.savez(
            first_state_npz,
            robot_states=np.asarray(first_robot_states),
            part_poses=np.asarray(first_part_poses),
            success=np.asarray(first_success),
        )

    # Handle wandb table after all rollouts
    if save_rollouts_to_wandb and table_rows:
        table_rows = sorted(table_rows, key=lambda x: x[4], reverse=True)
        for row in table_rows:
            tbl.add_data(*row)
        if wandb.run is not None:
            wandb.log(
                {
                    "rollouts": tbl,
                    "epoch": epoch_idx,
                }
            )

    if guidance_bank_out is not None:
        write_guidance_shuffle_bank(
            guidance_bank_out,
            task=str(
                getattr(env, "furniture_name", getattr(env, "task_name", ""))
            ),
            records=guidance_bank_records,
        )

    pbar.close()
    if perturb_runner is not None and perturb_runner.enabled:
        print(f"Perturbation stats: {perturb_runner.stats.summary()}")

    state_counts = normalize_progress_counts(
        state_counts,
        get_task_progress_labels(getattr(env, "furniture_name", None), "skill_states"),
    )
    skill_completion_counts = normalize_progress_counts(
        skill_completion_counts,
        state_counts.keys(),
    )
    step_counts = normalize_progress_counts(
        step_counts,
        get_task_progress_labels(getattr(env, "furniture_name", None), "assembly_steps"),
    )
    step_completion_counts = normalize_progress_counts(
        step_completion_counts,
        step_counts.keys(),
    )

    skill_success_rates = compute_success_rates(state_counts, skill_completion_counts)
    step_success_rates = compute_success_rates(step_counts, step_completion_counts)
    expected_skill_labels = get_task_progress_labels(
        getattr(env, "furniture_name", None), "skill_states"
    )
    final_total = n_total_rollouts if use_target_mode else n_rollouts
    tracking_error = build_tracking_error_summary(
        tracking_error_records,
        expected_labels=expected_skill_labels,
        metric_type=tracking_metric_type,
    )
    tracking_error["episode_count"] = tracking_episode_count
    tracking_error["expected_episode_count"] = final_total if collect_skill_stats else 0
    tracking_error["incomplete_episode_count"] = tracking_incomplete_episode_count
    tracking_error["complete"] = bool(
        collect_skill_stats
        and tracking_episode_count == final_total
        and tracking_incomplete_episode_count == 0
    )
    tracking_error["workspace_filter"] = build_tracking_workspace_filter_summary(
        tracking_workspace_counts
    )
    vlm_point_error = (
        merge_vlm_point_error_summaries(vlm_point_error_summaries)
        if vlm_point_error_summaries
        else None
    )
    if len(vlm_model_revisions) > 1:
        raise RuntimeError(
            f"VLM model revision changed during evaluation: {sorted(vlm_model_revisions)}"
        )
    vlm_model_revision = next(iter(vlm_model_revisions), None)

    return RolloutStats(
        success_rate=n_success / max(final_total, 1),
        n_success=n_success,
        n_rollouts=final_total,
        epoch_idx=epoch_idx,
        rollout_max_steps=rollout_max_steps,
        total_return=np.sum(episode_returns) if episode_returns else 0,
        total_reward=total_reward,
        state_counts=state_counts,
        skill_completion_counts=skill_completion_counts,
        skill_success_rates=skill_success_rates,
        step_counts=step_counts,
        step_completion_counts=step_completion_counts,
        step_success_rates=step_success_rates,
        tracking_error=tracking_error,
        vlm_point_error=vlm_point_error,
        vlm_model_revision=vlm_model_revision,
        n_saved_rollouts=saved_rollouts_count,
    )


def do_rollout_evaluation(
    config: DictConfig,
    env: Env,
    save_rollouts_to_file: bool,
    save_rollouts_to_wandb: bool,
    actor: Actor,
    best_success_rate: float,
    epoch_idx: int,
    annotate_grasp: bool = False,
    guidance_point_on_image: bool = False,
    grasp_annotation_on_image: bool = False,
    grasp_part_annotate: bool = False,
    guidance_point_colored: bool = False,
    grasp_annotation_colored: bool = False,
) -> float:
    rollout_task = config.rollout.get("task", config.task)
    rollout_randomness = config.rollout.get("randomness", config.randomness)
    rollout_save_dir = None
    if save_rollouts_to_file:
        rollout_save_dir = trajectory_save_dir(
            controller=env.ctrl_mode,
            domain="sim",
            task=rollout_task,
            demo_source="rollout",
            randomness=rollout_randomness,
            create=False,
        )

    actor.set_task(task2idx[rollout_task])
    provide_skill_input = model_requires_skill_input(config)
    annotate_guidance_point = model_uses_guidance_point(config)
    annotate_grasp = model_uses_grasp(config) or annotate_grasp
    model_guidance_point_colored = model_uses_guidance_point_colored(config)
    model_grasp_annotation_colored = model_uses_grasp_colored(config)

    rollout_stats = calculate_success_rate(
        env,
        actor,
        n_rollouts=config.rollout.count,
        rollout_max_steps=config.rollout.max_steps,
        epoch_idx=epoch_idx,
        discount=config.discount,
        rollout_save_dir=rollout_save_dir,
        save_rollouts_to_wandb=save_rollouts_to_wandb,
        save_failures=config.rollout.save_failures,
        annotate_guidance_point=annotate_guidance_point,
        annotate_grasp=annotate_grasp,
        provide_skill_input=provide_skill_input,
        collect_skill_stats=True,
        guidance_point_on_image=guidance_point_on_image,
        grasp_annotation_on_image=grasp_annotation_on_image,
        grasp_part_annotate=grasp_part_annotate,
        guidance_point_colored=guidance_point_colored,
        grasp_annotation_colored=grasp_annotation_colored,
        model_guidance_point_colored=model_guidance_point_colored,
        model_grasp_annotation_colored=model_grasp_annotation_colored,
    )
    success_rate = rollout_stats.success_rate
    best_success_rate = max(best_success_rate, success_rate)
    mean_return = rollout_stats.total_return / rollout_stats.n_rollouts

    # Log the success rate to wandb
    wandb.log(
        {
            "success_rate": success_rate,
            "best_success_rate": best_success_rate,
            "epoch_mean_return": mean_return,
            "n_success": rollout_stats.n_success,
            "n_rollouts": rollout_stats.n_rollouts,
            "epoch": epoch_idx,
        }
    )

    return best_success_rate
