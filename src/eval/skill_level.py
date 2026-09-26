"""Fixed-state FurnitureBench skill-stage rollout evaluation.

This module is intentionally narrower than the full-task evaluator.  It restores
state-bank records, runs the policy only until the recorded semantic stage is
completed (or fails/times out), and emits one auditable record per state and
stochastic repeat.
"""

from __future__ import annotations

import hashlib
import random
import math
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from furniture_bench.controllers import control_utils as C

from src.common.eepose import ROBOT_BASE, SIM_LOCAL, select_policy_eepose
from src.common.gripper import (
    binarize_robot_state_gripper_width,
    normalizer_expects_binary_gripper_width,
)
from src.eval.progress_schema import (
    _tracking_error_at_frame,
    get_task_progress_labels,
    tracking_target_workspace_status,
)
from src.eval.annotation_noise import AnnotationNoiseConfig
from src.eval.rollout import (
    _add_sim_local_ee_pose_to_robot_state,
    _apply_policy_visual_annotations,
    _attach_skill_tensor_to_obs,
    _policy_action_to_robot_base,
    resize_crop_depth,
    resize_crop_image,
    resize_depth,
    resize_image,
)
from src.eval.skill_annotation_util import (
    get_annotation_bundle_all_envs,
    reset_skill_annotator,
)
from src.eval.state_bank import (
    restore_state_records_batch,
    settle_gripper_with_pinned_furniture,
)


INCLUDED_SKILL_STAGES = {
    "one_leg": (
        "top-leg-pick",
        "top-leg-push",
        "leg-top-pick",
        "leg-top-place",
        "leg-top-screw",
    ),
    "round_table": (
        "top-leg-push",
        "leg-top-pick",
        "leg-top-place",
        "leg-top-screw",
        "base-leg-pick",
        "base-leg-place",
        "base-leg-screw",
    ),
    "lamp": (
        "base-bulb-push",
        "bulb-base-pick",
        "bulb-base-place",
        "bulb-base-screw",
        "hood-base-pick",
        "hood-base-place",
    ),
}

DEFAULT_STAGE_TIMEOUTS = {
    "pick": 240,
    "place": 240,
    "push": 180,
    "screw": 360,
}

RESTORE_PART_POSITION_TOLERANCE_M = 0.003
RESTORE_PART_ORIENTATION_TOLERANCE_DEG = 2.0
RESTORE_ARM_DOF_POSITION_TOLERANCE = 0.002
RESTORE_GRIPPER_DOF_POSITION_TOLERANCE = 0.005
# Backward-compatible alias for callers that have not yet split arm/gripper gates.
RESTORE_DOF_POSITION_TOLERANCE = RESTORE_ARM_DOF_POSITION_TOLERANCE

# Pair indices provide the final-stage completion fallback used when the
# environment terminates on the same physics step that completes the task,
# before the scripted annotator can publish its next-frame ``done`` label.
ASSEMBLY_COMPLETION_STAGE_PAIR_INDEX = {
    "one_leg": {
        "leg-top-screw": 0,
    },
    "round_table": {
        "leg-top-screw": 0,
        "base-leg-screw": 1,
    },
    "lamp": {
        "bulb-base-screw": 0,
        "hood-base-place": 1,
    },
}


def stage_skill(stage: str) -> str:
    skill = str(stage).rsplit("-", 1)[-1]
    if skill not in DEFAULT_STAGE_TIMEOUTS:
        raise ValueError(f"Unsupported clean skill-level stage: {stage!r}")
    return skill


def default_stage_timeout(stage: str) -> int:
    return int(DEFAULT_STAGE_TIMEOUTS[stage_skill(stage)])


def calibrated_stage_timeout(
    stage: str,
    manifest_entries: Sequence[Mapping[str, Any]],
    *,
    expert_margin: float = 1.25,
    quantum: int = 20,
) -> tuple[int, dict[str, Any]]:
    """Freeze one fair timeout from the selected bank before policy rollout.

    The floor is skill-specific.  If the expert itself needs longer from one of
    the selected states, add a fixed 25% margin and round upward.  Because every
    condition evaluates the identical manifest, this remains a single
    stage-specific timeout shared by all policies and repeats.
    """

    if not manifest_entries:
        raise ValueError("timeout calibration requires a non-empty state bank")
    if expert_margin < 1.0:
        raise ValueError("expert_margin must be at least 1.0")
    if quantum <= 0:
        raise ValueError("timeout quantum must be positive")
    remaining_steps = []
    for entry in manifest_entries:
        length = int(entry["stage_length_frames"])
        offset = int(entry["skill_frame_offset"])
        remaining = length - 1 - offset
        if remaining < 0:
            raise ValueError(
                f"invalid expert stage timing: length={length}, offset={offset}"
            )
        remaining_steps.append(remaining)
    expert_max = max(remaining_steps)
    calibrated = int(
        math.ceil((expert_margin * expert_max) / int(quantum)) * int(quantum)
    )
    floor = default_stage_timeout(stage)
    timeout = max(floor, calibrated)
    return timeout, {
        "method": "max_skill_floor_and_ceil_expert_remaining_margin",
        "skill_floor_steps": floor,
        "expert_remaining_max_steps": expert_max,
        "expert_margin": float(expert_margin),
        "rounding_quantum_steps": int(quantum),
        "resolved_steps": timeout,
    }


def classify_stage_transition(task: str, stage: str, next_stage: Any) -> str | None:
    """Classify an annotation transition out of the evaluated stage.

    Returns ``success`` for forward progress, ``wrong_stage`` for regression or
    an unknown label, and ``None`` while the stage remains active.  Final-stage
    completion is detected separately from the environment reward because the
    scripted annotator deliberately retains its last published label.
    """

    if next_stage is not None:
        next_stage = str(next_stage)
    if next_stage == stage:
        return None
    if next_stage == "done":
        return "success"
    labels = get_task_progress_labels(task, "skill_states")
    if stage not in labels or next_stage not in labels:
        return "wrong_stage"
    return "success" if labels.index(next_stage) > labels.index(stage) else "wrong_stage"


def stage_assembly_pair_index(task: str, stage: str) -> int | None:
    return ASSEMBLY_COMPLETION_STAGE_PAIR_INDEX.get(str(task), {}).get(str(stage))


def assembly_pair_mask_value(mask: Any, pair_index: int) -> bool:
    if torch.is_tensor(mask):
        values = mask.detach().reshape(-1)
        if pair_index < 0 or pair_index >= values.numel():
            raise ValueError(
                f"assembly pair index {pair_index} is outside runtime mask of size "
                f"{values.numel()}"
            )
        return bool(values[pair_index].item())
    values = np.asarray(mask).reshape(-1).astype(bool)
    if pair_index < 0 or pair_index >= values.size:
        raise ValueError(
            f"assembly pair index {pair_index} is outside runtime mask of size "
            f"{values.size}"
        )
    return bool(values[pair_index])


def adjudicate_stage_step(
    transition: str | None,
    *,
    completion_pair_index: int | None,
    pair_completed: bool,
    task_final_stage: bool = False,
    environment_done: bool = False,
) -> tuple[bool, str | None]:
    """Resolve one step under the skill-level completion protocol.

    Scripted FSM forward transitions remain authoritative for ordinary stages.
    The final task stage has one narrowly scoped fallback: when the environment
    terminates on the same step and its final assembly pair is complete, count
    that step as success because no subsequent annotation call exists in which
    the FSM could publish ``done``.  A published wrong-stage transition still
    takes precedence and remains a failure.
    """

    if transition == "success":
        return True, "stage_transition"
    if transition == "wrong_stage":
        return False, "wrong_stage"
    if (
        task_final_stage
        and environment_done
        and completion_pair_index is not None
        and pair_completed
    ):
        return True, "final_stage_environment_done"
    return False, None


def summarize_numeric(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "p90": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def summarize_skill_level_records(records: Sequence[Mapping[str, Any]]) -> dict:
    rows = list(records)
    completed = [bool(row.get("completed_current_stage")) for row in rows]

    def values(field: str, *, success_only: bool = False) -> list[float]:
        output = []
        for row in rows:
            if success_only and not row.get("completed_current_stage"):
                continue
            value = row.get(field)
            if value is not None:
                output.append(float(value))
        return output

    by_stratum = {}
    for stratum in ("early", "middle", "late"):
        subset = [row for row in rows if row.get("selection_stratum") == stratum]
        subset_success = sum(bool(row.get("completed_current_stage")) for row in subset)
        by_stratum[stratum] = {
            "attempted": len(subset),
            "completed": subset_success,
            "success_rate": subset_success / len(subset) if subset else None,
        }

    restore_audits = [
        row.get("restore_audit", {})
        for row in rows
        if isinstance(row.get("restore_audit"), Mapping)
    ]

    def audit_max(field: str) -> float | None:
        values = [
            float(audit[field])
            for audit in restore_audits
            if audit.get(field) is not None
        ]
        return max(values) if values else None

    return {
        "attempted": len(rows),
        "completed": sum(completed),
        "success_rate": sum(completed) / len(rows) if rows else None,
        "distinct_states": len({row.get("state_sha256") for row in rows}),
        "termination_reasons": dict(Counter(row.get("termination_reason") for row in rows)),
        "by_stratum": by_stratum,
        "e_gt_cm": summarize_numeric([100.0 * value for value in values("e_gt_m")]),
        "e_input_cm": summarize_numeric(
            [100.0 * value for value in values("e_input_m")]
        ),
        "e_gt_success_only_cm": summarize_numeric(
            [100.0 * value for value in values("e_gt_m", success_only=True)]
        ),
        "te_position_cm": summarize_numeric(
            [100.0 * value for value in values("te_position_m")]
        ),
        "te_orientation_deg": summarize_numeric(values("te_orientation_deg")),
        "te_normalized_total": summarize_numeric(values("te_normalized_total")),
        "completion_steps": summarize_numeric(values("completion_steps")),
        "restore_audit": {
            "count": len(restore_audits),
            "gate_pass_count": sum(bool(audit.get("gate_pass")) for audit in restore_audits),
            "part_position_max_abs_m": audit_max(
                "part_position_max_abs_m_after_commit"
            ),
            "part_orientation_max_abs_deg": audit_max(
                "part_orientation_max_abs_deg_after_commit"
            ),
            "dof_position_max_abs": audit_max(
                "dof_position_max_abs_after_commit"
            ),
        },
    }


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stable_annotation_noise_seed_offset(state_sha256: str, repeat_seed: int) -> int:
    """Build a batching-independent noise stream key for a paired rollout."""

    payload = f"rr-noisy-skill-level-v1\0{state_sha256}\0{int(repeat_seed)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _hold_action(env) -> torch.Tensor:
    ee_pos, ee_quat = env.get_ee_pose()
    gripper = env.last_grasp.reshape(env.num_envs, 1)
    if env.act_rot_repr == "quat":
        orientation = ee_quat
    elif env.act_rot_repr == "rot_6d":
        orientation = C.quaternion_to_rotation_6d(ee_quat)
    elif env.act_rot_repr == "axis":
        orientation = C.quaternion_to_axis_angle(ee_quat)
    else:
        raise ValueError(f"Unsupported rotation representation: {env.act_rot_repr!r}")
    if env.action_type == "pos":
        return torch.cat((ee_pos, orientation, gripper), dim=1)
    if env.action_type == "delta":
        zeros = torch.zeros_like(ee_pos)
        if env.act_rot_repr == "quat":
            orientation = torch.zeros_like(orientation)
            orientation[:, 3] = 1.0
        elif env.act_rot_repr == "rot_6d":
            identity_quat = torch.zeros_like(ee_quat)
            identity_quat[:, 3] = 1.0
            orientation = C.quaternion_to_rotation_6d(identity_quat)
        else:
            orientation = torch.zeros_like(orientation)
        return torch.cat((zeros, orientation, gripper), dim=1)
    raise ValueError(f"Unsupported action type: {env.action_type!r}")


def _prepare_policy_observation(
    env,
    actor,
    obs,
    annotations,
    *,
    annotate_guidance_point: bool,
    annotate_grasp: bool,
    grasp_part_annotate: bool,
    guidance_point_colored: bool,
    grasp_annotation_colored: bool,
    eepose_frame: str,
):
    resize_image(obs, "color_image1")
    resize_crop_image(obs, "color_image2")
    resize_depth(obs, "depth_image1")
    resize_crop_depth(obs, "depth_image2")
    _apply_policy_visual_annotations(
        obs,
        annotations,
        annotate_wrist_camera=False,
        annotate_guidance_point=annotate_guidance_point,
        annotate_grasp=annotate_grasp,
        grasp_part_annotate=grasp_part_annotate,
        guidance_point_colored=guidance_point_colored,
        grasp_annotation_colored=grasp_annotation_colored,
    )
    _attach_skill_tensor_to_obs(
        obs, actor, [bundle.get("skill") for bundle in annotations]
    )
    raw_robot_state = _add_sim_local_ee_pose_to_robot_state(env, obs["robot_state"])
    policy_robot_state = select_policy_eepose(
        raw_robot_state,
        eepose_frame,
        original_frame=SIM_LOCAL,
    )
    if (
        not getattr(actor, "expects_raw_robot_state", False)
        and normalizer_expects_binary_gripper_width(actor.normalizer)
    ):
        policy_robot_state = binarize_robot_state_gripper_width(policy_robot_state)
    obs["robot_state"] = policy_robot_state
    if not getattr(actor, "expects_raw_robot_state", False):
        obs["robot_state"] = env.filter_and_concat_robot_state(obs["robot_state"])
    return obs, raw_robot_state


def _tensor_row(value, env_idx: int) -> np.ndarray:
    if torch.is_tensor(value):
        return value[env_idx].detach().cpu().numpy().copy()
    return np.asarray(value[env_idx]).copy()


def _quaternion_angle_error_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    denominator = np.maximum(left_norm * right_norm, 1e-12)
    cosine_half = np.abs(np.sum(left * right, axis=-1) / denominator)
    cosine_half = np.clip(cosine_half, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cosine_half))


def _terminal_metric(
    *,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    clean_target_pose: Any,
    input_target_pose: Any,
    annotation_noise: Mapping[str, Any] | None,
    metric_type: str,
) -> dict[str, Any]:
    clean_target = (
        None
        if clean_target_pose is None
        else np.asarray(clean_target_pose, dtype=np.float64)
    )
    input_target = (
        None
        if input_target_pose is None
        else np.asarray(input_target_pose, dtype=np.float64)
    )
    workspace_status = tracking_target_workspace_status(clean_target)
    gt_error = _tracking_error_at_frame(
        {"ee_pos": ee_pos, "ee_quat": ee_quat},
        clean_target,
        metric_type=metric_type,
    )
    input_error = _tracking_error_at_frame(
        {"ee_pos": ee_pos, "ee_quat": ee_quat},
        input_target,
        metric_type=metric_type,
    )
    p0 = None if clean_target is None else clean_target[:3, 3]
    p_delta = None if input_target is None else input_target[:3, 3]
    delta = None if p0 is None or p_delta is None else p_delta - p0
    return {
        "target_workspace_status": workspace_status,
        "end_ee_pos_robot_base_m": ee_pos.tolist(),
        "end_ee_quat_xyzw": ee_quat.tolist(),
        "end_clean_target_pose_robot_base": (
            None if clean_target is None else clean_target.tolist()
        ),
        "end_target_pose_robot_base": (
            None if clean_target is None else clean_target.tolist()
        ),
        "end_input_target_pose_robot_base": (
            None if input_target is None else input_target.tolist()
        ),
        "p0_robot_base_m": None if p0 is None else p0.tolist(),
        "p_delta_robot_base_m": None if p_delta is None else p_delta.tolist(),
        "delta_robot_base_m": None if delta is None else delta.tolist(),
        "delta_norm_m": None if delta is None else float(np.linalg.norm(delta)),
        "annotation_noise": dict(annotation_noise or {"enabled": False}),
        "e_gt_m": None if gt_error is None else float(gt_error["pos_m"]),
        "e_input_m": None if input_error is None else float(input_error["pos_m"]),
        # TE.position is the error to the target actually shown to the policy.
        "te_position_m": (
            None if input_error is None else float(input_error["pos_m"])
        ),
        "te_orientation_deg": (
            None
            if input_error is None or "ori_deg" not in input_error
            else float(input_error["ori_deg"])
        ),
        "te_normalized_total": (
            None
            if input_error is None or "total" not in input_error
            else float(input_error["total"])
        ),
    }


@torch.no_grad()
def rollout_skill_stage_batch(
    *,
    env,
    actor,
    records: Sequence[Mapping[str, Any]],
    manifest_entries: Sequence[Mapping[str, Any]],
    accepted_count: int,
    rollout_seed: int,
    max_steps: int,
    metric_type: str,
    annotate_guidance_point: bool,
    annotate_grasp: bool,
    grasp_part_annotate: bool,
    guidance_point_colored: bool,
    grasp_annotation_colored: bool,
    annotation_noise_config: AnnotationNoiseConfig | None = None,
    annotation_noise_seed_offsets: Sequence[int] | None = None,
    eepose_frame: str = ROBOT_BASE,
    gripper_settle_steps: int = 3,
    frame_callback: Callable[[int, Any, Sequence[Mapping[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    if len(records) != int(env.num_envs) or len(manifest_entries) != len(records):
        raise ValueError("records and manifest_entries must match env.num_envs")
    if not 0 < accepted_count <= len(records):
        raise ValueError("accepted_count must select at least one restored environment")
    tasks = {str(record["metadata"]["task"]) for record in records}
    stages = {str(record["metadata"]["skill_state"]) for record in records}
    if len(tasks) != 1 or len(stages) != 1:
        raise ValueError("one stage batch must contain exactly one task and one stage")
    task = next(iter(tasks))
    stage = next(iter(stages))
    if stage not in INCLUDED_SKILL_STAGES.get(task, ()):
        raise ValueError(f"Stage {task}/{stage} is not in the clean analysis plan")
    completion_pair_index = stage_assembly_pair_index(task, stage)
    task_final_stage = stage == INCLUDED_SKILL_STAGES[task][-1]
    policy_receives_guidance = bool(
        annotate_guidance_point
        or annotate_grasp
        or grasp_part_annotate
        or guidance_point_colored
        or grasp_annotation_colored
    )

    env.reset()
    reset_skill_annotator(env)
    restore_state_records_batch(
        env,
        records,
        restore_velocity=False,
        restore_rng=False,
        render_cameras=False,
    )
    settle_gripper_with_pinned_furniture(
        env, records, physics_steps=gripper_settle_steps
    )
    # State restoration intentionally restores the recorded annotation FSM,
    # including its historical clean noise state. Override the noise stream
    # only after restore/settle so every evaluation state gets its paired key.
    if annotation_noise_seed_offsets is not None:
        if len(annotation_noise_seed_offsets) != len(records):
            raise ValueError("annotation noise offsets must match records")
        for annotator, offset in zip(
            env._skill_annotators, annotation_noise_seed_offsets
        ):
            annotator.noise_seed_offset = int(offset)
            annotator.noise_state = None

    root_actual = env.root_tensor.view(int(env.num_envs), -1, 13).detach().cpu().numpy()
    dof_actual = env.dof_states.view(int(env.num_envs), -1, 2).detach().cpu().numpy()
    restore_audits = []
    for env_idx, record in enumerate(records):
        root_saved = np.asarray(record["physics"]["root_state"], dtype=np.float32)
        dof_saved = np.asarray(record["physics"]["dof_state"], dtype=np.float32)
        part_indices = sorted(
            {
                int(index)
                for index in record["physics"]["layout"][
                    "part_actor_local_indices"
                ].values()
            }
        )
        part_position_error = float(
            np.max(
                np.abs(
                    root_actual[env_idx, part_indices, :3]
                    - root_saved[part_indices, :3]
                )
            )
        )
        part_orientation_error = float(
            np.max(
                _quaternion_angle_error_deg(
                    root_actual[env_idx, part_indices, 3:7],
                    root_saved[part_indices, 3:7],
                )
            )
        )
        dof_position_error = float(
            np.max(np.abs(dof_actual[env_idx, :, 0] - dof_saved[:, 0]))
        )
        dof_position_abs_error = np.abs(
            dof_actual[env_idx, :, 0] - dof_saved[:, 0]
        )
        arm_dof_position_error = float(np.max(dof_position_abs_error[:7]))
        gripper_dof_position_error = float(np.max(dof_position_abs_error[7:]))
        inputs = env.get_skill_annotation_inputs(env_idx=env_idx)
        restore_gate_pass = bool(
            np.isfinite(part_position_error)
            and np.isfinite(part_orientation_error)
            and np.isfinite(dof_position_error)
            and part_position_error <= RESTORE_PART_POSITION_TOLERANCE_M
            and part_orientation_error <= RESTORE_PART_ORIENTATION_TOLERANCE_DEG
            and arm_dof_position_error <= RESTORE_ARM_DOF_POSITION_TOLERANCE
            and gripper_dof_position_error
            <= RESTORE_GRIPPER_DOF_POSITION_TOLERANCE
        )
        audit = {
                "root_position_max_abs_m_after_commit": float(
                    np.max(np.abs(root_actual[env_idx, :, :3] - root_saved[:, :3]))
                ),
                "part_position_max_abs_m_after_commit": part_position_error,
                "part_orientation_max_abs_deg_after_commit": part_orientation_error,
                "dof_position_max_abs_after_commit": dof_position_error,
                "arm_dof_position_max_abs_after_commit": arm_dof_position_error,
                "gripper_dof_position_max_abs_after_commit": gripper_dof_position_error,
                "left_finger_contact_force_norm": float(
                    torch.linalg.norm(inputs["left_finger_force"]).item()
                ),
                "right_finger_contact_force_norm": float(
                    torch.linalg.norm(inputs["right_finger_force"]).item()
                ),
                "restore_velocity": "zeroed",
                "restore_mode": "pinned_gripper_settle",
                "gripper_settle_steps": int(gripper_settle_steps),
                "gate_thresholds": {
                    "part_position_max_abs_m": RESTORE_PART_POSITION_TOLERANCE_M,
                    "part_orientation_max_abs_deg": RESTORE_PART_ORIENTATION_TOLERANCE_DEG,
                    "arm_dof_position_max_abs": RESTORE_ARM_DOF_POSITION_TOLERANCE,
                    "gripper_dof_position_max_abs": RESTORE_GRIPPER_DOF_POSITION_TOLERANCE,
                },
                "gate_pass": restore_gate_pass,
            }
        restore_audits.append(audit)
        if env_idx < accepted_count and not restore_gate_pass:
            raise RuntimeError(
                f"restore gate failed for env {env_idx}: "
                f"part_pos={part_position_error:.6f}m "
                f"part_ori={part_orientation_error:.3f}deg "
                f"arm_dof={arm_dof_position_error:.6f} "
                f"gripper_dof={gripper_dof_position_error:.6f}"
            )

    _seed_everything(rollout_seed)
    actor.reset()
    actor.normalizer = actor.normalizer.to(actor.device)
    actor.model = actor.model.to(actor.device)

    obs = env.get_observation()
    previous_skills = [record["metadata"].get("skill") for record in records]
    annotations = get_annotation_bundle_all_envs(
        env,
        previous_skills=previous_skills,
        annotate_wrist_camera=False,
        resize_images=True,
        enable_verify=True,
        annotation_noise_config=annotation_noise_config,
    )
    initial_labels = [bundle.get("skill_state") for bundle in annotations]
    mismatches = [
        idx for idx, label in enumerate(initial_labels[:accepted_count]) if label != stage
    ]
    if mismatches:
        raise RuntimeError(
            f"Restored scripted stage mismatch for {task}/{stage}: envs={mismatches}, "
            f"labels={initial_labels[:accepted_count]}"
        )
    if frame_callback is not None:
        frame_callback(0, obs, annotations)

    initial_robot_state = obs["robot_state"]
    start_positions = [
        _tensor_row(initial_robot_state["ee_pos"], idx).reshape(3)
        for idx in range(env.num_envs)
    ]
    previous_positions = [value.copy() for value in start_positions]
    path_lengths = np.zeros(env.num_envs, dtype=np.float64)
    ee_position_trajectories: list[list[list[float]]] = [
        [] for _ in range(env.num_envs)
    ]
    ee_quaternion_trajectories: list[list[list[float]]] = [
        [] for _ in range(env.num_envs)
    ]
    policy_input_target_pose_trajectories: list[list[Any]] = [
        [] for _ in range(env.num_envs)
    ]
    evaluation_target_pose_trajectories: list[list[Any]] = [
        [] for _ in range(env.num_envs)
    ]
    active = np.zeros(env.num_envs, dtype=bool)
    active[:accepted_count] = True
    results: list[dict[str, Any] | None] = [None] * env.num_envs

    for step_idx in range(1, int(max_steps) + 1):
        policy_obs, raw_robot_state = _prepare_policy_observation(
            env,
            actor,
            obs,
            annotations,
            annotate_guidance_point=annotate_guidance_point,
            annotate_grasp=annotate_grasp,
            grasp_part_annotate=grasp_part_annotate,
            guidance_point_colored=guidance_point_colored,
            grasp_annotation_colored=grasp_annotation_colored,
            eepose_frame=eepose_frame,
        )
        action = actor.action(policy_obs)
        action = _policy_action_to_robot_base(
            env, action, raw_robot_state, eepose_frame
        )
        if not bool(np.all(active)):
            hold = _hold_action(env)
            inactive = torch.as_tensor(~active, device=action.device, dtype=torch.bool)
            action[inactive] = hold[inactive]

        next_obs, reward, env_done, _ = env.step(action, sample_perturbations=False)
        next_annotations = get_annotation_bundle_all_envs(
            env,
            previous_skills=previous_skills,
            annotate_wrist_camera=False,
            resize_images=True,
            enable_verify=True,
            annotation_noise_config=annotation_noise_config,
        )
        if frame_callback is not None:
            frame_callback(step_idx, next_obs, next_annotations)
        for env_idx, bundle in enumerate(next_annotations):
            if bundle.get("skill") is not None:
                previous_skills[env_idx] = bundle.get("skill")

        next_robot_state = next_obs["robot_state"]
        done_flat = env_done.reshape(-1).detach().cpu().numpy().astype(bool)
        for env_idx in range(accepted_count):
            if not active[env_idx]:
                continue
            ee_pos = _tensor_row(next_robot_state["ee_pos"], env_idx).reshape(3)
            ee_quat = _tensor_row(next_robot_state["ee_quat"], env_idx).reshape(4)
            path_lengths[env_idx] += float(
                np.linalg.norm(ee_pos - previous_positions[env_idx])
            )
            previous_positions[env_idx] = ee_pos
            ee_position_trajectories[env_idx].append(ee_pos.tolist())
            ee_quaternion_trajectories[env_idx].append(ee_quat.tolist())
            action_target = (
                annotations[env_idx].get("guidance_pose")
                if policy_receives_guidance
                else None
            )
            policy_input_target_pose_trajectories[env_idx].append(
                None
                if action_target is None
                else np.asarray(action_target, dtype=np.float64).tolist()
            )

            transition = classify_stage_transition(
                task, stage, next_annotations[env_idx].get("skill_state")
            )
            pair_completed = bool(
                completion_pair_index is not None
                and assembly_pair_mask_value(
                    env.already_assembled[env_idx], completion_pair_index
                )
            )
            completed, termination_reason = adjudicate_stage_step(
                transition,
                completion_pair_index=completion_pair_index,
                pair_completed=pair_completed,
                task_final_stage=task_final_stage,
                environment_done=bool(done_flat[env_idx]),
            )
            step_target_bundle = (
                annotations[env_idx]
                if termination_reason is not None
                else next_annotations[env_idx]
            )
            evaluation_target = step_target_bundle.get("guidance_pose_clean")
            input_target = step_target_bundle.get("guidance_pose")
            evaluation_target_pose_trajectories[env_idx].append(
                None
                if evaluation_target is None
                else np.asarray(evaluation_target, dtype=np.float64).tolist()
            )
            if termination_reason is None and bool(done_flat[env_idx]):
                termination_reason = "environment_done"
            elif termination_reason is None and step_idx >= int(max_steps):
                termination_reason = "timeout"

            if termination_reason is None:
                continue
            terminal = _terminal_metric(
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                clean_target_pose=evaluation_target,
                input_target_pose=input_target,
                annotation_noise=step_target_bundle.get("annotation_noise"),
                metric_type=metric_type,
            )
            entry = manifest_entries[env_idx]
            results[env_idx] = {
                "task": task,
                "skill_stage": stage,
                "skill_type": stage_skill(stage),
                "state_path": entry.get("path"),
                "state_sha256": entry.get("sha256"),
                "source_episode_index": int(entry["episode_index"]),
                "selection_stratum": entry.get("selection_stratum"),
                "stage_progress": float(entry.get("stage_progress", 0.0)),
                "rollout_seed": int(rollout_seed),
                "completed_current_stage": bool(completed),
                "completion_steps": int(step_idx),
                "termination_reason": termination_reason,
                "transition_from": stage,
                "transition_to": next_annotations[env_idx].get("skill_state"),
                "benchmark_assembly_pair_index": completion_pair_index,
                "benchmark_assembly_pair_completed_at_terminal": pair_completed,
                "start_ee_pos_robot_base_m": start_positions[env_idx].tolist(),
                "net_ee_displacement_m": float(
                    np.linalg.norm(ee_pos - start_positions[env_idx])
                ),
                "ee_path_length_m": float(path_lengths[env_idx]),
                "ee_position_trajectory_robot_base_m": ee_position_trajectories[
                    env_idx
                ],
                "ee_quaternion_trajectory_xyzw": ee_quaternion_trajectories[
                    env_idx
                ],
                "policy_input_clean_target_pose_trajectory_robot_base": (
                    policy_input_target_pose_trajectories[env_idx]
                ),
                "clean_target_pose_trajectory_robot_base": (
                    evaluation_target_pose_trajectories[env_idx]
                ),
                "restore_audit": restore_audits[env_idx],
                **terminal,
            }
            active[env_idx] = False

        if not bool(np.any(active[:accepted_count])):
            break
        annotations = next_annotations
        obs = next_obs

    if any(result is None for result in results[:accepted_count]):
        raise RuntimeError("stage rollout ended without terminal records")
    return [dict(result) for result in results[:accepted_count] if result is not None]
