"""Standalone Deoxys evaluation for RR absolute-pose RGBD policies.

This first hardware version intentionally supports one narrow contract:

* one-arm FurnitureBench tasks (initially ``one_leg``);
* Deoxys ``OSC_POSE`` with native absolute position + absolute axis-angle;
* online Prompt Depth Anything for both cameras;
* RR checkpoints trained with ``control.control_mode=pos``;
* timestamped action chunks, queried every four 10 Hz control steps by default.

The command is a dry-run unless ``--execute`` is supplied.  Hardware execution
also requires a measured latency profile and explicit workspace bounds.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from src.real.action_safety import ActionSafetyLimits, validate_absolute_action
from src.real.deoxys_runtime import (
    gripper_sample_from_record,
    interpolate_gripper_width,
    interpolate_robot_state,
    robot_sample_from_record,
)
from src.real.time_alignment import CoordinatedActionBuffer, LatencyProfile


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


class EvalEventLog:
    def __init__(self, path: Path, metadata: Mapping[str, Any]):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x")
        self.write("metadata", **dict(metadata))

    def write(self, event: str, **fields) -> None:
        payload = {
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_time_ns": time.monotonic_ns(),
            **fields,
        }
        self.file.write(json.dumps(payload, default=_json_value) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _load_actor(checkpoint_path: Path, config_path: Optional[Path], device: str):
    import torch
    from omegaconf import OmegaConf

    from src.behavior import get_actor
    from src.behavior.diffusion import DiffusionPolicy

    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    try:
        checkpoint = torch.load(
            resolved_checkpoint, map_location=device, weights_only=False
        )
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        checkpoint = torch.load(resolved_checkpoint, map_location=device)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if checkpoint_config is not None:
        cfg = OmegaConf.create(checkpoint_config)
    elif config_path is not None:
        cfg = OmegaConf.load(config_path.expanduser().resolve())
    else:
        raise ValueError("checkpoint has no config; provide --config")
    if cfg.control.control_mode != "pos":
        raise ValueError(
            "real Deoxys v1 requires an absolute-pose checkpoint: "
            f"control.control_mode={cfg.control.control_mode!r}"
        )
    if cfg.observation_type != "rgbd":
        raise ValueError(
            "real Deoxys v1 requires online PromptDA RGBD input; got "
            f"observation_type={cfg.observation_type!r}"
        )
    actor = get_actor(cfg=cfg, device=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", checkpoint)
        )
    actor.load_state_dict(state_dict)
    actor.eval()
    actor.to(device)
    if isinstance(actor, DiffusionPolicy):
        actor.inference_steps = 4
    return actor, cfg


def _annotation_mode(cfg) -> str:
    from src.behavior.base import (
        model_uses_grasp,
        model_uses_grasp_colored,
        model_uses_grasp_part,
        model_uses_guidance_point,
        model_uses_guidance_point_colored,
    )

    if model_uses_grasp(cfg) or model_uses_grasp_part(cfg):
        return "grasp-part-colored" if model_uses_grasp_colored(cfg) else "grasp-part"
    if model_uses_guidance_point(cfg):
        return (
            "guidance-point-colored"
            if model_uses_guidance_point_colored(cfg)
            else "guidance-point"
        )
    return "none"


def _absolute_controller_config(time_fraction: float):
    from deoxys.utils.config_utils import get_default_controller_config

    config = get_default_controller_config("OSC_POSE")
    config.is_delta = False
    config.action_scale.translation = 1.0
    config.action_scale.rotation = 1.0
    config.traj_interpolator_cfg.traj_interpolator_type = "LINEAR_POSE"
    config.traj_interpolator_cfg.time_fraction = float(time_fraction)
    return config


def _enhanced_camera_sample(prompt_result: Mapping[str, Any]) -> Dict[str, Any]:
    source = prompt_result.get("camera_sample")
    depths = prompt_result.get("depths") or {}
    if source is None:
        raise ValueError("PromptDA result has no source camera sample")
    output = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in source.items()
    }
    for depth_key in ("depth_image1", "depth_image2"):
        if depth_key not in depths:
            raise ValueError(f"PromptDA result is missing {depth_key}")
        raw = np.asarray(source[depth_key])
        enhanced = np.asarray(depths[depth_key], dtype=np.float32)
        if raw.shape != enhanced.shape:
            raise ValueError(f"PromptDA {depth_key} shape changed")
        output[f"{depth_key}_realsense"] = raw.copy()
        output[depth_key] = enhanced
    output["prompt_depth_submitted_wall_time_ns"] = prompt_result.get(
        "submitted_wall_time_ns"
    )
    output["prompt_depth_started_wall_time_ns"] = prompt_result.get(
        "processing_started_wall_time_ns"
    )
    output["prompt_depth_ready_wall_time_ns"] = prompt_result.get(
        "ready_wall_time_ns"
    )
    return output


def _timestamped_records(robot_interface, kind: str):
    method_name = f"timestamped_{kind}_state_buffer"
    method = getattr(robot_interface, method_name, None)
    if method is None:
        raise RuntimeError(
            f"Deoxys FrankaInterface lacks {method_name}(); use the timestamped "
            "Deoxys version paired with this evaluator"
        )
    return method(max_records=1000)


def _build_aligned_observation(
    *,
    prompt_result: Mapping[str, Any],
    robot_interface,
    latency: LatencyProfile,
    kinematics,
    max_observation_age_ms: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    camera = _enhanced_camera_sample(prompt_result)
    for camera_name in ("front", "wrist"):
        domain = str(camera.get(f"{camera_name}_timestamp_domain", "")).lower()
        if domain and "global_time" not in domain and "system_time" not in domain:
            raise ValueError(
                f"{camera_name} timestamp domain {domain!r} is not a shared wall clock"
            )
    front_time_ns = int(round(float(camera["front_sensor_timestamp_ms"]) * 1e6))
    wrist_time_ns = int(round(float(camera["wrist_sensor_timestamp_ms"]) * 1e6))
    residual_ms = (wrist_time_ns - front_time_ns) / 1e6
    if abs(residual_ms) > 50.0:
        raise ValueError(
            f"wrist/front source residual {residual_ms:.1f} ms exceeds 50 ms"
        )
    now_ns = time.time_ns()
    age_ms = (now_ns - front_time_ns) / 1e6
    if age_ms < -10.0 or age_ms > max_observation_age_ms:
        raise ValueError(
            f"front observation age {age_ms:.1f} ms is outside watchdog range"
        )
    ready_ns = camera.get("prompt_depth_ready_wall_time_ns")
    if ready_ns is None:
        raise RuntimeError("PromptDA worker did not record ready_wall_time_ns")

    robot_records = _timestamped_records(robot_interface, "robot")
    gripper_records = _timestamped_records(robot_interface, "gripper")
    robot_samples = [robot_sample_from_record(record) for record in robot_records]
    gripper_samples = [gripper_sample_from_record(record) for record in gripper_records]
    aligned = interpolate_robot_state(
        robot_samples,
        front_time_ns,
        observation_latency_ms=latency.robot_observation_ms,
    )
    gripper_width = interpolate_gripper_width(
        gripper_samples,
        front_time_ns,
        observation_latency_ms=latency.gripper_observation_ms,
    )
    ee_quat = Rotation.from_matrix(aligned.wrist_pose[:3, :3]).as_quat()
    ee_velocity = kinematics.ee_twist(
        aligned.joint_positions,
        aligned.joint_velocities,
        aligned.wrist_pose[:3, 3],
    )
    observation = dict(camera)
    observation.update(
        {
            "step_timestamp_ns": front_time_ns,
            "camera_anchor": "front",
            "front_source_wall_time_ns": front_time_ns,
            "wrist_source_wall_time_ns": wrist_time_ns,
            "wrist_time_residual_ms": residual_ms,
            "prompt_depth_latency_ms": (int(ready_ns) - front_time_ns) / 1e6,
            "robot_state": {
                "ee_pos": aligned.wrist_pose[:3, 3].copy(),
                "ee_quat": ee_quat,
                "ee_pose": aligned.wrist_pose.copy(),
                "wrist_pose": aligned.wrist_pose.copy(),
                "ee_pos_vel": ee_velocity[:3],
                "ee_ori_vel": ee_velocity[3:],
                "joint_positions": aligned.joint_positions,
                "joint_velocities": aligned.joint_velocities,
                "joint_torques": aligned.joint_torques,
                "gripper_width": gripper_width,
            },
            "skill": None,
            "guidance": None,
        }
    )
    timing = {
        "observation_time_ns": front_time_ns,
        "front_frame_number": camera.get("front_frame_number"),
        "wrist_frame_number": camera.get("wrist_frame_number"),
        "front_age_ms_at_build": age_ms,
        "wrist_residual_ms": residual_ms,
        "prompt_depth_ready_wall_time_ns": int(ready_ns),
        "prompt_depth_latency_ms": observation["prompt_depth_latency_ms"],
        "robot_left_receive_wall_time_ns": aligned.left_receive_wall_time_ns,
        "robot_right_receive_wall_time_ns": aligned.right_receive_wall_time_ns,
    }
    return observation, timing


def _policy_observation(
    observation: Dict[str, Any],
    *,
    actor,
    device: str,
    binary_gripper: bool,
) -> Dict[str, Any]:
    import torch
    from src.common.gripper import binarize_gripper_width
    from src.common.skills import skill_to_onehot_tensor

    state = observation["robot_state"]
    gripper = float(state["gripper_width"])
    if binary_gripper:
        gripper = float(np.asarray(binarize_gripper_width(np.asarray(gripper))))
    robot_state = np.concatenate(
        [
            np.asarray(state["ee_pos"]).reshape(3),
            np.asarray(state["ee_quat"]).reshape(4),
            np.asarray(state["ee_pos_vel"]).reshape(3),
            np.asarray(state["ee_ori_vel"]).reshape(3),
            [gripper],
        ]
    ).astype(np.float32)
    policy = {
        "robot_state": torch.as_tensor(robot_state, device=device).unsqueeze(0),
        "color_image1": torch.as_tensor(
            np.asarray(observation["color_image1"]), device=device
        ).unsqueeze(0),
        "color_image2": torch.as_tensor(
            np.asarray(observation["color_image2"]), device=device
        ).unsqueeze(0),
        "depth_image1": torch.as_tensor(
            np.asarray(observation["depth_image1"], dtype=np.float32), device=device
        ).unsqueeze(0),
        "depth_image2": torch.as_tensor(
            np.asarray(observation["depth_image2"], dtype=np.float32), device=device
        ).unsqueeze(0),
    }
    if actor.skill_dim:
        policy["skill"] = skill_to_onehot_tensor(
            observation.get("skill"), actor.skill_dim, device=device
        ).unsqueeze(0)
    return policy


def _default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("logs") / "real_policy_eval" / f"{timestamp}.jsonl"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=("one_leg",), default="one_leg")
    parser.add_argument("--interface-cfg", default="config/charmander.yml")
    parser.add_argument("--front-camera-serial", default="327122071654")
    parser.add_argument("--wrist-camera-serial", default="001622071252")
    parser.add_argument("--latency-profile", type=Path, default=None)
    parser.add_argument("--frequency", type=float, default=10.0)
    parser.add_argument("--query-interval-steps", type=int, default=4)
    parser.add_argument("--min-future-actions", type=int, default=2)
    parser.add_argument("--max-observation-age-ms", type=float, default=300.0)
    parser.add_argument("--max-action-lateness-ms", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-wall-time-s", type=float, default=180.0)
    parser.add_argument("--max-consecutive-rejections", type=int, default=20)
    parser.add_argument("--start-delay-s", type=float, default=3.0)
    parser.add_argument("--controller-time-fraction", type=float, default=2.0)
    parser.add_argument("--prompt-depth-model", choices=("vits", "vitl", "vits-transparent"), default="vitl")
    parser.add_argument("--prompt-depth-device", default="cuda")
    parser.add_argument("--prompt-depth-max-size", type=int, default=448)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workspace-min", type=float, nargs=3, default=None)
    parser.add_argument("--workspace-max", type=float, nargs=3, default=None)
    parser.add_argument("--min-ee-z", type=float, default=None)
    parser.add_argument("--max-translation-step-m", type=float, default=0.025)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.35)
    parser.add_argument("--max-translation-speed-m-s", type=float, default=0.25)
    parser.add_argument("--max-rotation-speed-rad-s", type=float, default=1.5)
    args = parser.parse_args(argv)
    if args.frequency <= 0 or args.query_interval_steps <= 0:
        parser.error("frequency and query interval must be positive")
    if args.max_wall_time_s <= 0 or args.max_consecutive_rejections <= 0:
        parser.error("watchdog limits must be positive")
    if args.execute:
        if args.latency_profile is None:
            parser.error("--execute requires --latency-profile")
        if args.workspace_min is None or args.workspace_max is None or args.min_ee_z is None:
            parser.error(
                "--execute requires --workspace-min, --workspace-max, and --min-ee-z"
            )
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    # Hardware imports stay local so alignment/tests do not require Deoxys or
    # RealSense.  This also gives a direct diagnosis for a wrong environment.
    try:
        import torch
        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils.furniture_bench_utils import DualRealSenseSnapshotter
        from deoxys.utils.panda_kinematics import PandaKinematics
        from deoxys.utils.prompt_depth_anything import (
            PromptDepthAnythingEstimator,
            PromptDepthWorker,
        )
    except ImportError as exc:
        raise RuntimeError(
            "real evaluation requires the Deoxys/RealSense environment"
        ) from exc

    from src.behavior.base import model_requires_skill_input
    from src.common.gripper import (
        GRIPPER_OPEN_THRESHOLD_METERS,
        normalizer_expects_binary_gripper_width,
    )
    from src.data_processing.offline_image_annotations import annotate_observation_image
    from src.eval.real_skill_annotation_util import RealSkillAnnotationSession

    actor, cfg = _load_actor(args.checkpoint, args.config, args.device)
    period_ns = int(round(1e9 / args.frequency))
    period_s = period_ns / 1e9
    if actor.obs_horizon != 1:
        raise ValueError(
            "real Deoxys v1 currently supports obs_horizon=1 only; a larger "
            "horizon needs a separate 10 Hz observation accumulator"
        )
    if int(cfg.action_dim) != 10:
        raise ValueError(
            f"expected 10D absolute pos/rot6d/gripper actions, got {cfg.action_dim}"
        )
    if actor.action_horizon < args.query_interval_steps:
        raise ValueError("policy action horizon is shorter than query interval")
    latency = (
        LatencyProfile.load(args.latency_profile)
        if args.latency_profile is not None
        else LatencyProfile(
            0,
            0,
            0,
            0,
            0,
            0,
            measured_at="dry-run-zero-profile",
            schema_version=2,
            latency_source="estimated",
            basis="dry-run zero-latency profile",
        )
    )
    default_min = [0.2, -0.5, 0.02]
    default_max = [0.8, 0.5, 0.8]
    limits = ActionSafetyLimits(
        workspace_min=np.asarray(args.workspace_min or default_min),
        workspace_max=np.asarray(args.workspace_max or default_max),
        min_ee_z=float(args.min_ee_z if args.min_ee_z is not None else 0.03),
        max_translation_step_m=args.max_translation_step_m,
        max_rotation_step_rad=args.max_rotation_step_rad,
        max_translation_speed_m_s=args.max_translation_speed_m_s,
        max_rotation_speed_rad_s=args.max_rotation_speed_rad_s,
    )
    log_path = args.log_path or _default_log_path()
    event_log = EvalEventLog(
        log_path,
        {
            "schema": "rr_deoxys_absolute_policy_eval_v2_umi_time",
            "mode": "execute" if args.execute else "dry_run",
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "task": args.task,
            "frequency": args.frequency,
            "action_horizon": actor.action_horizon,
            "query_interval_steps": args.query_interval_steps,
            "latency_profile": None if args.latency_profile is None else str(args.latency_profile),
            "latency_source": latency.latency_source,
            "latency_basis": latency.basis,
            "action_stale_guard_ms": latency.action_stale_guard_ms,
            "workspace_min": limits.workspace_min,
            "workspace_max": limits.workspace_max,
            "min_ee_z": limits.min_ee_z,
        },
    )
    print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN'} log={event_log.path}")

    camera = None
    worker = None
    robot = None
    controller_cfg = _absolute_controller_config(args.controller_time_fraction)
    action_buffer = CoordinatedActionBuffer(period_ns)
    validated_actions = {}
    annotation_mode = _annotation_mode(cfg)
    annotation_session = None
    last_prompt_token = None
    last_target_pose = None
    last_gripper_sign = None
    query_id = 0
    executed_steps = 0
    next_query_ns = 0
    next_hold_ns = 0
    consecutive_rejections = 0
    rollout_started_monotonic = None
    stop_request = {"signal": None}
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def request_stop(signum, _frame):
        stop_request["signal"] = int(signum)

    signal.signal(signal.SIGTERM, request_stop)
    try:
        camera = DualRealSenseSnapshotter(
            front_serial=args.front_camera_serial,
            wrist_serial=args.wrist_camera_serial,
            record_width=320,
            record_height=240,
            furniture_task=args.task,
            front_width=1280,
            front_height=720,
            front_fps=30,
            front_depth_width=1280,
            front_depth_height=720,
            front_depth_fps=30,
            wrist_width=424,
            wrist_height=240,
            wrist_fps=30,
            wrist_depth_width=480,
            wrist_depth_height=270,
            wrist_depth_fps=30,
        )
        camera.start()
        camera_info = camera.metadata()
        worker = PromptDepthWorker(
            PromptDepthAnythingEstimator(
                model=args.prompt_depth_model,
                device=args.prompt_depth_device,
                max_size=args.prompt_depth_max_size,
                min_depth_m=0.05,
                max_depth_m=5.0,
            ),
            cameras=("wrist", "front"),
        )
        worker.start()
        robot = FrankaInterface(
            args.interface_cfg,
            control_freq=args.frequency,
            state_freq=100.0,
            has_gripper=True,
            use_visualizer=False,
            automatic_gripper_reset=False,
        )
        kinematics = PandaKinematics()
        if model_requires_skill_input(cfg) or annotation_mode != "none":
            annotation_session = RealSkillAnnotationSession(
                args.task, camera_info, mode="online"
            )

        print("warming camera, PromptDA, and timestamped robot buffers...")
        warm_deadline = time.monotonic() + 30.0
        while time.monotonic() < warm_deadline:
            worker.submit(camera.latest())
            result = worker.latest()
            if (
                result is not None
                and result.get("ready_wall_time_ns") is not None
                and len(_timestamped_records(robot, "robot")) >= 2
                and len(_timestamped_records(robot, "gripper")) >= 2
            ):
                break
            time.sleep(0.005)
        else:
            raise RuntimeError("timed out warming PromptDA or robot state buffers")

        measured_gripper = robot.last_gripper_q
        if measured_gripper is not None:
            measured_width = float(np.asarray(measured_gripper).reshape(-1)[0])
            last_gripper_sign = (
                -1.0 if measured_width >= GRIPPER_OPEN_THRESHOLD_METERS else 1.0
            )
        if args.execute:
            warm_pose = np.asarray(robot.last_eef_pose, dtype=np.float64)
            warm_action = np.r_[
                warm_pose[:3, 3],
                Rotation.from_matrix(warm_pose[:3, :3]).as_rotvec(),
                -1.0,
            ]
            warm_timing = robot.control(
                "OSC_POSE",
                warm_action,
                controller_cfg=controller_cfg,
                control_gripper=False,
                enforce_control_frequency=False,
            )
            last_target_pose = warm_pose.copy()
            next_hold_ns = time.time_ns() + period_ns
            event_log.write(
                "controller_warmup",
                target_pose=warm_pose,
                **warm_timing,
            )
        time.sleep(args.start_delay_s)
        rollout_started_monotonic = time.monotonic()
        event_log.write("rollout_started")
        while executed_steps < args.max_steps:
            if stop_request["signal"] is not None:
                event_log.write(
                    "signal_stop",
                    signal=stop_request["signal"],
                )
                break
            if (
                time.monotonic() - rollout_started_monotonic
                >= args.max_wall_time_s
            ):
                event_log.write(
                    "watchdog_stop",
                    reason="max_wall_time",
                    consecutive_rejections=consecutive_rejections,
                )
                break
            if consecutive_rejections >= args.max_consecutive_rejections:
                event_log.write(
                    "watchdog_stop",
                    reason="consecutive_rejections",
                    consecutive_rejections=consecutive_rejections,
                )
                break
            camera_sample = camera.latest()
            worker.submit(camera_sample)
            prompt_result = worker.latest()
            now_ns = time.time_ns()
            prompt_token = None
            if prompt_result is not None and prompt_result.get("camera_sample") is not None:
                prompt_token = prompt_result["camera_sample"].get("front_frame_number")

            if (
                prompt_result is not None
                and prompt_token != last_prompt_token
                and now_ns >= next_query_ns
            ):
                try:
                    observation, timing = _build_aligned_observation(
                        prompt_result=prompt_result,
                        robot_interface=robot,
                        latency=latency,
                        kinematics=kinematics,
                        max_observation_age_ms=args.max_observation_age_ms,
                    )
                    if annotation_session is not None:
                        annotation_session.annotate_observation(observation)
                    if annotation_mode != "none":
                        observation = annotate_observation_image(
                            observation,
                            annotation_mode,
                            trajectory_camera_info=camera_info,
                        )
                    policy_obs = _policy_observation(
                        observation,
                        actor=actor,
                        device=args.device,
                        binary_gripper=normalizer_expects_binary_gripper_width(
                            actor.normalizer
                        ),
                    )
                    inference_start_ns = time.time_ns()
                    chunk_tensor = actor.action_chunk(policy_obs)
                    if chunk_tensor.shape[0] != 1:
                        raise ValueError("real evaluation requires policy batch size 1")
                    chunk = chunk_tensor[0].detach().cpu().numpy()
                    inference_end_ns = time.time_ns()
                    target_times = timing["observation_time_ns"] + np.arange(
                        len(chunk), dtype=np.int64
                    ) * period_ns
                    robot_latency_ns = int(round(latency.robot_action_ms * 1e6))
                    gripper_latency_ns = int(round(latency.gripper_action_ms * 1e6))
                    stale_guard_ns = int(
                        round(latency.action_stale_guard_ms * 1e6)
                    )
                    common_lead_ns = (
                        max(robot_latency_ns, gripper_latency_ns)
                        + stale_guard_ns
                    )
                    accepted, stale = action_buffer.update(
                        chunk,
                        target_times,
                        query_id=query_id,
                        admission_cutoff_ns=inference_end_ns + common_lead_ns,
                    )
                    validated_actions.clear()
                    coverage_end = action_buffer.coverage_end_ns()
                    minimum_coverage = (
                        inference_end_ns
                        + common_lead_ns
                        + args.min_future_actions * period_ns
                    )
                    scheduled = (
                        accepted > 0
                        and coverage_end is not None
                        and coverage_end >= minimum_coverage
                    )
                    if not scheduled:
                        action_buffer.clear()
                        validated_actions.clear()
                        consecutive_rejections += 1
                        next_query_ns = 0
                    else:
                        consecutive_rejections = 0
                    event_log.write(
                        "policy_query",
                        query_id=query_id,
                        **timing,
                        inference_start_wall_time_ns=inference_start_ns,
                        inference_end_wall_time_ns=inference_end_ns,
                        inference_latency_ms=(inference_end_ns - inference_start_ns) / 1e6,
                        target_times_ns=target_times,
                        action_chunk=chunk,
                        actions_accepted=accepted,
                        common_stale_prefix=stale,
                        common_admission_lead_ms=common_lead_ns / 1e6,
                        scheduled=scheduled,
                        robot_state=observation["robot_state"],
                        skill=observation.get("skill"),
                    )
                    query_id += 1
                    next_query_ns = (
                        now_ns + args.query_interval_steps * period_ns
                        if scheduled
                        else 0
                    )
                except Exception as exc:
                    action_buffer.clear()
                    validated_actions.clear()
                    consecutive_rejections += 1
                    next_query_ns = 0
                    event_log.write(
                        "observation_or_query_rejected",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                last_prompt_token = prompt_token

            scheduled_action = action_buffer.next()
            if scheduled_action is not None:
                target_time_ns = scheduled_action.target_time_ns
                validated = validated_actions.get(target_time_ns)
                if validated is None:
                    reference = (
                        last_target_pose
                        if last_target_pose is not None
                        else np.asarray(robot.last_eef_pose, dtype=np.float64)
                    )
                    try:
                        validated = validate_absolute_action(
                            scheduled_action.action,
                            reference_pose=reference,
                            period_s=period_s,
                            limits=limits,
                        )
                        validated_actions[target_time_ns] = validated
                    except Exception as exc:
                        action_buffer.clear()
                        validated_actions.clear()
                        consecutive_rejections += 1
                        next_query_ns = 0
                        event_log.write(
                            "action_rejected",
                            target_time_ns=target_time_ns,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        scheduled_action = None

            if scheduled_action is not None:
                robot_latency_ns = int(round(latency.robot_action_ms * 1e6))
                gripper_latency_ns = int(round(latency.gripper_action_ms * 1e6))
                channels = sorted(
                    (
                        ("robot", robot_latency_ns),
                        ("gripper", gripper_latency_ns),
                    ),
                    key=lambda item: scheduled_action.target_time_ns - item[1],
                )
                dispatch_failed = False
                for channel, channel_latency_ns in channels:
                    if (
                        channel == "robot"
                        and scheduled_action.robot_dispatched
                    ) or (
                        channel == "gripper"
                        and scheduled_action.gripper_dispatched
                    ):
                        continue
                    command_deadline_ns = (
                        scheduled_action.target_time_ns - channel_latency_ns
                    )
                    command_start_ns = time.time_ns()
                    if command_start_ns < command_deadline_ns:
                        continue
                    lateness_ns = command_start_ns - command_deadline_ns
                    expired = (
                        command_start_ns >= scheduled_action.target_time_ns
                        or lateness_ns > args.max_action_lateness_ms * 1e6
                    )
                    if expired:
                        partial = bool(
                            scheduled_action.robot_dispatched
                            or scheduled_action.gripper_dispatched
                        )
                        action_buffer.clear()
                        validated_actions.clear()
                        consecutive_rejections += 1
                        next_query_ns = 0
                        event_log.write(
                            "stale_coordinated_action_discarded",
                            channel=channel,
                            partial_dispatch=partial,
                            target_time_ns=scheduled_action.target_time_ns,
                            command_deadline_ns=command_deadline_ns,
                            lateness_ms=lateness_ns / 1e6,
                        )
                        dispatch_failed = True
                        break
                    try:
                        if channel == "robot":
                            target_pose = np.eye(4)
                            target_pose[:3, :3] = validated.rotation_matrix
                            target_pose[:3, 3] = validated.position
                            send_ns = command_start_ns
                            if args.execute:
                                command_result = robot.control(
                                    controller_type="OSC_POSE",
                                    action=validated.deoxys_action(),
                                    controller_cfg=controller_cfg,
                                    control_gripper=False,
                                    enforce_control_frequency=False,
                                )
                                send_ns = command_result[
                                    "robot_command_wall_time_ns"
                                ]
                            last_target_pose = target_pose
                            action_buffer.mark_dispatched(
                                scheduled_action.target_time_ns, "robot"
                            )
                            event_log.write(
                                "robot_action",
                                target_time_ns=scheduled_action.target_time_ns,
                                command_deadline_ns=command_deadline_ns,
                                send_wall_time_ns=send_ns,
                                deadline_residual_ms=(
                                    send_ns - command_deadline_ns
                                )
                                / 1e6,
                                target_residual_ms=(
                                    send_ns - scheduled_action.target_time_ns
                                )
                                / 1e6,
                                query_id=scheduled_action.query_id,
                                chunk_index=scheduled_action.chunk_index,
                                policy_action=scheduled_action.action,
                                deoxys_action=validated.deoxys_action(),
                                executed=args.execute,
                            )
                        else:
                            sign = float(
                                np.sign(scheduled_action.action[-1]) or -1.0
                            )
                            changed = (
                                last_gripper_sign is None
                                or sign != last_gripper_sign
                            )
                            send_ns = command_start_ns
                            if changed and args.execute:
                                robot.gripper_control(sign)
                                send_ns = (
                                    robot.last_gripper_command_wall_time_ns
                                )
                            if changed:
                                last_gripper_sign = sign
                            action_buffer.mark_dispatched(
                                scheduled_action.target_time_ns, "gripper"
                            )
                            event_log.write(
                                "gripper_action",
                                target_time_ns=scheduled_action.target_time_ns,
                                command_deadline_ns=command_deadline_ns,
                                send_wall_time_ns=send_ns,
                                deadline_residual_ms=(
                                    send_ns - command_deadline_ns
                                )
                                / 1e6,
                                target_residual_ms=(
                                    send_ns - scheduled_action.target_time_ns
                                )
                                / 1e6,
                                gripper_sign=sign,
                                sign_changed=changed,
                                executed=bool(args.execute and changed),
                            )
                    except Exception as exc:
                        partial = bool(
                            scheduled_action.robot_dispatched
                            or scheduled_action.gripper_dispatched
                        )
                        action_buffer.clear()
                        validated_actions.clear()
                        consecutive_rejections += 1
                        next_query_ns = 0
                        event_log.write(
                            "action_dispatch_failed",
                            channel=channel,
                            partial_dispatch=partial,
                            target_time_ns=scheduled_action.target_time_ns,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        dispatch_failed = True
                        break

                if not dispatch_failed:
                    refreshed = action_buffer.next()
                    if (
                        refreshed is not None
                        and refreshed.target_time_ns
                        == scheduled_action.target_time_ns
                        and refreshed.complete
                    ):
                        action_buffer.remove(refreshed.target_time_ns)
                        validated_actions.pop(refreshed.target_time_ns, None)
                        executed_steps += 1
                        consecutive_rejections = 0
                        next_hold_ns = refreshed.target_time_ns + period_ns
                        event_log.write(
                            "coordinated_action_complete",
                            target_time_ns=refreshed.target_time_ns,
                            query_id=refreshed.query_id,
                            chunk_index=refreshed.chunk_index,
                        )

            if (
                args.execute
                and last_target_pose is not None
                and len(action_buffer) == 0
                and time.time_ns() >= next_hold_ns
            ):
                hold = np.concatenate(
                    [
                        last_target_pose[:3, 3],
                        Rotation.from_matrix(last_target_pose[:3, :3]).as_rotvec(),
                        [last_gripper_sign if last_gripper_sign is not None else -1.0],
                    ]
                )
                robot.control(
                    "OSC_POSE",
                    hold,
                    controller_cfg=controller_cfg,
                    control_gripper=False,
                    enforce_control_frequency=False,
                )
                next_hold_ns = time.time_ns() + period_ns
                event_log.write("controller_hold", target_pose=last_target_pose)
            time.sleep(0.002)
    except KeyboardInterrupt:
        event_log.write("keyboard_interrupt")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        event_log.write("rollout_stopped", executed_steps=executed_steps)
        if robot is not None:
            try:
                if args.execute and last_target_pose is not None:
                    termination = np.r_[
                        last_target_pose[:3, 3],
                        Rotation.from_matrix(last_target_pose[:3, :3]).as_rotvec(),
                        -1.0,
                    ]
                    robot.control(
                        "OSC_POSE",
                        termination,
                        controller_cfg=controller_cfg,
                        termination=True,
                        control_gripper=False,
                        enforce_control_frequency=False,
                    )
            finally:
                robot.close()
        if worker is not None:
            worker.stop()
        if camera is not None:
            camera.stop()
        event_log.close()
    print(f"finished {executed_steps} policy actions; log={event_log.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
