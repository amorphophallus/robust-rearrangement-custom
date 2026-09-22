"""Standalone Deoxys evaluation for RR absolute-pose RGBD policies.

This first hardware version intentionally supports one narrow contract:

* one-arm FurnitureBench tasks ``one_leg`` and ``round_table``;
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
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
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
from src.real.time_alignment import IndependentActionQueues, LatencyProfile


# Keep this target identical to
# deoxys.examples.run_deoxys_with_space_mouse_V3_record.RESET_JOINT_POSITIONS.
RESET_JOINT_POSITIONS = np.asarray(
    [
        0.0916502534874562,
        0.006205358472252432,
        -0.02085815329544379,
        -2.552429972459778,
        -0.010695882435351968,
        2.587622772050635,
        0.8472435743003388,
    ],
    dtype=np.float64,
)


class EvalCommandReader:
    """Read single-key eval commands without blocking the UMI scheduler."""

    def __init__(self):
        self._fd = None
        self._old_settings = None
        self.enabled = False

    def start(self):
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True
        return self

    def close(self):
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self.enabled = False

    def read_keys(self):
        keys = []
        if not self.enabled:
            return keys
        while select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if not char:
                break
            keys.append(char.lower())
        return keys


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


@dataclass(frozen=True)
class PolicyInferenceRequest:
    rollout_generation: int
    query_id: int
    observation: Mapping[str, Any]
    timing: Mapping[str, Any]
    period_ns: int
    warmstart_indices: Tuple[int, ...]
    warmstart_actions: np.ndarray


@dataclass(frozen=True)
class PolicyInferenceResult:
    request: PolicyInferenceRequest
    inference_start_ns: int
    inference_end_ns: int
    chunk: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class PolicyInferenceReset:
    reason: str


class AsyncPolicyInference:
    """Run policy inference off the hardware dispatch thread."""

    _STOP = object()

    def __init__(self, actor, *, device: str, binary_gripper: bool):
        from src.behavior.diffusion import DiffusionPolicy

        self.actor = actor
        self.device = device
        self.binary_gripper = bool(binary_gripper)
        self.is_diffusion = isinstance(actor, DiffusionPolicy)
        self._requests = queue.Queue()
        self._results = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run,
            name="rr-policy-inference",
            daemon=True,
        )
        self._thread.start()

    def submit(self, request: PolicyInferenceRequest) -> bool:
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            return False
        return True

    def poll(self) -> Optional[PolicyInferenceResult]:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def reset(self, reason: str) -> None:
        """Reset actor queues and warm-start state on the worker thread."""

        self._requests.put_nowait(PolicyInferenceReset(reason=str(reason)))

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        self._requests.put(self._STOP)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        active_generation = None
        previous_observation_time_ns = None
        while True:
            request = self._requests.get()
            if request is self._STOP:
                return
            if isinstance(request, PolicyInferenceReset):
                self.actor.reset()
                active_generation = None
                previous_observation_time_ns = None
                continue
            if request.rollout_generation != active_generation:
                self.actor.reset()
                active_generation = request.rollout_generation
                previous_observation_time_ns = None

            inference_start_ns = time.time_ns()
            try:
                policy_obs = _policy_observation(
                    request.observation,
                    actor=self.actor,
                    device=self.device,
                    binary_gripper=self.binary_gripper,
                )
                sampling_kwargs = {}
                if self.is_diffusion:
                    shift_steps = _prediction_shift_steps(
                        current_observation_time_ns=int(
                            request.timing["observation_time_ns"]
                        ),
                        previous_observation_time_ns=previous_observation_time_ns,
                        period_ns=request.period_ns,
                        default_steps=self.actor.action_horizon,
                    )
                    sampling_kwargs["warmstart_shift_steps"] = shift_steps
                    if request.warmstart_indices:
                        import torch

                        warm_actions = torch.as_tensor(
                            request.warmstart_actions,
                            device=self.device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        sampling_kwargs["warmstart_nactions"] = self.actor.normalizer(
                            warm_actions, "action", forward=True
                        )
                        sampling_kwargs["warmstart_indices"] = (
                            request.warmstart_indices
                        )
                chunk_tensor = self.actor.action_chunk(
                    policy_obs,
                    **sampling_kwargs,
                )
                if chunk_tensor.shape[0] != 1:
                    raise ValueError("real evaluation requires policy batch size 1")
                chunk = chunk_tensor[0].detach().cpu().numpy()
                inference_end_ns = time.time_ns()
                previous_observation_time_ns = int(
                    request.timing["observation_time_ns"]
                )
                result = PolicyInferenceResult(
                    request=request,
                    inference_start_ns=inference_start_ns,
                    inference_end_ns=inference_end_ns,
                    chunk=chunk,
                )
            except Exception as exc:
                inference_end_ns = time.time_ns()
                result = PolicyInferenceResult(
                    request=request,
                    inference_start_ns=inference_start_ns,
                    inference_end_ns=inference_end_ns,
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._results.put(result)


def _prediction_shift_steps(
    *,
    current_observation_time_ns: int,
    previous_observation_time_ns: Optional[int],
    period_ns: int,
    default_steps: int,
) -> int:
    """Convert actual observation elapsed time to a prediction-grid shift."""

    if previous_observation_time_ns is None:
        return int(default_steps)
    elapsed_ns = int(current_observation_time_ns) - int(
        previous_observation_time_ns
    )
    return max(0, int(round(elapsed_ns / int(period_ns))))


def _channel_action_expired(
    *,
    command_start_ns: int,
    target_time_ns: int,
    command_deadline_ns: int,
    max_lateness_ms: float,
) -> bool:
    return bool(
        int(command_start_ns) >= int(target_time_ns)
        or int(command_start_ns) - int(command_deadline_ns)
        > float(max_lateness_ms) * 1e6
    )


def _gripper_dispatch_decision(
    *,
    desired_sign: float,
    last_sign: Optional[float],
    command_start_ns: int,
    target_time_ns: int,
    command_deadline_ns: int,
    max_lateness_ms: float,
) -> Tuple[bool, bool]:
    """Return ``(changed, expired)``; same-sign no-ops never expire."""

    changed = last_sign is None or float(desired_sign) != float(last_sign)
    expired = changed and _channel_action_expired(
        command_start_ns=command_start_ns,
        target_time_ns=target_time_ns,
        command_deadline_ns=command_deadline_ns,
        max_lateness_ms=max_lateness_ms,
    )
    return changed, expired


def _queue_warmstart(
    action_queues: IndependentActionQueues,
    *,
    observation_time_ns: int,
    period_ns: int,
    pred_horizon: int,
    action_dim: int,
) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Map immutable future reservations onto one policy prediction grid."""

    indices = []
    actions = []
    for scheduled in action_queues.future_reservations(observation_time_ns):
        delta_ns = scheduled.target_time_ns - int(observation_time_ns)
        index = int(round(delta_ns / period_ns))
        if index < 0 or index >= pred_horizon:
            continue
        if abs(delta_ns - index * period_ns) > period_ns // 2:
            continue
        if index in indices:
            continue
        indices.append(index)
        actions.append(np.asarray(scheduled.action).copy())
    if not actions:
        return (), np.empty((0, action_dim), dtype=np.float32)
    return tuple(indices), np.asarray(actions, dtype=np.float32)


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


def _move_to_reset_joint_positions(
    robot_interface,
    joint_controller_cfg,
    *,
    timeout: float,
    tolerance: float,
    gripper_open: bool,
) -> bool:
    """Run the same operator-triggered joint reset used by data collection."""

    target = RESET_JOINT_POSITIONS
    action = target.tolist() + [-1.0 if gripper_open else 1.0]
    deadline = time.monotonic() + float(timeout)
    max_error = float("inf")
    while time.monotonic() < deadline:
        current_q = robot_interface.last_q
        if current_q is not None:
            max_error = float(
                np.max(np.abs(np.asarray(current_q, dtype=np.float64) - target))
            )
            if max_error < tolerance:
                print(
                    f"RESET reached; max_joint_error={max_error:.6f}",
                    flush=True,
                )
                return True
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=joint_controller_cfg,
        )
    print(
        f"RESET timed out; max_joint_error={max_error:.6f}",
        flush=True,
    )
    return False


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
    parser.add_argument(
        "--task",
        choices=("one_leg", "round_table"),
        default="one_leg",
        help="FurnitureBench task used by camera pose tracking and real annotation",
    )
    parser.add_argument("--interface-cfg", default="config/charmander.yml")
    parser.add_argument("--front-camera-serial", default="327122071654")
    parser.add_argument("--wrist-camera-serial", default="001622071252")
    parser.add_argument(
        "--latency-profile",
        type=Path,
        default=None,
        help=(
            "profile JSON, or a directory from which the newest profile measured "
            "on the local current date is selected"
        ),
    )
    parser.add_argument(
        "--execution-frequency",
        "--frequency",
        dest="frequency",
        type=float,
        default=5.0,
        help=(
            "robot action frequency in Hz; lower values stretch the complete "
            "UMI target-time action sequence for slower emergency-stop trials"
        ),
    )
    parser.add_argument("--query-interval-steps", type=int, default=4)
    parser.add_argument("--min-future-actions", type=int, default=2)
    parser.add_argument("--max-observation-age-ms", type=float, default=300.0)
    parser.add_argument("--max-action-lateness-ms", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-wall-time-s", type=float, default=180.0)
    parser.add_argument("--max-consecutive-rejections", type=int, default=20)
    parser.add_argument("--warmup-timeout-s", type=float, default=120.0)
    parser.add_argument("--start-delay-s", type=float, default=3.0)
    parser.add_argument("--reset-timeout-s", type=float, default=7.0)
    parser.add_argument("--reset-tolerance", type=float, default=1e-3)
    parser.add_argument("--keep-gripper-closed-during-reset", action="store_true")
    parser.add_argument(
        "--auto-begin",
        action="store_true",
        help="start immediately for non-interactive dry-runs; never use for first motion",
    )
    parser.add_argument("--controller-time-fraction", type=float, default=2.0)
    parser.add_argument("--prompt-depth-model", choices=("vits", "vitl", "vits-transparent"), default="vitl")
    parser.add_argument("--prompt-depth-device", default="cuda")
    parser.add_argument("--prompt-depth-max-size", type=int, default=448)
    parser.add_argument(
        "--show-input-dashboard",
        action="store_true",
        help=(
            "show one OpenCV page with every RGB-D, annotation, proprioceptive "
            "and action value used by each policy query"
        ),
    )
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        default=[0.30, -0.35, 0.00],
    )
    parser.add_argument(
        "--workspace-max",
        type=float,
        nargs=3,
        default=[0.75, 0.35, 0.60],
    )
    parser.add_argument("--min-ee-z", type=float, default=0.005)
    parser.add_argument("--max-translation-step-m", type=float, default=0.05)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.35)
    parser.add_argument("--max-translation-speed-m-s", type=float, default=0.25)
    parser.add_argument("--max-rotation-speed-rad-s", type=float, default=1.5)
    args = parser.parse_args(argv)
    if args.frequency <= 0 or args.query_interval_steps <= 0:
        parser.error("frequency and query interval must be positive")
    if (
        args.max_wall_time_s <= 0
        or args.max_consecutive_rejections <= 0
        or args.warmup_timeout_s <= 0
        or args.reset_timeout_s <= 0
        or args.reset_tolerance <= 0
    ):
        parser.error("watchdog limits must be positive")
    if args.execute:
        if args.latency_profile is None:
            parser.error("--execute requires --latency-profile")
    return args


def _initialize_policy_runtime(args):
    """Load CUDA policy only after the native RealSense pipelines are live."""

    actor, cfg = _load_actor(args.checkpoint, args.config, args.device)
    period_ns = int(round(1e9 / args.frequency))
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
    resolved_latency_profile = (
        LatencyProfile.resolve_path(args.latency_profile)
        if args.latency_profile is not None
        else None
    )
    if resolved_latency_profile is not None:
        print(f"resolved latency profile: {resolved_latency_profile}", flush=True)
    latency = (
        LatencyProfile.load(resolved_latency_profile)
        if resolved_latency_profile is not None
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
    limits = ActionSafetyLimits(
        workspace_min=np.asarray(args.workspace_min),
        workspace_max=np.asarray(args.workspace_max),
        min_ee_z=float(args.min_ee_z),
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
            "annotation_mode": _annotation_mode(cfg),
            "input_dashboard_enabled": args.show_input_dashboard,
            "input_video_mode": "record_each_rollout_then_save_with_s",
            "frequency": args.frequency,
            "execution_frequency_hz": args.frequency,
            "action_period_ms": period_ns / 1e6,
            "action_horizon": actor.action_horizon,
            "query_interval_steps": args.query_interval_steps,
            "latency_profile": (
                None if resolved_latency_profile is None else str(resolved_latency_profile)
            ),
            "latency_profile_requested": (
                None if args.latency_profile is None else str(args.latency_profile)
            ),
            "latency_source": latency.latency_source,
            "latency_basis": latency.basis,
            "action_stale_guard_ms": latency.action_stale_guard_ms,
            "workspace_min": limits.workspace_min,
            "workspace_max": limits.workspace_max,
            "min_ee_z": limits.min_ee_z,
            "max_translation_step_m": limits.max_translation_step_m,
            "max_rotation_step_rad": limits.max_rotation_step_rad,
            "max_translation_speed_m_s": limits.max_translation_speed_m_s,
            "max_rotation_speed_rad_s": limits.max_rotation_speed_rad_s,
        },
    )
    return actor, cfg, period_ns, latency, limits, event_log


def _start_camera_and_initialize_policy(camera, args):
    """Preserve the hardware-proven RealSense-before-CUDA ordering."""

    camera.start()
    return _initialize_policy_runtime(args)


def main(argv=None) -> int:
    args = _parse_args(argv)
    # Hardware imports stay local so alignment/tests do not require Deoxys or
    # RealSense.  This also gives a direct diagnosis for a wrong environment.
    try:
        import torch
        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils.config_utils import get_default_controller_config
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
    from src.real.input_dashboard import EvalInputDashboard, EvalInputVideoRecorder

    camera = None
    worker = None
    robot = None
    event_log = None
    command_reader = None
    dashboard = None
    video_recorder = None
    inference_worker = None
    try:
        # librealsense and CUDA allocator initialization conflict on this host
        # when CUDA wins the ordering race.  Start both pipelines first and keep
        # them running while the checkpoint is loaded.
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
        actor, cfg, period_ns, latency, limits, event_log = (
            _start_camera_and_initialize_policy(camera, args)
        )
        controller_cfg = _absolute_controller_config(args.controller_time_fraction)
        joint_controller_cfg = get_default_controller_config("JOINT_POSITION")
    except BaseException:
        if event_log is not None:
            event_log.close()
        if camera is not None:
            camera.stop()
        raise
    period_s = period_ns / 1e9
    print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN'} log={event_log.path}")

    robot_latency_ns = int(round(latency.robot_action_ms * 1e6))
    gripper_latency_ns = int(round(latency.gripper_action_ms * 1e6))
    stale_guard_ns = int(round(latency.action_stale_guard_ms * 1e6))
    common_lead_ns = max(robot_latency_ns, gripper_latency_ns) + stale_guard_ns
    action_queues = IndependentActionQueues(
        period_ns,
        arm_latency_ns=robot_latency_ns,
        gripper_latency_ns=gripper_latency_ns,
    )
    validated_actions = {}
    failed_channels = {}
    annotation_mode = _annotation_mode(cfg)
    dashboard = EvalInputDashboard(
        enabled=args.show_input_dashboard,
        checkpoint=args.checkpoint.expanduser().resolve(),
        actor=actor,
        capture_policy_inputs=True,
    )
    video_recorder = EvalInputVideoRecorder(
        enabled=True,
        fps=args.frequency / args.query_interval_steps,
    )
    annotation_session = None
    last_prompt_token = None
    last_target_pose = None
    last_reserved_pose = None
    last_gripper_sign = None
    query_id = 0
    rollout_generation = 0
    inference_inflight = False
    pending_dashboard = None
    pending_video = None
    executed_steps = 0
    rollout_executed_steps = 0
    rollout_index = 0
    rollout_active = False
    quit_requested = False
    next_query_ns = 0
    next_hold_ns = 0
    last_hold_status_ns = 0
    consecutive_rejections = 0
    rollout_started_monotonic = None
    stop_request = {"signal": None}
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def request_stop(signum, _frame):
        stop_request["signal"] = int(signum)

    signal.signal(signal.SIGTERM, request_stop)
    try:
        camera_info = camera.metadata()
        dashboard.set_camera_info(camera_info)
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
        inference_worker = AsyncPolicyInference(
            actor,
            device=args.device,
            binary_gripper=normalizer_expects_binary_gripper_width(
                actor.normalizer
            ),
        )
        kinematics = PandaKinematics()
        if model_requires_skill_input(cfg) or annotation_mode != "none":
            annotation_session = RealSkillAnnotationSession(
                args.task, camera_info, mode="online"
            )

        print("warming camera, PromptDA, and timestamped robot buffers...")
        warm_deadline = time.monotonic() + args.warmup_timeout_s
        result = None
        prompt_ready = False
        robot_records = []
        gripper_records = []
        while time.monotonic() < warm_deadline:
            worker.submit(camera.latest())
            result = worker.latest()
            prompt_ready = bool(
                result is not None
                and result.get("ready_wall_time_ns") is not None
                and not result.get("error")
                and {
                    "depth_image1",
                    "depth_image2",
                }.issubset(result.get("depths", {}))
            )
            robot_records = _timestamped_records(robot, "robot")
            gripper_records = _timestamped_records(robot, "gripper")
            if (
                prompt_ready
                and len(robot_records) >= 2
                and len(gripper_records) >= 2
            ):
                break
            time.sleep(0.005)
        else:
            prompt_error = None if result is None else result.get("error")
            event_log.write(
                "warmup_failed",
                warmup_timeout_s=args.warmup_timeout_s,
                prompt_ready=prompt_ready,
                prompt_error=prompt_error,
                robot_state_records=len(robot_records),
                gripper_state_records=len(gripper_records),
            )
            raise RuntimeError(
                "timed out warming hardware inputs: "
                f"PromptDA ready={prompt_ready} error={prompt_error!r}, "
                f"robot states={len(robot_records)}, "
                f"gripper states={len(gripper_records)}"
            )

        measured_gripper = robot.last_gripper_q
        if measured_gripper is not None:
            measured_width = float(np.asarray(measured_gripper).reshape(-1)[0])
            last_gripper_sign = (
                -1.0 if measured_width >= GRIPPER_OPEN_THRESHOLD_METERS else 1.0
            )
        command_reader = EvalCommandReader().start()
        if not command_reader.enabled and not args.auto_begin:
            raise RuntimeError(
                "interactive eval requires a TTY; use --auto-begin only for "
                "non-moving scripted dry-runs"
            )

        def reset_annotation(reason):
            if annotation_session is not None:
                annotation_session.reset()
                event_log.write("annotation_reset", reason=reason)

        def rollout_video_path(index):
            return event_log.path.with_name(
                f"{event_log.path.stem}-rollout-{index:03d}-rgbd-grid.mp4"
            )

        def discard_pending_video(reason):
            nonlocal pending_video
            if pending_video is None:
                return
            pending_path = Path(pending_video["pending_path"])
            try:
                if pending_path.exists():
                    pending_path.unlink()
                event_log.write(
                    "input_video_discarded",
                    rollout_index=pending_video["rollout_index"],
                    path=str(pending_path),
                    reason=reason,
                )
            except OSError as exc:
                event_log.write(
                    "input_video_discard_failed",
                    rollout_index=pending_video["rollout_index"],
                    path=str(pending_path),
                    reason=reason,
                    error=f"{type(exc).__name__}: {exc}",
                )
            pending_video = None

        def save_pending_video():
            nonlocal pending_video
            if rollout_active:
                print("SAVE refused: press e before s", flush=True)
                return
            if pending_video is None:
                print("SAVE ignored: no ended rollout is waiting", flush=True)
                return
            pending_path = Path(pending_video["pending_path"])
            final_path = Path(pending_video["final_path"])
            try:
                if final_path.exists():
                    raise FileExistsError(f"video already exists: {final_path}")
                pending_path.rename(final_path)
            except OSError as exc:
                event_log.write(
                    "input_video_save_failed",
                    rollout_index=pending_video["rollout_index"],
                    pending_path=str(pending_path),
                    final_path=str(final_path),
                    error=f"{type(exc).__name__}: {exc}",
                )
                print(f"SAVE VIDEO failed: {exc}", flush=True)
                return
            event_log.write(
                "input_video_saved",
                rollout_index=pending_video["rollout_index"],
                path=str(final_path),
                frame_count=pending_video["frame_count"],
                fps=pending_video["fps"],
            )
            print(f"SAVE VIDEO {final_path}", flush=True)
            pending_video = None

        def begin_rollout():
            nonlocal rollout_active, rollout_started_monotonic
            nonlocal rollout_executed_steps, rollout_index
            nonlocal consecutive_rejections, next_query_ns, next_hold_ns
            nonlocal last_prompt_token, last_target_pose, last_hold_status_ns
            nonlocal last_reserved_pose, rollout_generation, pending_dashboard
            nonlocal pending_video
            if rollout_active:
                print("BEGIN ignored: rollout is already active", flush=True)
                return
            if pending_video is not None:
                print(
                    "BEGIN discarding the previous unsaved video; press s after e "
                    "to keep a rollout",
                    flush=True,
                )
                discard_pending_video("next_rollout_started")
            action_queues.clear()
            validated_actions.clear()
            failed_channels.clear()
            inference_worker.reset("operator_begin")
            reset_annotation("operator_begin")
            rollout_index += 1
            rollout_generation += 1
            rollout_executed_steps = 0
            consecutive_rejections = 0
            next_query_ns = 0
            last_prompt_token = None
            last_hold_status_ns = 0
            pending_dashboard = None
            final_video_path = rollout_video_path(rollout_index)
            pending_video_path = final_video_path.with_name(
                f".{final_video_path.stem}.pending.mp4"
            )
            try:
                started_video_path = video_recorder.start(pending_video_path)
                if started_video_path is not None:
                    event_log.write(
                        "input_video_started",
                        rollout_index=rollout_index,
                        path=started_video_path,
                        fps=video_recorder.fps,
                    )
            except Exception as exc:
                failed_video = video_recorder.close()
                failed_path = failed_video.get("path")
                if failed_path is not None:
                    failed_path = Path(failed_path)
                    if failed_path.exists():
                        failed_path.unlink()
                event_log.write(
                    "input_video_failed",
                    rollout_index=rollout_index,
                    error=f"{type(exc).__name__}: {exc}",
                )
                print(f"INPUT VIDEO disabled for this rollout: {exc}", flush=True)
            warm_pose = np.asarray(robot.last_eef_pose, dtype=np.float64)
            last_target_pose = warm_pose.copy()
            last_reserved_pose = warm_pose.copy()
            if args.execute:
                warm_action = np.r_[
                    warm_pose[:3, 3],
                    Rotation.from_matrix(warm_pose[:3, :3]).as_rotvec(),
                    last_gripper_sign if last_gripper_sign is not None else -1.0,
                ]
                warm_timing = robot.control(
                    "OSC_POSE",
                    warm_action,
                    controller_cfg=controller_cfg,
                    control_gripper=False,
                    enforce_control_frequency=False,
                )
                event_log.write(
                    "controller_warmup",
                    rollout_index=rollout_index,
                    target_pose=warm_pose,
                    **warm_timing,
                )
            next_hold_ns = time.time_ns() + period_ns
            print(
                f"BEGIN rollout={rollout_index}; starting in "
                f"{args.start_delay_s:.1f}s at {args.frequency:g} Hz",
                flush=True,
            )
            start_deadline = time.monotonic() + args.start_delay_s
            while time.monotonic() < start_deadline:
                # Keep PromptDA paired with fresh camera frames during the
                # operator countdown; otherwise the first query sees a frame
                # older than --max-observation-age-ms.
                worker.submit(camera.latest())
                time.sleep(0.01)
            rollout_started_monotonic = time.monotonic()
            rollout_active = True
            event_log.write(
                "rollout_started",
                rollout_index=rollout_index,
                execution_frequency_hz=args.frequency,
            )

        def end_rollout(reason):
            nonlocal rollout_active, rollout_started_monotonic
            nonlocal last_target_pose, last_reserved_pose, next_hold_ns
            nonlocal quit_requested, pending_video
            if video_recorder.path is not None:
                video_result = video_recorder.close()
                event_log.write(
                    "input_video_stopped",
                    rollout_index=rollout_index,
                    **video_result,
                )
                pending_path = video_result.get("path")
                if (
                    pending_path is not None
                    and video_result.get("error") is None
                    and int(video_result.get("frame_count", 0)) > 0
                ):
                    pending_video = {
                        "rollout_index": rollout_index,
                        "pending_path": pending_path,
                        "final_path": str(rollout_video_path(rollout_index)),
                        "frame_count": int(video_result["frame_count"]),
                        "fps": float(video_result["fps"]),
                    }
                    event_log.write("input_video_pending_save", **pending_video)
                else:
                    if pending_path is not None and Path(pending_path).exists():
                        Path(pending_path).unlink()
            action_queues.clear()
            validated_actions.clear()
            failed_channels.clear()
            inference_worker.reset(reason)
            if rollout_active and args.execute:
                measured_pose = np.asarray(robot.last_eef_pose, dtype=np.float64)
                hold = np.r_[
                    measured_pose[:3, 3],
                    Rotation.from_matrix(measured_pose[:3, :3]).as_rotvec(),
                    last_gripper_sign if last_gripper_sign is not None else -1.0,
                ]
                robot.control(
                    "OSC_POSE",
                    hold,
                    controller_cfg=controller_cfg,
                    control_gripper=False,
                    enforce_control_frequency=False,
                )
                last_target_pose = measured_pose.copy()
                last_reserved_pose = measured_pose.copy()
                next_hold_ns = time.time_ns() + period_ns
            event_log.write(
                "rollout_ended",
                rollout_index=rollout_index,
                reason=reason,
                rollout_executed_steps=rollout_executed_steps,
                total_executed_steps=executed_steps,
                consecutive_rejections=consecutive_rejections,
            )
            reset_annotation(reason)
            rollout_active = False
            rollout_started_monotonic = None
            print(
                f"END rollout={rollout_index} reason={reason} "
                f"steps={rollout_executed_steps}; state=IDLE; "
                "press s to save the RGB-D video",
                flush=True,
            )
            if args.auto_begin:
                quit_requested = True

        print(
            "READY state=IDLE keys: r=reset joints, b=begin, e=end, "
            "s=save ended rollout video, q=quit; "
            f"execution_frequency={args.frequency:g} Hz",
            flush=True,
        )
        event_log.write(
            "evaluator_ready",
            execution_frequency_hz=args.frequency,
            interactive=command_reader.enabled,
        )
        auto_begin_pending = bool(args.auto_begin)
        while not quit_requested:
            if stop_request["signal"] is not None:
                event_log.write(
                    "signal_stop",
                    signal=stop_request["signal"],
                )
                if rollout_active:
                    end_rollout("signal_stop")
                break

            keys = command_reader.read_keys()
            if auto_begin_pending:
                keys.append("b")
                auto_begin_pending = False
            for key in keys:
                if key == "r":
                    if rollout_active:
                        print("RESET refused: press e before r", flush=True)
                        continue
                    if not args.execute:
                        print("RESET refused in DRY-RUN mode", flush=True)
                        continue
                    reset_annotation("operator_reset")
                    print("RESET moving to data-collection joint target", flush=True)
                    reset_ok = _move_to_reset_joint_positions(
                        robot,
                        joint_controller_cfg,
                        timeout=args.reset_timeout_s,
                        tolerance=args.reset_tolerance,
                        gripper_open=(
                            not args.keep_gripper_closed_during_reset
                        ),
                    )
                    last_target_pose = None
                    inference_worker.reset("operator_reset")
                    if reset_ok:
                        last_gripper_sign = (
                            1.0
                            if args.keep_gripper_closed_during_reset
                            else -1.0
                        )
                    event_log.write("joint_reset", succeeded=reset_ok)
                elif key == "b":
                    begin_rollout()
                elif key == "e":
                    if rollout_active:
                        end_rollout("operator_end")
                    else:
                        reset_annotation("operator_end_idle")
                        print("END state=IDLE; annotation reset", flush=True)
                elif key == "s":
                    save_pending_video()
                elif key == "q":
                    if rollout_active:
                        end_rollout("operator_quit")
                    quit_requested = True
                    print("QUIT requested", flush=True)

            if quit_requested:
                break
            if not rollout_active:
                camera_sample = camera.latest()
                worker.submit(camera_sample)
                time.sleep(0.01)
                continue
            if rollout_executed_steps >= args.max_steps:
                end_rollout("max_steps")
                continue
            if (
                time.monotonic() - rollout_started_monotonic
                >= args.max_wall_time_s
            ):
                event_log.write(
                    "watchdog_stop",
                    reason="max_wall_time",
                    consecutive_rejections=consecutive_rejections,
                )
                end_rollout("max_wall_time")
                continue
            if consecutive_rejections >= args.max_consecutive_rejections:
                event_log.write(
                    "watchdog_stop",
                    reason="consecutive_rejections",
                    consecutive_rejections=consecutive_rejections,
                )
                end_rollout("consecutive_rejections")
                continue
            camera_sample = camera.latest()
            worker.submit(camera_sample)
            prompt_result = worker.latest()
            now_ns = time.time_ns()
            prompt_token = None
            if prompt_result is not None and prompt_result.get("camera_sample") is not None:
                prompt_token = prompt_result["camera_sample"].get("front_frame_number")

            dispatch_failed = False
            while True:
                dispatch = action_queues.next_dispatch()
                if dispatch is None:
                    break
                command_start_ns = time.time_ns()
                if command_start_ns < dispatch.command_deadline_ns:
                    break
                scheduled_action = dispatch.scheduled
                target_time_ns = scheduled_action.target_time_ns
                lateness_ns = command_start_ns - dispatch.command_deadline_ns
                try:
                    validated = validated_actions[target_time_ns]
                    if dispatch.channel == "gripper":
                        sign = validated.gripper_sign
                        changed, expired = _gripper_dispatch_decision(
                            desired_sign=sign,
                            last_sign=last_gripper_sign,
                            command_start_ns=command_start_ns,
                            target_time_ns=target_time_ns,
                            command_deadline_ns=dispatch.command_deadline_ns,
                            max_lateness_ms=args.max_action_lateness_ms,
                        )
                        # A repeated gripper state is a timeline no-op.  It is
                        # consumed even when its theoretical send deadline has
                        # passed, and therefore cannot poison later arm events.
                        if expired:
                            failed_channels.setdefault(target_time_ns, set()).add(
                                "gripper"
                            )
                            complete = action_queues.consume(
                                "gripper", target_time_ns
                            )
                            consecutive_rejections += 1
                            next_query_ns = min(
                                next_query_ns, time.time_ns() + period_ns
                            )
                            event_log.write(
                                "stale_channel_action_discarded",
                                channel="gripper",
                                target_time_ns=target_time_ns,
                                command_deadline_ns=dispatch.command_deadline_ns,
                                lateness_ms=lateness_ns / 1e6,
                                query_id=scheduled_action.query_id,
                                chunk_index=scheduled_action.chunk_index,
                                arm_queue_pending=action_queues.pending_count("arm"),
                                gripper_queue_pending=action_queues.pending_count(
                                    "gripper"
                                ),
                            )
                        else:
                            send_ns = command_start_ns
                            if changed and args.execute:
                                robot.gripper_control(sign)
                                send_ns = robot.last_gripper_command_wall_time_ns
                            if changed:
                                last_gripper_sign = sign
                            complete = action_queues.consume(
                                "gripper", target_time_ns
                            )
                            event_log.write(
                                "gripper_action",
                                target_time_ns=target_time_ns,
                                command_deadline_ns=dispatch.command_deadline_ns,
                                send_wall_time_ns=send_ns,
                                dispatch_lateness_ms=(
                                    send_ns - dispatch.command_deadline_ns
                                )
                                / 1e6,
                                target_residual_ms=(send_ns - target_time_ns) / 1e6,
                                query_id=scheduled_action.query_id,
                                chunk_index=scheduled_action.chunk_index,
                                gripper_sign=sign,
                                sign_changed=changed,
                                no_op=not changed,
                                executed=bool(args.execute and changed),
                                arm_queue_pending=action_queues.pending_count("arm"),
                                gripper_queue_pending=action_queues.pending_count(
                                    "gripper"
                                ),
                            )
                    else:
                        expired = _channel_action_expired(
                            command_start_ns=command_start_ns,
                            target_time_ns=target_time_ns,
                            command_deadline_ns=dispatch.command_deadline_ns,
                            max_lateness_ms=args.max_action_lateness_ms,
                        )
                        if expired:
                            failed_channels.setdefault(target_time_ns, set()).add(
                                "arm"
                            )
                            complete = action_queues.consume("arm", target_time_ns)
                            consecutive_rejections += 1
                            next_query_ns = min(
                                next_query_ns, time.time_ns() + period_ns
                            )
                            event_log.write(
                                "stale_channel_action_discarded",
                                channel="arm",
                                target_time_ns=target_time_ns,
                                command_deadline_ns=dispatch.command_deadline_ns,
                                lateness_ms=lateness_ns / 1e6,
                                query_id=scheduled_action.query_id,
                                chunk_index=scheduled_action.chunk_index,
                                arm_queue_pending=action_queues.pending_count("arm"),
                                gripper_queue_pending=action_queues.pending_count(
                                    "gripper"
                                ),
                            )
                        else:
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
                            complete = action_queues.consume("arm", target_time_ns)
                            event_log.write(
                                "robot_action",
                                target_time_ns=target_time_ns,
                                command_deadline_ns=dispatch.command_deadline_ns,
                                send_wall_time_ns=send_ns,
                                dispatch_lateness_ms=(
                                    send_ns - dispatch.command_deadline_ns
                                )
                                / 1e6,
                                target_residual_ms=(send_ns - target_time_ns) / 1e6,
                                query_id=scheduled_action.query_id,
                                chunk_index=scheduled_action.chunk_index,
                                policy_action=scheduled_action.action,
                                deoxys_action=validated.deoxys_action(),
                                executed=args.execute,
                                arm_queue_pending=action_queues.pending_count("arm"),
                                gripper_queue_pending=action_queues.pending_count(
                                    "gripper"
                                ),
                            )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    event_log.write(
                        "action_dispatch_failed",
                        channel=dispatch.channel,
                        target_time_ns=target_time_ns,
                        query_id=scheduled_action.query_id,
                        chunk_index=scheduled_action.chunk_index,
                        error=error,
                    )
                    print(
                        f"ERROR dispatch channel={dispatch.channel}: {error}",
                        flush=True,
                    )
                    dispatch_failed = True
                    break

                if complete:
                    failures = sorted(failed_channels.pop(target_time_ns, set()))
                    validated_actions.pop(target_time_ns, None)
                    next_hold_ns = target_time_ns + period_ns
                    if failures:
                        event_log.write(
                            "coordinated_action_incomplete",
                            target_time_ns=target_time_ns,
                            query_id=scheduled_action.query_id,
                            chunk_index=scheduled_action.chunk_index,
                            failed_channels=failures,
                        )
                    else:
                        executed_steps += 1
                        rollout_executed_steps += 1
                        consecutive_rejections = 0
                        event_log.write(
                            "coordinated_action_complete",
                            target_time_ns=target_time_ns,
                            query_id=scheduled_action.query_id,
                            chunk_index=scheduled_action.chunk_index,
                            arm_queue_pending=action_queues.pending_count("arm"),
                            gripper_queue_pending=action_queues.pending_count(
                                "gripper"
                            ),
                        )
                        print(
                            f"STEP rollout={rollout_index} "
                            f"step={rollout_executed_steps}/{args.max_steps} "
                            f"total={executed_steps} query={scheduled_action.query_id} "
                            f"chunk={scheduled_action.chunk_index} "
                            f"xyz={last_target_pose[:3, 3].round(4).tolist()}",
                            flush=True,
                        )

            if dispatch_failed:
                end_rollout("action_dispatch_failed")
                continue

            next_dispatch = action_queues.next_dispatch()
            result_slack_ns = (
                None
                if next_dispatch is None
                else next_dispatch.command_deadline_ns - time.time_ns()
            )
            inference_result = (
                inference_worker.poll()
                if result_slack_ns is None or result_slack_ns > 20_000_000
                else None
            )
            if inference_result is not None:
                inference_inflight = False
                request = inference_result.request
                if (
                    rollout_active
                    and request.rollout_generation == rollout_generation
                ):
                    if inference_result.error is not None:
                        consecutive_rejections += 1
                        next_query_ns = min(
                            next_query_ns, time.time_ns() + period_ns
                        )
                        event_log.write(
                            "policy_query_completed",
                            query_id=request.query_id,
                            succeeded=False,
                            error=inference_result.error,
                            inference_latency_ms=(
                                inference_result.inference_end_ns
                                - inference_result.inference_start_ns
                            )
                            / 1e6,
                        )
                    else:
                        chunk = inference_result.chunk
                        target_times = int(
                            request.timing["observation_time_ns"]
                        ) + np.arange(len(chunk), dtype=np.int64) * period_ns
                        plan = action_queues.plan_update(
                            chunk,
                            target_times,
                            query_id=request.query_id,
                            admission_cutoff_ns=(
                                inference_result.inference_end_ns + common_lead_ns
                            ),
                        )
                        valid_prefix = []
                        reference = (
                            last_reserved_pose
                            if last_reserved_pose is not None
                            else np.asarray(robot.last_eef_pose, dtype=np.float64)
                        )
                        validation_error = None
                        for scheduled in plan.candidates:
                            try:
                                validated = validate_absolute_action(
                                    scheduled.action,
                                    reference_pose=reference,
                                    period_s=period_s,
                                    limits=limits,
                                )
                            except Exception as exc:
                                validation_error = f"{type(exc).__name__}: {exc}"
                                consecutive_rejections += 1
                                event_log.write(
                                    "action_rejected",
                                    target_time_ns=scheduled.target_time_ns,
                                    query_id=scheduled.query_id,
                                    chunk_index=scheduled.chunk_index,
                                    error=validation_error,
                                )
                                break
                            valid_prefix.append(scheduled)
                            validated_actions[scheduled.target_time_ns] = validated
                            reference = np.eye(4)
                            reference[:3, :3] = validated.rotation_matrix
                            reference[:3, 3] = validated.position

                        action_queues.reserve(valid_prefix)
                        if valid_prefix:
                            last_reserved_pose = reference
                        coverage_end = action_queues.coverage_end_ns(
                            inference_result.inference_end_ns
                        )
                        minimum_coverage = (
                            inference_result.inference_end_ns
                            + common_lead_ns
                            + args.min_future_actions * period_ns
                        )
                        scheduled = bool(
                            coverage_end is not None
                            and coverage_end >= minimum_coverage
                        )
                        if not scheduled or validation_error is not None:
                            next_query_ns = min(
                                next_query_ns, time.time_ns() + period_ns
                            )
                        observation = request.observation
                        if video_recorder.active:
                            video_recorder.submit(
                                dashboard.policy_inputs_snapshot()
                            )
                        skill = observation.get("skill")
                        gripper_prediction = np.sign(chunk[:, -1])
                        near_gripper_prediction = gripper_prediction[
                            plan.stale : min(
                                plan.stale + args.query_interval_steps,
                                len(gripper_prediction),
                            )
                        ]
                        event_log.write(
                            "policy_query_completed",
                            query_id=request.query_id,
                            succeeded=True,
                            **request.timing,
                            inference_start_wall_time_ns=(
                                inference_result.inference_start_ns
                            ),
                            inference_end_wall_time_ns=(
                                inference_result.inference_end_ns
                            ),
                            inference_latency_ms=(
                                inference_result.inference_end_ns
                                - inference_result.inference_start_ns
                            )
                            / 1e6,
                            target_times_ns=target_times,
                            action_chunk=chunk,
                            actions_accepted=len(valid_prefix),
                            common_stale_prefix=plan.stale,
                            occupied_preserved=plan.occupied,
                            common_admission_lead_ms=common_lead_ns / 1e6,
                            scheduled=scheduled,
                            validation_error=validation_error,
                            arm_queue_pending=action_queues.pending_count("arm"),
                            gripper_queue_pending=action_queues.pending_count(
                                "gripper"
                            ),
                            immutable_coverage_end_ns=coverage_end,
                            warmstart_mapped_count=len(request.warmstart_indices),
                            robot_state=observation["robot_state"],
                            skill=skill,
                            skill_state=observation.get("skill_state"),
                            assembly_step=observation.get("assembly_step"),
                            guidance_point=observation.get("guidance_point"),
                            guidance_point_2d=observation.get("guidance_point_2d"),
                            parts_poses=observation.get("parts_poses"),
                            parts_founds=observation.get("parts_founds"),
                            parts_pose_valid=observation.get("parts_pose_valid"),
                            real_annotation_debug=observation.get(
                                "real_annotation_debug"
                            ),
                            annotation_mode=annotation_mode,
                            gripper_open_predictions=int(
                                np.count_nonzero(gripper_prediction < 0)
                            ),
                            gripper_closed_predictions=int(
                                np.count_nonzero(gripper_prediction > 0)
                            ),
                            gripper_close_prediction_count=int(
                                np.sum(gripper_prediction >= 0)
                            ),
                            gripper_close_near_window_count=int(
                                np.sum(near_gripper_prediction >= 0)
                            ),
                        )
                        print(
                            f"QUERY rollout={rollout_index} q={request.query_id} "
                            f"accepted={len(valid_prefix)} stale={plan.stale} "
                            f"occupied={plan.occupied} scheduled={scheduled} "
                            f"arm_q={action_queues.pending_count('arm')} "
                            f"gripper_q={action_queues.pending_count('gripper')} "
                            "close(all/next-window)="
                            f"{int(np.sum(gripper_prediction >= 0))}/"
                            f"{int(np.sum(near_gripper_prediction >= 0))} "
                            f"inference={(inference_result.inference_end_ns - inference_result.inference_start_ns) / 1e6:.1f}ms",
                            flush=True,
                        )
                        pending_dashboard = {
                            "query_id": request.query_id,
                            "observation": observation,
                            "action_chunk": chunk,
                            "timing": request.timing,
                            "inference_latency_ms": (
                                inference_result.inference_end_ns
                                - inference_result.inference_start_ns
                            )
                            / 1e6,
                            "actions_accepted": len(valid_prefix),
                            "common_stale_prefix": plan.stale,
                            "query_interval_steps": args.query_interval_steps,
                            "scheduled": scheduled,
                            "annotation_mode": annotation_mode,
                            "occupied_preserved": plan.occupied,
                            "warmstart_mapped_count": len(
                                request.warmstart_indices
                            ),
                        }
                else:
                    event_log.write(
                        "policy_query_result_discarded",
                        query_id=request.query_id,
                        result_rollout_generation=request.rollout_generation,
                        active_rollout_generation=rollout_generation,
                    )

            now_ns = time.time_ns()
            next_dispatch = action_queues.next_dispatch()
            query_slack_ns = (
                None
                if next_dispatch is None
                else next_dispatch.command_deadline_ns - now_ns
            )
            if (
                prompt_result is not None
                and prompt_token != last_prompt_token
                and now_ns >= next_query_ns
                and not inference_inflight
                and (query_slack_ns is None or query_slack_ns > 50_000_000)
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
                    warmstart_indices, warmstart_actions = _queue_warmstart(
                        action_queues,
                        observation_time_ns=timing["observation_time_ns"],
                        period_ns=period_ns,
                        pred_horizon=actor.pred_horizon,
                        action_dim=actor.action_dim,
                    )
                    request = PolicyInferenceRequest(
                        rollout_generation=rollout_generation,
                        query_id=query_id,
                        observation=observation,
                        timing=timing,
                        period_ns=period_ns,
                        warmstart_indices=warmstart_indices,
                        warmstart_actions=warmstart_actions,
                    )
                    if not inference_worker.submit(request):
                        raise RuntimeError("inference worker request queue is full")
                    inference_inflight = True
                    submitted_ns = time.time_ns()
                    event_log.write(
                        "policy_query_submitted",
                        query_id=query_id,
                        rollout_generation=rollout_generation,
                        observation_time_ns=timing["observation_time_ns"],
                        submitted_wall_time_ns=submitted_ns,
                        query_interval_steps=args.query_interval_steps,
                        arm_queue_pending=action_queues.pending_count("arm"),
                        gripper_queue_pending=action_queues.pending_count(
                            "gripper"
                        ),
                        warmstart_mapped_count=len(warmstart_indices),
                    )
                    query_id += 1
                    next_query_ns = (
                        submitted_ns + args.query_interval_steps * period_ns
                    )
                except Exception as exc:
                    consecutive_rejections += 1
                    next_query_ns = time.time_ns() + period_ns
                    error = f"{type(exc).__name__}: {exc}"
                    event_log.write(
                        "observation_or_query_rejected",
                        error=error,
                    )
                    if (
                        consecutive_rejections == 1
                        or consecutive_rejections % 5 == 0
                    ):
                        print(
                            f"REJECT observation/query count={consecutive_rejections}/"
                            f"{args.max_consecutive_rejections}: {error}",
                            flush=True,
                        )
                last_prompt_token = prompt_token

            action_queues.prune_reservations(time.time_ns())

            if pending_dashboard is not None:
                next_dispatch = action_queues.next_dispatch()
                dashboard_slack_ns = (
                    None
                    if next_dispatch is None
                    else next_dispatch.command_deadline_ns - time.time_ns()
                )
                if dashboard_slack_ns is None or dashboard_slack_ns > 50_000_000:
                    try:
                        coverage_end = action_queues.coverage_end_ns(
                            time.time_ns()
                        )
                        dashboard_error = dashboard.show(
                            **pending_dashboard,
                            arm_queue_pending=action_queues.pending_count("arm"),
                            gripper_queue_pending=action_queues.pending_count(
                                "gripper"
                            ),
                            immutable_coverage_ms=(
                                0.0
                                if coverage_end is None
                                else max(0.0, (coverage_end - time.time_ns()) / 1e6)
                            ),
                        )
                        if dashboard_error is not None:
                            event_log.write(
                                "input_dashboard_failed",
                                error=dashboard_error,
                            )
                            print(
                                f"INPUT DASHBOARD disabled: {dashboard_error}",
                                flush=True,
                            )
                    except Exception as exc:
                        event_log.write(
                            "input_dashboard_failed",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        dashboard.close()
                    pending_dashboard = None

            if (
                args.execute
                and last_target_pose is not None
                and len(action_queues) == 0
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
                hold_sent_ns = time.time_ns()
                next_hold_ns = hold_sent_ns + period_ns
                event_log.write("controller_hold", target_pose=last_target_pose)
                if hold_sent_ns - last_hold_status_ns >= int(1e9):
                    last_hold_status_ns = hold_sent_ns
                    print(
                        f"HOLD rollout={rollout_index} "
                        f"xyz={last_target_pose[:3, 3].round(4).tolist()} "
                        f"rejections={consecutive_rejections}",
                        flush=True,
                    )
            time.sleep(0.002)
    except KeyboardInterrupt:
        event_log.write("keyboard_interrupt")
        if rollout_active:
            end_rollout("keyboard_interrupt")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        event_log.write("rollout_stopped", executed_steps=executed_steps)
        if command_reader is not None:
            command_reader.close()
        if inference_worker is not None:
            inference_worker.stop()
        if dashboard is not None:
            dashboard.close()
        if video_recorder is not None:
            video_result = video_recorder.close()
            if video_result["path"] is not None:
                unfinished_path = Path(video_result["path"])
                if unfinished_path.exists():
                    unfinished_path.unlink()
                event_log.write(
                    "input_video_discarded",
                    path=str(unfinished_path),
                    reason="evaluator_exit_during_rollout",
                    frame_count=video_result["frame_count"],
                    error=video_result["error"],
                )
        if pending_video is not None:
            discard_pending_video("evaluator_exit_without_save")
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
