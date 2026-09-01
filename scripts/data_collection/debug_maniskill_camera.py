#!/usr/bin/env python3
"""Replay a successful ManiSkill rollout and tune the front camera in GUI."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import pickle
import re
import socket
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PPO_TASKS = {
    "LiftPegUpright-v1",
    "PickCube-v1",
    "PokeCube-v1",
    "PullCube-v1",
    "PushCube-v1",
    "StackCube-v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_success_trajectory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rb") as stream:
        trajectory = pickle.load(stream)
    if not isinstance(trajectory, dict):
        raise TypeError("trajectory must be a dictionary")
    if trajectory.get("success") is not True:
        raise ValueError("camera tuning requires a successful trajectory")
    if trajectory.get("annotation_source") != "scripted":
        raise ValueError("camera tuning requires annotation_source=scripted")
    if not trajectory.get("task"):
        raise ValueError("trajectory is missing task")
    return trajectory


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("camera quaternion is invalid")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_proposal(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    fov_radians: float,
    target_distance: float,
) -> dict[str, Any]:
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("camera position must contain three finite values")
    if not np.isfinite(fov_radians) or not 0 < fov_radians < math.pi:
        raise ValueError("camera FOV must be in (0, pi)")
    if not np.isfinite(target_distance) or target_distance <= 0:
        raise ValueError("target distance must be positive")
    rotation = quaternion_wxyz_to_matrix(quaternion_wxyz)
    # SAPIEN camera convention: the first rotation column is forward.
    forward = rotation[:, 0]
    target = position + target_distance * forward
    return {
        "eye_world": position.tolist(),
        "target_world": target.tolist(),
        "fov_radians": float(fov_radians),
        "fov_degrees": float(math.degrees(fov_radians)),
        "target_distance": float(target_distance),
    }


def _format_tuple(values: list[float]) -> str:
    return "(" + ", ".join(f"{float(value):.9g}" for value in values) + ")"


def camera_contract_lines(proposal: Mapping[str, Any]) -> list[str]:
    return [
        f"RR_FRONT_EYE_WORLD = {_format_tuple(proposal['eye_world'])}",
        f"RR_FRONT_TARGET_WORLD = {_format_tuple(proposal['target_world'])}",
        "RR_FRONT_FOV_RADIANS = "
        f"math.radians({float(proposal['fov_degrees']):.9g})",
    ]


def apply_camera_contract(path: Path, proposal: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    original_mode = path.stat().st_mode
    text = path.read_text()
    replacements = {
        "RR_FRONT_EYE_WORLD": camera_contract_lines(proposal)[0],
        "RR_FRONT_TARGET_WORLD": camera_contract_lines(proposal)[1],
        "RR_FRONT_FOV_RADIANS": camera_contract_lines(proposal)[2],
    }
    for name, replacement in replacements.items():
        text, count = re.subn(
            rf"(?m)^{name}\s*=.*$",
            replacement,
            text,
        )
        if count != 1:
            raise ValueError(f"expected exactly one {name} assignment, found {count}")
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_capture(path: Path, capture: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing existing camera capture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(capture, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rollout_provenance(task: str, checkpoint: Path | None) -> dict[str, Any]:
    """Resolve replay provenance without requiring a fake MP checkpoint."""

    if task in PPO_TASKS:
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(
                f"PPO camera replay requires a checkpoint: {checkpoint}"
            )
        return {
            "rollout_source": "official_state_ppo",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_path = str(checkpoint.resolve())
        checkpoint_hash = sha256_file(checkpoint)
    else:
        checkpoint_path = None
        checkpoint_hash = None
    return {
        "rollout_source": "bundled_panda_motion_planning_solver",
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_hash,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Required for PPO tasks; optional provenance for MP trajectories.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--playback-fps", type=float, default=5.0)
    parser.add_argument("--end-hold-seconds", type=float, default=1.0)
    parser.add_argument("--marker-half-size", type=float, default=0.012)
    parser.add_argument(
        "--scroll-speed",
        type=float,
        default=0.02,
        help="camera travel in metres per wheel notch; Shift applies another 0.1x",
    )
    parser.add_argument(
        "--viewer-size",
        type=int,
        default=800,
        help="square viewer width/height; its full viewport represents the 224x224 crop",
    )
    # The SAPIEN control window queries the Segmentation picture on mouse
    # clicks. The minimal pack omits it and crashes during interactive orbit.
    parser.add_argument("--shader", default="default")
    parser.add_argument(
        "--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cpu"
    )
    # Retained so older launch commands fail neither parsing nor provenance
    # checks. Recorded-state playback does not run the policy on this device.
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help=argparse.SUPPRESS
    )
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--camera-contract", type=Path)
    parser.add_argument("--apply-on-confirm", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.playback_fps <= 0:
        parser.error("--playback-fps must be positive")
    if args.end_hold_seconds < 0:
        parser.error("--end-hold-seconds cannot be negative")
    if args.marker_half_size <= 0:
        parser.error("--marker-half-size must be positive")
    if args.scroll_speed <= 0:
        parser.error("--scroll-speed must be positive")
    if args.viewer_size <= 0:
        parser.error("--viewer-size must be positive")
    if args.apply_on_confirm and args.camera_contract is None:
        parser.error("--apply-on-confirm requires --camera-contract")
    return args


def recorded_frame_count(trajectory: Mapping[str, Any]) -> int:
    observations = trajectory.get("observations")
    actions = trajectory.get("actions")
    if not isinstance(observations, list) or len(observations) < 2:
        raise ValueError("recorded-state playback requires at least two observations")
    if not isinstance(actions, list) or len(actions) + 1 != len(observations):
        raise ValueError("recorded-state playback requires T actions and T+1 observations")
    return len(observations)


def recorded_qpos(observation: Mapping[str, Any]) -> np.ndarray:
    state = observation.get("robot_state")
    if not isinstance(state, Mapping):
        raise ValueError("recorded observation is missing robot_state")
    try:
        qpos = np.concatenate(
            [
                np.asarray(state["joint_positions"], dtype=np.float32).reshape(-1),
                np.asarray(state["gripper_finger_1_pos"], dtype=np.float32).reshape(
                    -1
                ),
                np.asarray(state["gripper_finger_2_pos"], dtype=np.float32).reshape(
                    -1
                ),
            ]
        )
    except KeyError as error:
        raise ValueError(f"recorded robot_state is missing {error.args[0]}") from error
    if qpos.shape != (9,) or not np.isfinite(qpos).all():
        raise ValueError(f"recorded Panda qpos must be nine finite values, got {qpos}")
    return qpos


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trajectory = load_success_trajectory(args.trajectory)
    frame_count = recorded_frame_count(trajectory)
    seed = args.seed
    if seed is None:
        seed = trajectory.get("diagnostic_seed")
    if seed is None:
        raise ValueError("trajectory has no diagnostic_seed; pass --seed explicitly")
    provenance = rollout_provenance(trajectory["task"], args.checkpoint)
    if args.output.exists() and not args.overwrite_output:
        raise FileExistsError(f"refusing existing camera capture: {args.output}")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "task": trajectory["task"],
                    "seed": int(seed),
                    "success": True,
                    "annotation_source": "scripted",
                    "replay_mode": "recorded_state",
                    "recorded_frames": frame_count,
                    "trajectory_sha256": sha256_file(args.trajectory),
                    **provenance,
                },
                indent=2,
            )
        )
        return 0

    import gymnasium as gym
    import sapien
    import torch
    from sapien.utils.viewer.control_window import ControlWindow

    import mani_skill.envs  # noqa: F401
    from mani_skill.trajectory.pickle.camera_contract import (
        rr_front_camera_parameters,
    )
    from mani_skill.trajectory.pickle.task_registry import get_pickle_task_spec
    from mani_skill.trajectory.pickle.transforms import (
        as_numpy,
        matrix_to_pose,
        pose_to_matrix,
        xyzw_to_wxyz,
    )
    from mani_skill.utils.sapien_utils import look_at

    front_eye_world, front_target_world, front_fov_radians = (
        rr_front_camera_parameters(trajectory["task"])
    )
    target_distance = float(
        np.linalg.norm(
            np.asarray(front_target_world) - np.asarray(front_eye_world)
        )
    )
    env = gym.make(
        trajectory["task"],
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda_wristcam",
        reward_mode="sparse",
        render_mode="human",
        sim_backend=args.sim_backend,
        human_render_camera_configs={"shader_pack": args.shader},
        viewer_camera_configs={
            "shader_pack": args.shader,
            "fov": float(front_fov_radians),
            "near": 0.001,
            "far": 10.0,
        },
    )
    base_env = env.unwrapped
    observations = trajectory["observations"]
    task_spec = get_pickle_task_spec(trajectory["task"])
    current_step = 0
    current_skill = None
    current_target = None
    finished = False
    hold_until = 0.0
    paused = bool(args.start_paused)
    next_step_at = time.perf_counter()

    def apply_recorded_frame(index: int) -> None:
        nonlocal current_step, current_skill, current_target
        observation = observations[index]
        if not isinstance(observation, Mapping):
            raise ValueError(f"recorded observation {index} is not a mapping")
        qpos = recorded_qpos(observation)
        qpos_tensor = torch.as_tensor(qpos, device=base_env.device).reshape(1, -1)
        base_env.agent.robot.set_qpos(qpos_tensor)
        base_env.agent.robot.set_qvel(torch.zeros_like(qpos_tensor))

        world_base = as_numpy(
            base_env.agent.robot.pose.to_transformation_matrix(), dtype=np.float64
        )[0]
        parts = np.asarray(observation.get("parts_poses"), dtype=np.float64)
        expected_parts_shape = (7 * len(task_spec.parts),)
        if parts.shape != expected_parts_shape or not np.isfinite(parts).all():
            raise ValueError(
                f"recorded parts_poses must have shape {expected_parts_shape}, "
                f"got {parts.shape}"
            )
        for part, part_pose in zip(task_spec.parts, parts.reshape(-1, 7)):
            world_part = world_base @ pose_to_matrix(part_pose[:3], part_pose[3:])
            world_position, world_quaternion = matrix_to_pose(world_part)
            actor = getattr(base_env, part.env_attribute)
            actor.set_pose(
                sapien.Pose(
                    p=world_position,
                    q=xyzw_to_wxyz(world_quaternion),
                )
            )

        point = observation.get("guidance_point_clean")
        if point is None:
            point = observation.get("guidance_point")
        if point is None:
            current_target = None
        else:
            point = np.asarray(point, dtype=np.float64)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(
                    f"recorded guidance point at frame {index} is invalid: {point}"
                )
            current_target = (world_base @ np.concatenate([point, [1.0]]))[:3]
        skill = observation.get("skill")
        current_skill = None if skill is None else str(skill)
        current_step = index

    def reset_rollout() -> None:
        nonlocal finished, hold_until, next_step_at
        apply_recorded_frame(0)
        finished = False
        hold_until = 0.0
        next_step_at = time.perf_counter()

    try:
        env.reset(seed=int(seed))
        reset_rollout()
        viewer = base_env.render_human()
        # SAPIEN Viewer.set_scenes() unconditionally resets the interactive
        # camera to a 90-degree vertical FOV.  Apply the dataset contract only
        # after render_human() has attached the scene, and use a square viewport
        # so the complete GUI view represents the final 224x224 front image.
        viewer.resolution = (args.viewer_size, args.viewer_size)
        viewer.window.set_camera_parameters(
            0.001, 10.0, float(front_fov_radians)
        )
        control_window = next(
            (plugin for plugin in viewer.plugins if isinstance(plugin, ControlWindow)),
            None,
        )
        if control_window is None:
            raise RuntimeError("SAPIEN viewer has no ControlWindow plugin")
        control_window.scroll_speed = float(args.scroll_speed)
        viewer.set_camera_pose(
            look_at(front_eye_world, front_target_world).sp
        )
        marker = viewer.add_bounding_box(
            sapien.Pose(current_target if current_target is not None else [1000] * 3),
            np.full(3, args.marker_half_size),
            np.array([1.0, 0.0, 0.0]),
            line_width=4.0,
        )
        print(
            "Controls: drag/orbit/zoom with the mouse; P pause/resume; "
            "R restart; C confirm+save; Q quit without saving; "
            "hold Shift for 10x finer camera motion."
        )
        print(
            f"task={trajectory['task']} seed={seed} source=scripted "
            f"backend={args.sim_backend} replay=recorded_state "
            f"frames={frame_count} rollout_source={provenance['rollout_source']} "
            f"scroll_speed={control_window.scroll_speed}m/notch"
        )

        while not viewer.closed:
            base_env.render_human()
            if viewer.window.key_press("q"):
                print("Quit without saving a camera view.")
                return 1
            if viewer.window.key_press("p"):
                paused = not paused
                print(f"paused={paused} step={current_step} skill={current_skill}")
            if viewer.window.key_press("r"):
                reset_rollout()
                print("restarted trajectory")
            if viewer.window.key_press("c"):
                camera_pose = viewer.window.get_camera_pose()
                viewer_fov_radians = float(viewer.window.fovy)
                if not math.isclose(
                    viewer_fov_radians,
                    float(front_fov_radians),
                    abs_tol=1e-6,
                    rel_tol=0.0,
                ):
                    raise RuntimeError(
                        "viewer FOV drifted away from the locked dataset contract: "
                        f"viewer={math.degrees(viewer_fov_radians):.6f}deg "
                        f"contract={math.degrees(front_fov_radians):.6f}deg"
                    )
                proposal = camera_proposal(
                    np.asarray(camera_pose.p),
                    np.asarray(camera_pose.q),
                    float(front_fov_radians),
                    target_distance,
                )
                capture = {
                    "schema": "rr-maniskill-front-camera-view-v1",
                    "saved_at": datetime.now().astimezone().isoformat(),
                    "host": socket.gethostname(),
                    "task": trajectory["task"],
                    "seed": int(seed),
                    "annotation_source": "scripted",
                    "replay_mode": "recorded_state",
                    "viewer_scroll_speed_metres_per_notch": float(
                        control_window.scroll_speed
                    ),
                    "viewer_resolution": [args.viewer_size, args.viewer_size],
                    "viewer_contract_fov_locked": True,
                    "trajectory": str(args.trajectory.resolve()),
                    "trajectory_sha256": sha256_file(args.trajectory),
                    **provenance,
                    "rollout_step_at_capture": current_step,
                    "skill_at_capture": current_skill,
                    "guidance_target_world_at_capture": (
                        None if current_target is None else current_target.tolist()
                    ),
                    "viewer_pose_wxyz": {
                        "position": np.asarray(camera_pose.p).tolist(),
                        "quaternion": np.asarray(camera_pose.q).tolist(),
                    },
                    "previous_front_camera": {
                        "eye_world": list(front_eye_world),
                        "target_world": list(front_target_world),
                        "fov_radians": float(front_fov_radians),
                        "fov_degrees": float(math.degrees(front_fov_radians)),
                    },
                    "proposed_front_camera": proposal,
                    "camera_contract_lines": camera_contract_lines(proposal),
                }
                write_capture(args.output, capture, args.overwrite_output)
                print(f"saved camera view: {args.output}")
                for line in capture["camera_contract_lines"]:
                    print(line)
                if args.apply_on_confirm:
                    apply_camera_contract(args.camera_contract, proposal)
                    print(f"updated camera contract: {args.camera_contract}")
                return 0

            now = time.perf_counter()
            if not paused and now >= next_step_at:
                if finished:
                    if now >= hold_until:
                        reset_rollout()
                        print("loop: restarted successful trajectory")
                else:
                    next_frame = current_step + 1
                    if next_frame >= frame_count:
                        finished = True
                        hold_until = now + args.end_hold_seconds
                        print(
                            f"recorded trajectory end transitions={frame_count - 1} "
                            f"success=True skill={current_skill}"
                        )
                    else:
                        apply_recorded_frame(next_frame)
                    next_step_at = now + 1.0 / args.playback_fps
                marker_position = (
                    current_target if current_target is not None else [1000] * 3
                )
                viewer.update_bounding_box(
                    marker,
                    sapien.Pose(marker_position),
                    np.full(3, args.marker_half_size),
                )
            time.sleep(0.002)
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
