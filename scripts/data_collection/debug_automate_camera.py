#!/usr/bin/env python3
"""Replay an AutoMate pickle and tune the shared front camera in Isaac Sim."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import lzma
import math
import os
import pickle
import re
import socket
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

TASK_PATTERN = re.compile(r"^automate_insertion_(\d{5,})$")
EXCLUDED_ASSEMBLY_IDS = frozenset({"00755"})
TUNING_CAMERA_PATH = "/World/RR_AutoMateFrontCameraTuning"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rb") as stream:
        trajectory = pickle.load(stream)
    if not isinstance(trajectory, dict):
        raise TypeError("trajectory must be a dictionary")
    match = TASK_PATTERN.fullmatch(str(trajectory.get("task", "")))
    if match is None:
        raise ValueError("trajectory task must match automate_insertion_<assembly_id>")
    assembly_id = match.group(1)
    if assembly_id in EXCLUDED_ASSEMBLY_IDS:
        raise ValueError(
            f"AutoMate assembly {assembly_id} is excluded from the 99-task "
            "production set and cannot be used for camera tuning"
        )
    if trajectory.get("env") != "AutoMate":
        raise ValueError("camera tuning requires an AutoMate trajectory")
    if trajectory.get("annotation_source") != "scripted":
        raise ValueError("camera tuning requires annotation_source=scripted")
    if trajectory.get("image_annotation_mode") != "none":
        raise ValueError("camera tuning requires raw images with image_annotation_mode=none")
    observations = trajectory.get("observations")
    actions = trajectory.get("actions")
    if not isinstance(observations, list) or len(observations) < 2:
        raise ValueError("trajectory must contain at least two observations")
    if not isinstance(actions, list) or len(actions) + 1 != len(observations):
        raise ValueError("trajectory must contain T actions and T+1 observations")
    trajectory["_camera_debug_assembly_id"] = assembly_id
    return trajectory


def validate_asset_root(asset_root: Path, assembly_id: str) -> dict[str, Path]:
    isaac_root = asset_root.resolve()
    if isaac_root.name != "Isaac":
        candidate = isaac_root / "Isaac"
        if candidate.is_dir():
            isaac_root = candidate
    automate_root = isaac_root / "IsaacLab" / "AutoMate"
    assembly_root = automate_root / assembly_id
    required = {
        "isaac_root": isaac_root,
        "automate_root": automate_root,
        "assembly_root": assembly_root,
        "robot_usd": automate_root / "franka_mimic.usd",
        "plug_grasps": automate_root / "plug_grasps.json",
        "disassembly_dist": automate_root / "disassembly_dist.json",
        "held_usd": assembly_root / "plug.usd",
        "fixed_usd": assembly_root / "socket.usd",
        "disassembly": assembly_root / "disassemble_traj.json",
        "ground_usd": isaac_root / "Environments" / "Grid" / "default_environment.usd",
        "table_usd": isaac_root
        / "Props"
        / "Mounts"
        / "SeattleLabTable"
        / "table_instanceable.usd",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing AutoMate camera assets: " + ", ".join(missing))
    return required


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
    quaternion_opengl_wxyz: np.ndarray,
    target_distance: float,
) -> dict[str, Any]:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_opengl_wxyz, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("camera position must contain three finite values")
    if not np.isfinite(target_distance) or target_distance <= 0:
        raise ValueError("target distance must be positive")
    rotation = quaternion_wxyz_to_matrix(quaternion)
    # USD/OpenGL camera convention: -Z is the optical forward direction.
    target = position + target_distance * (-rotation[:, 2])
    return {
        "pos": position.tolist(),
        "rot_opengl_wxyz": (quaternion / np.linalg.norm(quaternion)).tolist(),
        "eye_world": position.tolist(),
        "target_world": target.tolist(),
        "target_distance": float(target_distance),
    }


def camera_cfg_lines(proposal: Mapping[str, Any]) -> list[str]:
    def values(items) -> str:
        return "(" + ", ".join(f"{float(item):.9g}" for item in items) + ")"

    return [
        f"pos={values(proposal['pos'])},",
        f"rot={values(proposal['rot_opengl_wxyz'])},",
        'convention="opengl",',
    ]


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


def recorded_qpos(observation: Mapping[str, Any]) -> np.ndarray:
    state = observation.get("robot_state")
    if not isinstance(state, Mapping):
        raise ValueError("recorded observation is missing robot_state")
    qpos = np.concatenate(
        [
            np.asarray(state["joint_positions"], dtype=np.float32).reshape(-1),
            np.asarray(state["gripper_finger_1_pos"], dtype=np.float32).reshape(-1),
            np.asarray(state["gripper_finger_2_pos"], dtype=np.float32).reshape(-1),
        ]
    )
    if qpos.shape != (9,) or not np.isfinite(qpos).all():
        raise ValueError(f"recorded Franka qpos must be nine finite values, got {qpos}")
    return qpos


def validate_recorded_observation(observation: Mapping[str, Any], index: int) -> None:
    recorded_qpos(observation)
    parts = np.asarray(observation.get("parts_poses"), dtype=np.float64)
    if parts.shape != (14,) or not np.isfinite(parts).all():
        raise ValueError(f"observation {index}: parts_poses must be 14 finite values")
    point = np.asarray(observation.get("guidance_point"), dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"observation {index}: guidance_point must be three finite values")


def build_parser(app_launcher_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        required=True,
        help="Local official asset root; either the bundle root or its Isaac directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--playback-fps", type=float, default=5.0)
    parser.add_argument("--end-hold-seconds", type=float, default=1.0)
    parser.add_argument("--marker-radius", type=float, default=0.012)
    parser.add_argument(
        "--camera-speed",
        type=float,
        default=0.02,
        help="Omniverse camera move speed; restored to its previous value on exit.",
    )
    parser.add_argument(
        "--fine-dolly-step",
        type=float,
        default=0.01,
        help="Metres moved by comma/period along the optical axis.",
    )
    parser.add_argument("--viewer-size", type=int, default=800)
    parser.add_argument(
        "--front-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Start from this AssemblyCameraCfg.front position in robot-base coordinates.",
    )
    parser.add_argument(
        "--front-rot",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="Optional OpenGL WXYZ rotation paired with --front-pos.",
    )
    parser.add_argument(
        "--render-video",
        type=Path,
        help=(
            "Non-interactively render old-front | candidate-raw-front | "
            "candidate-standard-point from recorded states, then exit."
        ),
    )
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--runtime-smoke",
        action="store_true",
        help="Construct the Isaac environment, replay first/last frames, then exit without GUI input.",
    )
    app_launcher_cls.add_app_launcher_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = build_parser(AppLauncher)
    args, hydra_args = parser.parse_known_args(argv)
    if args.playback_fps <= 0:
        parser.error("--playback-fps must be positive")
    if args.end_hold_seconds < 0:
        parser.error("--end-hold-seconds cannot be negative")
    if args.marker_radius <= 0 or args.camera_speed <= 0 or args.fine_dolly_step <= 0:
        parser.error("marker radius and camera movement values must be positive")
    if args.viewer_size <= 0:
        parser.error("--viewer-size must be positive")
    if args.front_rot is not None:
        rotation = np.asarray(args.front_rot, dtype=np.float64)
        if not np.isfinite(rotation).all() or np.linalg.norm(rotation) < 1e-12:
            parser.error("--front-rot must contain a finite non-zero quaternion")
    if args.front_pos is not None and not np.isfinite(args.front_pos).all():
        parser.error("--front-pos must contain three finite values")

    trajectory = load_trajectory(args.trajectory)
    assembly_id = trajectory.pop("_camera_debug_assembly_id")
    assets = validate_asset_root(args.asset_root, assembly_id)
    for index, observation in enumerate(trajectory["observations"]):
        validate_recorded_observation(observation, index)
    if args.output.exists() and not args.overwrite_output:
        raise FileExistsError(f"refusing existing camera capture: {args.output}")
    if args.render_video is not None:
        video_metadata = args.render_video.with_suffix(args.render_video.suffix + ".json")
        for path in (args.render_video, video_metadata):
            if path.exists():
                raise FileExistsError(f"refusing existing candidate video artifact: {path}")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "all_pass": True,
                    "task": trajectory["task"],
                    "assembly_id": assembly_id,
                    "success": trajectory["success"],
                    "annotation_source": "scripted",
                    "image_annotation_mode": "none",
                    "frames": len(trajectory["observations"]),
                    "trajectory_sha256": sha256_file(args.trajectory),
                    "isaac_asset_root": str(assets["isaac_root"]),
                    "shared_front_camera": True,
                },
                indent=2,
            )
        )
        return 0

    args.enable_cameras = True
    sys.argv = [sys.argv[0], *hydra_args]
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb
    import gymnasium as gym
    import omni.appwindow
    import torch
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Sdf, UsdGeom

    import isaaclab.sim as sim_utils
    import isaaclab_tasks  # noqa: F401
    import isaaclab_tasks.direct.automate.assembly_env as assembly_env_module
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
    from isaaclab.utils.math import combine_frame_transforms, quat_apply
    from isaaclab_tasks.utils.hydra import hydra_task_config

    assembly_env_module.ISAAC_NUCLEUS_DIR = str(assets["isaac_root"])

    @hydra_task_config("Isaac-AutoMate-Assembly-Direct-v0", "rl_games_cfg_entry_point")
    def run(env_cfg: DirectRLEnvCfg, _agent_cfg: dict) -> int:
        env_cfg.scene.num_envs = 1
        if args.device is not None:
            env_cfg.sim.device = args.device
        env_cfg.seed = 0
        env_cfg.camera.enabled = True
        env_cfg.action_noise_model = None
        env_cfg.robot.spawn.usd_path = str(assets["robot_usd"])
        task_cfg = env_cfg.tasks[env_cfg.task_name]
        task_cfg.assembly_id = assembly_id
        task_cfg.assembly_dir = f"{assets['assembly_root']}/"
        task_cfg.fixed_asset.spawn.usd_path = str(assets["fixed_usd"])
        task_cfg.held_asset.spawn.usd_path = str(assets["held_usd"])
        task_cfg.plug_grasp_json = str(assets["plug_grasps"])
        task_cfg.disassembly_dist_json = str(assets["disassembly_dist"])
        task_cfg.disassembly_path_json = str(assets["disassembly"])
        task_cfg.if_sbc = False
        task_cfg.if_logging_eval = False
        if args.front_pos is not None:
            env_cfg.camera.front.offset.pos = tuple(float(value) for value in args.front_pos)
        if args.front_rot is not None:
            quaternion = np.asarray(args.front_rot, dtype=np.float64)
            quaternion /= np.linalg.norm(quaternion)
            env_cfg.camera.front.offset.rot = tuple(float(value) for value in quaternion)

        # AutoMate's mesh loader retrieves an asset and then opens only its
        # basename.  For an already-local absolute asset it does not copy the
        # file, so construct the environment from the assembly directory where
        # those basenames exist. Restore the caller's cwd immediately after.
        original_cwd = Path.cwd()
        try:
            os.chdir(assets["assembly_root"])
            gym_env = gym.make("Isaac-AutoMate-Assembly-Direct-v0", cfg=env_cfg)
        finally:
            os.chdir(original_cwd)
        raw_env = gym_env.unwrapped
        input_interface = None
        keyboard = None
        keyboard_sub = None
        settings = carb.settings.get_settings()
        move_speed_keys = [
            "/persistent/exts/omni.kit.manipulator.camera/moveSpeed/0",
            "/persistent/exts/omni.kit.manipulator.camera/moveSpeed/1",
            "/persistent/exts/omni.kit.manipulator.camera/moveSpeed/2",
        ]
        previous_move_speeds = [settings.get(key) for key in move_speed_keys]

        try:
            raw_env.reset(seed=0)
            observations = trajectory["observations"]
            device = raw_env.device
            env_ids = torch.tensor([0], dtype=torch.long, device=device)

            front_cfg = env_cfg.camera.front
            tuning_spawn = copy.deepcopy(front_cfg.spawn)
            tuning_spawn.horizontal_aperture *= 224.0 / 320.0
            tuning_spawn.vertical_aperture *= 224.0 / 240.0
            tuning_spawn.func(
                TUNING_CAMERA_PATH,
                tuning_spawn,
                translation=front_cfg.offset.pos,
                orientation=front_cfg.offset.rot,
            )
            stage = sim_utils.get_current_stage()
            tuning_prim = stage.GetPrimAtPath(TUNING_CAMERA_PATH)
            if not tuning_prim.IsValid() or not tuning_prim.IsA(UsdGeom.Camera):
                raise RuntimeError("failed to create the AutoMate tuning camera")
            initial_position, initial_quaternion = sim_utils.resolve_prim_pose(tuning_prim)
            initial_forward = -quaternion_wxyz_to_matrix(initial_quaternion)[:, 2]
            insertion_center = np.asarray(
                observations[0]["guidance_point"], dtype=np.float64
            )
            target_distance = float(
                np.dot(insertion_center - np.asarray(initial_position), initial_forward)
            )
            if target_distance <= 0:
                target_distance = float(
                    np.linalg.norm(insertion_center - np.asarray(initial_position))
                )

            marker = None
            if args.render_video is None:
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/RR_AutoMateGuidancePoint"
                marker_cfg.markers["sphere"].radius = args.marker_radius
                marker = VisualizationMarkers(marker_cfg)

            current_step = 0
            paused = bool(args.start_paused)
            finished = False
            hold_until = 0.0
            next_step_at = time.perf_counter()
            commands: list[str] = []

            def apply_recorded_frame(index: int) -> None:
                nonlocal current_step
                observation = observations[index]
                qpos = torch.as_tensor(
                    recorded_qpos(observation), dtype=torch.float32, device=device
                ).reshape(1, -1)
                zeros = torch.zeros_like(qpos)
                raw_env._robot.write_joint_state_to_sim(qpos, zeros, env_ids=env_ids)
                raw_env._robot.set_joint_position_target(qpos, env_ids=env_ids)

                base_pos = raw_env._robot.data.root_pos_w[0:1]
                base_quat = raw_env._robot.data.root_quat_w[0:1]
                parts = torch.as_tensor(
                    np.asarray(observation["parts_poses"]).reshape(2, 7),
                    dtype=torch.float32,
                    device=device,
                )
                part_quat_wxyz = torch.cat((parts[:, 6:7], parts[:, 3:6]), dim=1)
                base_pos_batch = base_pos.expand(2, -1)
                base_quat_batch = base_quat.expand(2, -1)
                world_pos, world_quat = combine_frame_transforms(
                    base_pos_batch, base_quat_batch, parts[:, :3], part_quat_wxyz
                )
                raw_env._held_asset.write_root_pose_to_sim(
                    torch.cat((world_pos[0:1], world_quat[0:1]), dim=1),
                    env_ids=env_ids,
                )
                raw_env._fixed_asset.write_root_pose_to_sim(
                    torch.cat((world_pos[1:2], world_quat[1:2]), dim=1),
                    env_ids=env_ids,
                )
                point_b = torch.as_tensor(
                    observation["guidance_point"], dtype=torch.float32, device=device
                ).reshape(1, 3)
                point_w = base_pos + quat_apply(base_quat, point_b)
                if marker is not None:
                    marker.visualize(point_w)
                raw_env.scene.write_data_to_sim()
                raw_env.sim.render()
                raw_env.scene.update(0.0)
                current_step = index

            def reset_replay() -> None:
                nonlocal finished, hold_until, next_step_at
                apply_recorded_frame(0)
                finished = False
                hold_until = 0.0
                next_step_at = time.perf_counter()

            def on_keyboard(event, *_unused) -> bool:
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    name = event.input.name
                    if name in {"P", "R", "C", "Q", "COMMA", "PERIOD"}:
                        commands.append(name)
                return True

            for key in move_speed_keys:
                settings.set_float(key, float(args.camera_speed))
            reset_replay()

            if args.render_video is not None:
                import imageio.v2 as imageio

                from isaaclab_tasks.direct.automate.data_collection.state_adapter import (
                    StateAdapter,
                )
                from src.data_processing.offline_image_annotations import (
                    annotate_observation_image,
                )

                adapter = StateAdapter()
                camera_info = adapter.camera_info(raw_env, 0)["front_camera"]
                args.render_video.parent.mkdir(parents=True, exist_ok=True)
                in_frame = 0
                with imageio.get_writer(
                    args.render_video,
                    fps=args.playback_fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=None,
                ) as writer:
                    for index, observation in enumerate(observations):
                        apply_recorded_frame(index)
                        candidate_front = adapter._rgb_image(
                            raw_env.get_camera_observations(0)["color_image2"],
                            "candidate color_image2",
                        )
                        point_2d = adapter._project_front_point(
                            np.asarray(observation["guidance_point"], dtype=np.float32),
                            camera_info,
                        )
                        if point_2d is not None:
                            in_frame += 1
                        candidate_observation = {
                            "color_image2": candidate_front,
                            "skill": observation.get("skill"),
                            "guidance_point_2d": {"color_image2": point_2d},
                        }
                        candidate_annotated = annotate_observation_image(
                            candidate_observation,
                            "guidance-point",
                            camera="color_image2",
                        )["color_image2"]
                        old_front = np.asarray(observation["color_image2"], dtype=np.uint8)
                        writer.append_data(
                            np.concatenate(
                                (old_front, candidate_front, candidate_annotated), axis=1
                            )
                        )

                video_metadata = args.render_video.with_suffix(
                    args.render_video.suffix + ".json"
                )
                capture = {
                    "schema": "rr-automate-front-camera-candidate-video-v1",
                    "saved_at": datetime.now().astimezone().isoformat(),
                    "host": socket.gethostname(),
                    "task": trajectory["task"],
                    "trajectory": str(args.trajectory.resolve()),
                    "trajectory_sha256": sha256_file(args.trajectory),
                    "annotation_source": "scripted",
                    "image_annotation_mode": "none",
                    "candidate_front_camera": {
                        "pos": list(front_cfg.offset.pos),
                        "rot_opengl_wxyz": list(front_cfg.offset.rot),
                    },
                    "frames": len(observations),
                    "guidance_points_in_frame": in_frame,
                    "video": str(args.render_video.resolve()),
                    "video_sha256": sha256_file(args.render_video),
                    "fps": args.playback_fps,
                    "panels": [
                        "recorded_old_front_raw",
                        "candidate_front_raw_224_center_crop",
                        "candidate_front_standard_guidance_point",
                    ],
                }
                write_capture(video_metadata, capture, overwrite=False)
                print(json.dumps(capture, indent=2, sort_keys=True))
                return 0

            if args.runtime_smoke:
                apply_recorded_frame(len(observations) - 1)
                position, quaternion = sim_utils.resolve_prim_pose(tuning_prim)
                print(
                    json.dumps(
                        {
                            "runtime_smoke": True,
                            "task": trajectory["task"],
                            "frames_replayed": [0, len(observations) - 1],
                            "tuning_camera_position": list(position),
                            "tuning_camera_quaternion_opengl_wxyz": list(quaternion),
                            "shared_front_camera": True,
                        },
                        sort_keys=True,
                    )
                )
                return 0

            input_interface = carb.input.acquire_input_interface()
            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("Isaac Sim has no app window; do not use --headless for GUI tuning")
            keyboard = app_window.get_keyboard()
            keyboard_sub = input_interface.subscribe_to_keyboard_events(
                keyboard, on_keyboard
            )

            viewport = None
            for _ in range(120):
                simulation_app.update()
                viewport = get_active_viewport()
                if viewport is not None:
                    break
            if viewport is None:
                raise RuntimeError("Isaac Sim has no active viewport")
            viewport.camera_path = Sdf.Path(TUNING_CAMERA_PATH)
            viewport.resolution = (args.viewer_size, args.viewer_size)

            print(
                "Controls: mouse orbit/pan/dolly in the active viewport; "
                "comma/period fine-dolly; P pause/resume; R restart; "
                "C confirm+save; Q quit without saving."
            )
            print(
                f"task={trajectory['task']} source=scripted replay=recorded_state "
                f"frames={len(observations)} success={trajectory['success']} "
                f"camera_speed={args.camera_speed} fine_dolly={args.fine_dolly_step}m "
                "shared_front_camera=True"
            )

            while simulation_app.is_running():
                while commands:
                    command = commands.pop(0)
                    if command == "Q":
                        print("Quit without saving a camera view.")
                        return 1
                    if command == "P":
                        paused = not paused
                        print(f"paused={paused} step={current_step}")
                    elif command == "R":
                        reset_replay()
                        print("restarted trajectory")
                    elif command in {"COMMA", "PERIOD"}:
                        position, quaternion = sim_utils.resolve_prim_pose(tuning_prim)
                        forward = -quaternion_wxyz_to_matrix(quaternion)[:, 2]
                        direction = -1.0 if command == "COMMA" else 1.0
                        new_position = np.asarray(position) + direction * args.fine_dolly_step * forward
                        sim_utils.standardize_xform_ops(
                            tuning_prim,
                            translation=tuple(new_position),
                            orientation=tuple(quaternion),
                        )
                        print(f"fine_dolly position={new_position.tolist()}")
                    elif command == "C":
                        position, quaternion = sim_utils.resolve_prim_pose(tuning_prim)
                        camera = UsdGeom.Camera(tuning_prim)
                        actual_focal = float(camera.GetFocalLengthAttr().Get())
                        actual_h_aperture = float(camera.GetHorizontalApertureAttr().Get())
                        actual_v_aperture = float(camera.GetVerticalApertureAttr().Get())
                        expected = (
                            float(tuning_spawn.focal_length),
                            float(tuning_spawn.horizontal_aperture),
                            float(tuning_spawn.vertical_aperture),
                        )
                        actual = (actual_focal, actual_h_aperture, actual_v_aperture)
                        if not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
                            raise RuntimeError(
                                f"camera intrinsics changed during tuning: expected={expected}, actual={actual}"
                            )
                        proposal = camera_proposal(
                            np.asarray(position), np.asarray(quaternion), target_distance
                        )
                        capture = {
                            "schema": "rr-automate-shared-front-camera-view-v1",
                            "saved_at": datetime.now().astimezone().isoformat(),
                            "host": socket.gethostname(),
                            "task_used_for_replay": trajectory["task"],
                            "assembly_id_used_for_replay": assembly_id,
                            "applies_to_all_automate_tasks": True,
                            "annotation_source": "scripted",
                            "replay_mode": "recorded_state",
                            "trajectory_success": trajectory["success"],
                            "trajectory": str(args.trajectory.resolve()),
                            "trajectory_sha256": sha256_file(args.trajectory),
                            "rollout_step_at_capture": current_step,
                            "guidance_target_base_at_capture": np.asarray(
                                observations[current_step]["guidance_point"]
                            ).tolist(),
                            "viewer_resolution": [args.viewer_size, args.viewer_size],
                            "effective_stored_crop_aperture": {
                                "horizontal": actual_h_aperture,
                                "vertical": actual_v_aperture,
                                "focal_length": actual_focal,
                            },
                            "previous_front_camera": {
                                "pos": list(front_cfg.offset.pos),
                                "rot_opengl_wxyz": list(front_cfg.offset.rot),
                            },
                            "proposed_front_camera": proposal,
                            "assembly_env_cfg_offset_lines": camera_cfg_lines(proposal),
                        }
                        write_capture(args.output, capture, args.overwrite_output)
                        print(f"saved shared AutoMate front camera proposal: {args.output}")
                        for line in capture["assembly_env_cfg_offset_lines"]:
                            print(line)
                        return 0

                now = time.perf_counter()
                if not paused and now >= next_step_at:
                    if finished:
                        if now >= hold_until:
                            reset_replay()
                            print("loop: restarted recorded trajectory")
                    else:
                        next_frame = current_step + 1
                        if next_frame >= len(observations):
                            finished = True
                            hold_until = now + args.end_hold_seconds
                            print(
                                f"recorded trajectory end transitions={len(observations) - 1} "
                                f"success={trajectory['success']}"
                            )
                        else:
                            apply_recorded_frame(next_frame)
                        next_step_at = now + 1.0 / args.playback_fps
                simulation_app.update()
            return 1
        finally:
            if input_interface is not None and keyboard_sub is not None:
                input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
            for key, previous in zip(move_speed_keys, previous_move_speeds):
                if previous is None:
                    settings.destroy_item(key)
                else:
                    settings.set_float(key, float(previous))
            gym_env.close()

    try:
        return int(run())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
