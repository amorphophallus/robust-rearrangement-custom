"""Strict validator for the shared cross-simulator raw-pickle contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.data_collection.pickle_contract import CANONICAL_IMAGE_SIZE


SOURCE_ENVS = ("FurnitureBench", "ManiSkill", "AutoMate")
TRAJECTORY_KEYS = (
    "env",
    "task",
    "success",
    "action_type",
    "observations",
    "actions",
    "rewards",
    "camera_info",
)
OBSERVATION_KEYS = (
    "robot_state",
    "color_image1",
    "color_image2",
    "depth_image1",
    "depth_image2",
    "parts_poses",
    "point_cloud",
    "skill",
    "guidance_point",
    "guidance_point_clean",
    "guidance_pose",
    "guidance_pose_clean",
    "guidance_gripper_width",
    "guidance_point_2d",
    "grasp_annotation_2d",
)
ROBOT_STATE_SHAPES = {
    "ee_pos": (3,),
    "ee_quat": (4,),
    "ee_pos_sim": (3,),
    "ee_quat_sim": (4,),
    "ee_pos_vel": (3,),
    "ee_ori_vel": (3,),
    "gripper_width": (1,),
    "joint_positions": (7,),
    "joint_velocities": (7,),
    "joint_torques": (9,),
    "gripper_finger_1_pos": (1,),
    "gripper_finger_2_pos": (1,),
}
CAMERA_CALIBRATION_KEYS = (
    "image_size",
    "intrinsics",
    "camera_to_sim_local",
    "sim_local_to_camera",
)


class PickleContractError(ValueError):
    """A raw trajectory does not satisfy the canonical shared contract."""


def _fail(path: str, message: str) -> None:
    raise PickleContractError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        _fail(path, f"expected a mapping, got {type(value).__name__}")
    return value


def _required_keys(value: Mapping, required: Sequence[str], path: str) -> None:
    missing = set(required) - set(value)
    if missing:
        _fail(path, f"missing required keys {sorted(missing)}")


def _array(
    value: Any,
    shape: tuple[int, ...],
    dtype: np.dtype,
    path: str,
    *,
    finite: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _fail(path, f"expected numpy.ndarray, got {type(value).__name__}")
    if value.shape != shape:
        _fail(path, f"expected shape {shape}, got {value.shape}")
    if value.dtype != np.dtype(dtype):
        _fail(path, f"expected dtype {np.dtype(dtype)}, got {value.dtype}")
    if finite and not np.isfinite(value).all():
        _fail(path, "contains NaN or infinite values")
    return value


def _unit_quaternion(value: np.ndarray, path: str) -> None:
    norms = np.linalg.norm(value, axis=-1)
    if not np.allclose(norms, 1.0, atol=1.0e-4, rtol=0.0):
        _fail(path, f"expected unit xyzw quaternion(s), norms={norms}")


def _validate_robot_state(value: Any, path: str) -> None:
    state = _mapping(value, path)
    _required_keys(state, tuple(ROBOT_STATE_SHAPES), path)
    arrays = {
        key: _array(state[key], shape, np.float32, f"{path}.{key}")
        for key, shape in ROBOT_STATE_SHAPES.items()
    }
    _unit_quaternion(arrays["ee_quat"], f"{path}.ee_quat")
    _unit_quaternion(arrays["ee_quat_sim"], f"{path}.ee_quat_sim")
    if not np.array_equal(arrays["ee_pos"], arrays["ee_pos_sim"]):
        _fail(path, "ee_pos_sim must be an exact base-frame alias of ee_pos")
    if not np.array_equal(arrays["ee_quat"], arrays["ee_quat_sim"]):
        _fail(path, "ee_quat_sim must be an exact base-frame alias of ee_quat")
    expected_width = (
        arrays["gripper_finger_1_pos"] + arrays["gripper_finger_2_pos"]
    )
    if not np.allclose(
        arrays["gripper_width"], expected_width, atol=1.0e-6, rtol=0.0
    ):
        _fail(path, "gripper_width must equal both finger positions summed")


def _optional_float32_array(
    value: Any, shape: tuple[int, ...], path: str
) -> Optional[np.ndarray]:
    if value is None:
        return None
    return _array(value, shape, np.float32, path)


def _validate_2d_annotations(observation: Mapping, path: str) -> None:
    points = observation["guidance_point_2d"]
    if points is not None:
        points = _mapping(points, f"{path}.guidance_point_2d")
        for camera_key, value in points.items():
            if value is None:
                continue
            point = _array(
                value,
                (2,),
                np.float32,
                f"{path}.guidance_point_2d.{camera_key}",
            )
            if np.any(point < 0) or np.any(point >= CANONICAL_IMAGE_SIZE):
                _fail(
                    f"{path}.guidance_point_2d.{camera_key}",
                    "visible point is outside the stored image",
                )

    grasps = observation["grasp_annotation_2d"]
    if grasps is None:
        return
    grasps = _mapping(grasps, f"{path}.grasp_annotation_2d")
    for camera_key, value in grasps.items():
        if value is None:
            continue
        grasp = _mapping(value, f"{path}.grasp_annotation_2d.{camera_key}")
        _required_keys(
            grasp,
            ("style", "center", "corners"),
            f"{path}.grasp_annotation_2d.{camera_key}",
        )
        if grasp["style"] != "grasp_rect":
            _fail(
                f"{path}.grasp_annotation_2d.{camera_key}.style",
                "expected 'grasp_rect'",
            )
        _array(
            grasp["center"],
            (2,),
            np.float32,
            f"{path}.grasp_annotation_2d.{camera_key}.center",
        )
        corners = np.asarray(grasp["corners"])
        if (
            not isinstance(grasp["corners"], np.ndarray)
            or corners.dtype != np.float32
            or corners.ndim != 2
            or corners.shape[-1] != 2
            or not np.isfinite(corners).all()
        ):
            _fail(
                f"{path}.grasp_annotation_2d.{camera_key}.corners",
                "expected a finite float32 array shaped (N,2)",
            )


def _validate_observation(value: Any, index: int) -> dict[str, tuple[float, float]]:
    path = f"observations[{index}]"
    observation = _mapping(value, path)
    _required_keys(observation, OBSERVATION_KEYS, path)
    _validate_robot_state(observation["robot_state"], f"{path}.robot_state")

    depth_ranges = {}
    for camera_idx in (1, 2):
        color_key = f"color_image{camera_idx}"
        depth_key = f"depth_image{camera_idx}"
        _array(
            observation[color_key],
            (CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE, 3),
            np.uint8,
            f"{path}.{color_key}",
            finite=False,
        )
        depth = _array(
            observation[depth_key],
            (CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE),
            np.float32,
            f"{path}.{depth_key}",
        )
        if np.any(depth < 0):
            _fail(f"{path}.{depth_key}", "depth must use positive metres")
        depth_ranges[depth_key] = (float(np.min(depth)), float(np.max(depth)))

    parts = np.asarray(observation["parts_poses"])
    if (
        not isinstance(observation["parts_poses"], np.ndarray)
        or parts.dtype != np.float32
        or parts.ndim != 1
        or parts.size == 0
        or parts.size % 7 != 0
        or not np.isfinite(parts).all()
    ):
        _fail(
            f"{path}.parts_poses",
            "expected a nonempty finite float32 vector of flattened 7D poses",
        )
    _unit_quaternion(parts.reshape(-1, 7)[:, 3:7], f"{path}.parts_poses")

    skill = observation["skill"]
    if skill is not None and not isinstance(skill, str):
        _fail(f"{path}.skill", "expected string or None")
    _optional_float32_array(
        observation["guidance_point"], (3,), f"{path}.guidance_point"
    )
    _optional_float32_array(
        observation["guidance_point_clean"],
        (3,),
        f"{path}.guidance_point_clean",
    )
    for key in ("guidance_pose", "guidance_pose_clean"):
        pose = _optional_float32_array(observation[key], (4, 4), f"{path}.{key}")
        if pose is not None and not np.allclose(
            pose[3], [0, 0, 0, 1], atol=1.0e-5, rtol=0.0
        ):
            _fail(f"{path}.{key}", "last row must be homogeneous [0,0,0,1]")
    guidance_width = observation["guidance_gripper_width"]
    if guidance_width is not None:
        array = np.asarray(guidance_width)
        if array.size != 1 or not np.isfinite(array).all():
            _fail(f"{path}.guidance_gripper_width", "expected one finite scalar")
    _validate_2d_annotations(observation, path)
    return depth_ranges


def _validate_camera_info(
    value: Any,
    *,
    front_focal_reference: Optional[float],
    front_focal_rtol: float,
    front_translation_reference: Optional[Sequence[float]],
    front_translation_atol: float,
    front_forward_reference: Optional[Sequence[float]],
    front_forward_cosine_min: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    camera_info = _mapping(value, "camera_info")
    if "front_camera" not in camera_info or camera_info["front_camera"] is None:
        _fail("camera_info", "missing non-null front_camera calibration")
    front = _mapping(camera_info["front_camera"], "camera_info.front_camera")
    _required_keys(front, CAMERA_CALIBRATION_KEYS, "camera_info.front_camera")
    image_size = _array(
        front["image_size"], (2,), np.int32, "camera_info.front_camera.image_size"
    )
    if not np.array_equal(
        image_size, [CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE]
    ):
        _fail(
            "camera_info.front_camera.image_size",
            f"expected [{CANONICAL_IMAGE_SIZE},{CANONICAL_IMAGE_SIZE}]",
        )
    intrinsics = _array(
        front["intrinsics"], (3, 3), np.float32, "camera_info.front_camera.intrinsics"
    )
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        _fail("camera_info.front_camera.intrinsics", "focal lengths must be positive")
    if not (0 <= cx < CANONICAL_IMAGE_SIZE and 0 <= cy < CANONICAL_IMAGE_SIZE):
        _fail("camera_info.front_camera.intrinsics", "principal point is outside image")
    if front_focal_reference is not None and not np.isclose(
        fx, front_focal_reference, rtol=front_focal_rtol, atol=0.0
    ):
        _fail(
            "camera_info.front_camera.intrinsics",
            f"fx={fx:.6g} differs from reference {front_focal_reference:.6g} "
            f"by more than rtol={front_focal_rtol}",
        )

    transforms = []
    for key in ("camera_to_sim_local", "sim_local_to_camera"):
        transform = _array(front[key], (4, 4), np.float32, f"camera_info.front_camera.{key}")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1.0e-5, rtol=0.0):
            _fail(f"camera_info.front_camera.{key}", "invalid homogeneous last row")
        rotation = transform[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=2.0e-4, rtol=0.0
        ) or not np.isclose(abs(np.linalg.det(rotation)), 1.0, atol=2.0e-4):
            _fail(
                f"camera_info.front_camera.{key}",
                "camera basis must be orthonormal (left- or right-handed)",
            )
        transforms.append(transform)
    if not np.allclose(transforms[0] @ transforms[1], np.eye(4), atol=2.0e-4, rtol=0.0):
        _fail("camera_info.front_camera", "camera transforms are not mutual inverses")
    translation = transforms[0][:3, 3]
    forward = transforms[0][:3, 2]
    if front_translation_reference is not None:
        reference = np.asarray(front_translation_reference, dtype=np.float64)
        if reference.shape != (3,):
            raise ValueError("front_translation_reference must contain three values")
        if not np.allclose(
            translation, reference, atol=front_translation_atol, rtol=0.0
        ):
            _fail(
                "camera_info.front_camera.camera_to_sim_local",
                f"translation={translation.tolist()} differs from robot-base "
                f"reference {reference.tolist()} by more than atol="
                f"{front_translation_atol}",
            )
    if front_forward_reference is not None:
        reference = np.asarray(front_forward_reference, dtype=np.float64)
        if reference.shape != (3,) or np.linalg.norm(reference) == 0:
            raise ValueError("front_forward_reference must contain three nonzero values")
        cosine = float(
            np.dot(forward, reference)
            / (np.linalg.norm(forward) * np.linalg.norm(reference))
        )
        if cosine < front_forward_cosine_min:
            _fail(
                "camera_info.front_camera.camera_to_sim_local",
                f"forward-axis cosine={cosine:.6g} is below "
                f"{front_forward_cosine_min}",
            )
    return fx, fy, translation, forward


def _validate_front_guidance_projection(
    observation: Mapping,
    calibration: Mapping,
    *,
    index: int,
    annotation_source: Any,
) -> None:
    """Check calibrated 3D-to-2D geometry when a comparable label exists.

    VLM front points are predictions and are intentionally not required to
    match the oracle 3D target.  Such trajectories carry a separate
    ``oracle_guidance_point_2d`` field, which is the correct comparison.
    Scripted trajectories compare their regular 2D point directly.
    """
    point = observation.get("guidance_point_clean")
    if point is None:
        point = observation.get("guidance_point")
    if point is None:
        return

    point_field = "oracle_guidance_point_2d"
    point_mapping = observation.get(point_field)
    if not isinstance(point_mapping, Mapping):
        if annotation_source != "scripted":
            return
        point_field = "guidance_point_2d"
        point_mapping = observation.get(point_field)
    if not isinstance(point_mapping, Mapping):
        return
    pixel = point_mapping.get("color_image2")
    if pixel is None:
        return

    point_h = np.concatenate(
        (np.asarray(point, dtype=np.float32), np.ones(1, dtype=np.float32))
    )
    point_camera = calibration["sim_local_to_camera"] @ point_h
    path = f"observations[{index}]"
    if not np.isfinite(point_camera).all() or point_camera[2] <= 1.0e-8:
        _fail(
            f"{path}.guidance_point_clean",
            "labelled point lies behind the calibrated front camera",
        )
    point_cv = point_camera[:3].copy()
    point_cv[1] *= -1.0
    pixel_h = calibration["intrinsics"] @ point_cv
    expected = pixel_h[:2] / pixel_h[2]
    actual = np.asarray(pixel, dtype=np.float32)
    if actual.shape != (2,) or not np.allclose(
        actual, expected, atol=1.0, rtol=0.0
    ):
        _fail(
            f"{path}.{point_field}.color_image2",
            "does not match the calibrated base-frame 3D guidance point "
            f"(stored={actual.tolist()}, projected={expected.tolist()})",
        )


def observation_indices(length: int, sample_observations: Optional[int]) -> list[int]:
    if sample_observations is None or sample_observations >= length:
        return list(range(length))
    if sample_observations <= 0:
        raise ValueError("sample_observations must be positive or None")
    return sorted(
        set(np.linspace(0, length - 1, sample_observations, dtype=np.int64).tolist())
    )


def validate_pickle_trajectory(
    data: Any,
    *,
    source: str = "<pickle>",
    sample_observations: Optional[int] = None,
    front_focal_reference: Optional[float] = None,
    front_focal_rtol: float = 0.15,
    front_translation_reference: Optional[Sequence[float]] = None,
    front_translation_atol: float = 0.10,
    front_forward_reference: Optional[Sequence[float]] = None,
    front_forward_cosine_min: float = 0.98,
    expected_image_annotation_mode: Optional[str] = None,
) -> dict[str, Any]:
    """Validate one loaded trajectory and return a compact audit summary."""
    trajectory = _mapping(data, source)
    _required_keys(trajectory, TRAJECTORY_KEYS, source)

    env_name = trajectory["env"]
    if env_name not in SOURCE_ENVS:
        _fail(f"{source}.env", f"expected one of {SOURCE_ENVS}, got {env_name!r}")
    if not isinstance(trajectory["task"], str) or not trajectory["task"]:
        _fail(f"{source}.task", "expected a nonempty string")
    if not isinstance(trajectory["success"], (bool, np.bool_)):
        _fail(f"{source}.success", "expected bool")
    image_annotation_mode = trajectory.get("image_annotation_mode")
    if (
        expected_image_annotation_mode is not None
        and image_annotation_mode != expected_image_annotation_mode
    ):
        _fail(
            f"{source}.image_annotation_mode",
            f"expected {expected_image_annotation_mode!r}, got {image_annotation_mode!r}",
        )
    if trajectory["action_type"] != "delta":
        _fail(f"{source}.action_type", "expected 'delta'")

    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] == 0:
        _fail(f"{source}.actions", f"expected nonempty shape (T,8), got {actions.shape}")
    if not np.isfinite(actions).all():
        _fail(f"{source}.actions", "contains NaN or infinite values")
    _unit_quaternion(actions[:, 3:7], f"{source}.actions[:,3:7]")
    if np.any(actions[:, 7] < -1) or np.any(actions[:, 7] > 1):
        _fail(f"{source}.actions[:,7]", "absolute gripper commands must be in [-1,1]")

    observations = trajectory["observations"]
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        _fail(f"{source}.observations", "expected a sequence")
    if len(observations) != actions.shape[0] + 1:
        _fail(
            f"{source}.observations",
            f"expected T+1={actions.shape[0] + 1}, got {len(observations)}",
        )
    rewards = np.asarray(trajectory["rewards"], dtype=np.float32).reshape(-1)
    if rewards.shape != (actions.shape[0],) or not np.isfinite(rewards).all():
        _fail(
            f"{source}.rewards",
            f"expected T finite rewards with shape ({actions.shape[0]},), got {rewards.shape}",
        )

    indices = observation_indices(len(observations), sample_observations)
    depth_ranges = {}
    for index in indices:
        ranges = _validate_observation(observations[index], index)
        for key, (minimum, maximum) in ranges.items():
            old_min, old_max = depth_ranges.get(key, (minimum, maximum))
            depth_ranges[key] = (min(old_min, minimum), max(old_max, maximum))

    fx, fy, front_translation, front_forward = _validate_camera_info(
        trajectory["camera_info"],
        front_focal_reference=front_focal_reference,
        front_focal_rtol=front_focal_rtol,
        front_translation_reference=front_translation_reference,
        front_translation_atol=front_translation_atol,
        front_forward_reference=front_forward_reference,
        front_forward_cosine_min=front_forward_cosine_min,
    )
    front_calibration = trajectory["camera_info"]["front_camera"]
    for index in indices:
        _validate_front_guidance_projection(
            observations[index],
            front_calibration,
            index=index,
            annotation_source=trajectory.get("annotation_source"),
        )
    return {
        "source": source,
        "env": env_name,
        "task": trajectory["task"],
        "success": bool(trajectory["success"]),
        "image_annotation_mode": image_annotation_mode,
        "transitions": int(actions.shape[0]),
        "observations_checked": len(indices),
        "front_fx": fx,
        "front_fy": fy,
        "front_translation": front_translation.tolist(),
        "front_forward": front_forward.tolist(),
        "gripper_min": float(np.min(actions[:, 7])),
        "gripper_max": float(np.max(actions[:, 7])),
        "max_delta_position": float(np.max(np.abs(actions[:, :3]))),
        "depth_ranges": depth_ranges,
    }


def discover_pickle_paths(inputs: Sequence[Path]) -> list[Path]:
    paths = []
    for input_path in inputs:
        input_path = input_path.expanduser()
        if input_path.is_file():
            paths.append(input_path)
            continue
        if input_path.is_dir():
            paths.extend(
                path
                for path in input_path.rglob("*")
                if path.is_file()
                and (path.name.endswith(".pkl") or path.name.endswith(".pkl.xz"))
            )
            continue
        raise FileNotFoundError(input_path)
    return sorted(set(path.resolve() for path in paths))
