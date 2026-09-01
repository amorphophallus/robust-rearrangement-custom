"""Canonical cross-simulator raw-pickle contract helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


CANONICAL_IMAGE_SIZE = 224


def normalize_depth_meters(array: np.ndarray) -> np.ndarray:
    """Return finite depth in positive metres as contiguous float32 values.

    FurnitureBench's simulator historically emitted negative camera-axis
    depth, while ManiSkill and AutoMate emit positive metric depth.  The raw
    cross-simulator contract uses the latter convention.
    """
    depth = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(depth)):
        raise ValueError("Depth images contain NaN or infinite values.")
    return np.ascontiguousarray(np.abs(depth), dtype=np.float32)


def center_crop_offsets(height: int, width: int, image_size: int) -> tuple[int, int]:
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError(f"image_size must be a positive integer, got {image_size!r}.")
    if height < image_size or width < image_size:
        raise ValueError(
            f"Image is too small for a {image_size}x{image_size} center crop: "
            f"received {height}x{width}."
        )
    return (height - image_size) // 2, (width - image_size) // 2


def center_crop_observation_images(
    color_image1: np.ndarray,
    color_image2: np.ndarray,
    depth_image1: np.ndarray,
    depth_image2: np.ndarray,
    image_size: int,
):
    """Center-crop both RGB-D streams to one square storage size.

    The same crop is applied to RGB and depth within each camera stream. This
    provides a deterministic, simulator-independent storage contract without
    interpolating metric depth values.
    """
    arrays = {
        "color_image1": np.asarray(color_image1),
        "color_image2": np.asarray(color_image2),
        "depth_image1": np.asarray(depth_image1),
        "depth_image2": np.asarray(depth_image2),
    }
    for camera_idx in (1, 2):
        color_key = f"color_image{camera_idx}"
        depth_key = f"depth_image{camera_idx}"
        color = arrays[color_key]
        depth = arrays[depth_key]
        if color.ndim != 4 or color.shape[-1] != 3:
            raise ValueError(
                f"{color_key} must have shape (T, H, W, 3), got {color.shape}."
            )
        if depth.ndim != 3:
            raise ValueError(
                f"{depth_key} must have shape (T, H, W), got {depth.shape}."
            )
        if color.shape[:3] != depth.shape:
            raise ValueError(
                f"{color_key} and {depth_key} must have matching T/H/W dimensions, "
                f"got {color.shape[:3]} and {depth.shape}."
            )

        height, width = color.shape[1:3]
        top, left = center_crop_offsets(height, width, image_size)
        arrays[color_key] = np.ascontiguousarray(
            color[:, top : top + image_size, left : left + image_size, :]
        )
        arrays[depth_key] = np.ascontiguousarray(
            depth[:, top : top + image_size, left : left + image_size]
        )

    return (
        arrays["color_image1"],
        arrays["color_image2"],
        arrays["depth_image1"],
        arrays["depth_image2"],
    )


def _cropped_points(points, source_shape, image_size: int):
    points = np.asarray(points, dtype=np.float32)
    if points.shape[-1] != 2:
        raise ValueError(f"2D annotations must end in dimension 2, got {points.shape}.")
    top, left = center_crop_offsets(*source_shape, image_size)
    cropped = points.copy()
    cropped[..., 0] -= left
    cropped[..., 1] -= top
    visible = (
        (cropped[..., 0] >= 0)
        & (cropped[..., 0] < image_size)
        & (cropped[..., 1] >= 0)
        & (cropped[..., 1] < image_size)
    )
    return cropped if np.all(visible) else None


def center_crop_point_mapping(point_mapping, source_shapes, image_size: int):
    if point_mapping is None:
        return None
    if not isinstance(point_mapping, dict):
        raise TypeError("2D point annotations must be camera-keyed dictionaries.")
    cropped = dict(point_mapping)
    for image_key, point in point_mapping.items():
        if point is None or image_key not in source_shapes:
            continue
        cropped[image_key] = _cropped_points(
            point, source_shapes[image_key], image_size
        )
    return cropped


def center_crop_grasp_mapping(grasp_mapping, source_shapes, image_size: int):
    if grasp_mapping is None:
        return None
    if not isinstance(grasp_mapping, dict):
        raise TypeError("2D grasp annotations must be camera-keyed dictionaries.")
    cropped = dict(grasp_mapping)
    for image_key, annotation in grasp_mapping.items():
        if annotation is None or image_key not in source_shapes:
            continue
        annotation = dict(annotation)
        center = _cropped_points(
            annotation["center"], source_shapes[image_key], image_size
        )
        corners = _cropped_points(
            annotation["corners"], source_shapes[image_key], image_size
        )
        if center is None or corners is None:
            cropped[image_key] = None
        else:
            annotation["center"] = center
            annotation["corners"] = corners
            cropped[image_key] = annotation
    return cropped


def center_crop_camera_calibration(calibration, image_size: int):
    if calibration is None:
        return None
    calibration = dict(calibration)
    source_width, source_height = np.asarray(calibration["image_size"]).tolist()
    top, left = center_crop_offsets(source_height, source_width, image_size)
    intrinsics = np.asarray(calibration["intrinsics"], dtype=np.float32).copy()
    intrinsics[0, 2] -= left
    intrinsics[1, 2] -= top
    calibration["image_size"] = np.asarray(
        [image_size, image_size], dtype=np.int32
    )
    calibration["intrinsics"] = intrinsics
    return calibration


def _pose_matrix(position, quaternion_xyzw, name: str) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64).reshape(-1)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError(
            f"{name} must contain position (3,) and xyzw quaternion (4,), "
            f"got {position.shape} and {quaternion.shape}."
        )
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError(f"{name} contains non-finite values.")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return matrix


def legacy_sim_local_to_robot_base_matrix(robot_state) -> np.ndarray:
    """Recover the legacy FB sim-local-to-robot-base transform.

    Legacy FurnitureBench state records contain the same end-effector pose in
    robot-base (``ee_pos``/``ee_quat``) and simulator-local
    (``ee_pos_sim``/``ee_quat_sim``) coordinates.  One paired pose is enough
    to recover the fixed transform between those frames.
    """
    if not isinstance(robot_state, dict):
        raise TypeError("robot_state must be a mapping for frame conversion.")
    required = ("ee_pos", "ee_quat", "ee_pos_sim", "ee_quat_sim")
    missing = set(required) - set(robot_state)
    if missing:
        raise ValueError(
            "Cannot recover the robot-base transform; missing state keys "
            f"{sorted(missing)}."
        )

    ee_to_base = _pose_matrix(
        robot_state["ee_pos"], robot_state["ee_quat"], "base-frame EE pose"
    )
    ee_to_sim_local = _pose_matrix(
        robot_state["ee_pos_sim"],
        robot_state["ee_quat_sim"],
        "sim-local EE pose",
    )
    base_to_sim_local = ee_to_sim_local @ np.linalg.inv(ee_to_base)
    return np.linalg.inv(base_to_sim_local)


def point_to_robot_base(point, sim_local_to_base: np.ndarray):
    """Transform one optional legacy sim-local 3D point to robot base."""
    if point is None:
        return None
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"3D point must be finite with shape (3,), got {point.shape}.")
    point_h = np.concatenate((point, np.ones(1, dtype=np.float64)))
    return np.ascontiguousarray((sim_local_to_base @ point_h)[:3], dtype=np.float32)


def pose_to_robot_base(pose, sim_local_to_base: np.ndarray):
    """Transform one optional legacy sim-local homogeneous pose to robot base."""
    if pose is None:
        return None
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Pose must be finite with shape (4,4), got {pose.shape}.")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1.0e-6, rtol=0.0):
        raise ValueError("Pose must have homogeneous last row [0,0,0,1].")
    return np.ascontiguousarray(sim_local_to_base @ pose, dtype=np.float32)


def flattened_poses_to_robot_base(poses, sim_local_to_base: np.ndarray):
    """Transform flattened ``[xyz, quat_xyzw]`` poses to robot base."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 1 or poses.size == 0 or poses.size % 7 != 0:
        raise ValueError(
            "Flattened poses must be a nonempty vector with length divisible by 7, "
            f"got {poses.shape}."
        )
    if not np.isfinite(poses).all():
        raise ValueError("Flattened poses contain non-finite values.")

    transformed = []
    for index, pose_vector in enumerate(poses.reshape(-1, 7)):
        pose = _pose_matrix(
            pose_vector[:3], pose_vector[3:7], f"parts_poses[{index}]"
        )
        pose_base = sim_local_to_base @ pose
        quat_base = R.from_matrix(pose_base[:3, :3]).as_quat()
        transformed.append(np.concatenate((pose_base[:3, 3], quat_base)))
    return np.ascontiguousarray(np.concatenate(transformed), dtype=np.float32)


def camera_calibration_to_robot_base(calibration, robot_state):
    """Express a legacy FB sim-local camera transform in robot base.

    Legacy FB observations contain the same EE pose in both robot-base
    (``ee_pos``/``ee_quat``) and simulator-local
    (``ee_pos_sim``/``ee_quat_sim``) coordinates. Their relation recovers the
    otherwise implicit robot-base transform without hard-coding a scene offset.
    """
    if calibration is None:
        return None
    sim_local_to_base = legacy_sim_local_to_robot_base_matrix(robot_state)

    converted = dict(calibration)
    camera_to_sim_local = np.asarray(
        calibration["camera_to_sim_local"], dtype=np.float64
    )
    if camera_to_sim_local.shape != (4, 4) or not np.isfinite(
        camera_to_sim_local
    ).all():
        raise ValueError("camera_to_sim_local must be a finite 4x4 transform.")
    camera_to_base = sim_local_to_base @ camera_to_sim_local
    converted["camera_to_sim_local"] = camera_to_base.astype(np.float32)
    converted["sim_local_to_camera"] = np.linalg.inv(camera_to_base).astype(
        np.float32
    )
    return converted


def robot_state_with_base_frame_aliases(robot_state):
    """Return a copied pickle state whose legacy EE aliases use robot base.

    FurnitureBench historically exposed ``ee_pos_sim``/``ee_quat_sim`` in the
    simulator-local frame even though the canonical ``ee_pos``/``ee_quat``
    fields and actions use the robot-base frame. Cross-simulator pickles keep
    the legacy keys for schema compatibility, but all EE poses must share the
    canonical frame.
    """
    if not isinstance(robot_state, dict):
        return robot_state

    state = dict(robot_state)
    if "ee_pos_sim" in state and "ee_pos" in state:
        state["ee_pos_sim"] = np.asarray(state["ee_pos"]).copy()
    if "ee_quat_sim" in state and "ee_quat" in state:
        state["ee_quat_sim"] = np.asarray(state["ee_quat"]).copy()
    return state


def validate_and_align_pickle_timeseries(observations, actions, rewards):
    """Validate the canonical raw-pickle transition contract.

    Legacy FurnitureBench rollout evaluation can accumulate reward entries at
    a finer cadence than observations/actions. Keep the historical leading
    transition rewards, but never write a raw pickle with mismatched lengths.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(
            "Canonical pickle actions must have shape (T, 8), got "
            f"{actions.shape}."
        )
    if actions.shape[0] == 0:
        raise ValueError("Canonical pickle trajectories must contain an action.")
    if not np.isfinite(actions).all():
        raise ValueError("Canonical pickle actions contain non-finite values.")
    quat_norms = np.linalg.norm(actions[:, 3:7], axis=-1)
    if not np.allclose(quat_norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError(
            "Canonical pickle action quaternions must be unit xyzw quaternions."
        )
    if np.any(actions[:, 7] < -1.0) or np.any(actions[:, 7] > 1.0):
        raise ValueError("Canonical pickle gripper commands must be in [-1, 1].")

    transition_count = actions.shape[0]
    if len(observations) != transition_count + 1:
        raise ValueError(
            "Canonical pickle trajectories require T+1 observations for T actions: "
            f"observations={len(observations)}, actions={transition_count}."
        )

    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if rewards.shape[0] < transition_count:
        raise ValueError(
            "Canonical pickle trajectories require at least T rewards before "
            f"alignment: rewards={rewards.shape[0]}, actions={transition_count}."
        )
    rewards = rewards[:transition_count]
    if not np.isfinite(rewards).all():
        raise ValueError("Canonical pickle rewards contain non-finite values.")
    return actions, rewards
