"""Coordinate-frame helpers for guidance points and poses.

The stable ``guidance_point`` and ``guidance_pose`` keys use the Panda robot
base frame.  Legacy simulator rollouts stored those values in the environment's
sim-local frame; camera metadata may retain both transforms so those rollouts
remain inspectable without making the canonical frame ambiguous.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.common.eepose import ROBOT_BASE, SIM_LOCAL


GUIDANCE_FRAME = ROBOT_BASE
GUIDANCE_SCHEMA_VERSION = 2


def normalize_guidance_frame(value: Any) -> str:
    """Normalize supported guidance-frame spellings."""

    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "base": ROBOT_BASE,
        "robot": ROBOT_BASE,
        "robot-base": ROBOT_BASE,
        "sim": SIM_LOCAL,
        "sim-local": SIM_LOCAL,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported guidance frame {value!r}; expected robot-base or sim-local"
        ) from exc


def translation_transform(translation: Any) -> np.ndarray:
    """Return a float32 homogeneous transform with an identity rotation."""

    value = np.asarray(translation, dtype=np.float32).reshape(3)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = value
    return transform


def transform_guidance_point(point: Any, source_to_target: Any) -> np.ndarray:
    """Transform one 3-D guidance point with a homogeneous transform."""

    point_array = np.asarray(point, dtype=np.float32).reshape(3)
    transform = np.asarray(source_to_target, dtype=np.float32).reshape(4, 4)
    point_h = np.ones(4, dtype=np.float32)
    point_h[:3] = point_array
    return (transform @ point_h)[:3].astype(np.float32)


def transform_guidance_pose(pose: Any, source_to_target: Any) -> np.ndarray:
    """Transform one 4x4 guidance pose with a homogeneous transform."""

    pose_array = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    transform = np.asarray(source_to_target, dtype=np.float32).reshape(4, 4)
    return (transform @ pose_array).astype(np.float32)


def robot_base_to_sim_local_from_state(robot_state: Mapping[str, Any]) -> np.ndarray:
    """Recover the simulator robot-base origin from paired EE positions.

    FurnitureBench's simulated Panda base axes are aligned with sim-local, so
    the frame difference is a translation.  Deriving it from each rollout keeps
    migration code independent of a particular environment placement.
    """

    missing = [key for key in ("ee_pos", "ee_pos_sim") if key not in robot_state]
    if missing:
        raise KeyError(
            "cannot recover robot-base/sim-local transform; missing "
            + ", ".join(missing)
        )
    robot_position = np.asarray(robot_state["ee_pos"], dtype=np.float32).reshape(3)
    sim_position = np.asarray(robot_state["ee_pos_sim"], dtype=np.float32).reshape(3)
    return translation_transform(sim_position - robot_position)


def camera_info_with_robot_base(
    camera_info: Mapping[str, Any],
    robot_base_to_sim_local: Any,
) -> dict[str, Any]:
    """Add canonical robot-base camera extrinsics while retaining legacy keys."""

    output = dict(camera_info)
    robot_to_sim = np.asarray(robot_base_to_sim_local, dtype=np.float32).reshape(4, 4)
    if "camera_to_sim_local" in output:
        camera_to_sim = np.asarray(output["camera_to_sim_local"], dtype=np.float32)
    elif "sim_local_to_camera" in output:
        camera_to_sim = np.linalg.inv(
            np.asarray(output["sim_local_to_camera"], dtype=np.float32)
        ).astype(np.float32)
    else:
        raise KeyError(
            "camera metadata must contain camera_to_sim_local or sim_local_to_camera"
        )

    sim_to_robot = np.linalg.inv(robot_to_sim).astype(np.float32)
    camera_to_robot = (sim_to_robot @ camera_to_sim).astype(np.float32)
    output["camera_to_robot_base"] = camera_to_robot
    output["robot_base_to_camera"] = np.linalg.inv(camera_to_robot).astype(np.float32)
    output["point_frame"] = ROBOT_BASE
    return output


def camera_info_for_image(
    trajectory_camera_info: Mapping[str, Any] | None,
    image_key: str,
) -> Mapping[str, Any] | None:
    """Resolve saved per-image camera metadata across rollout schema versions."""

    if not isinstance(trajectory_camera_info, Mapping):
        return None
    direct = trajectory_camera_info.get(image_key)
    if isinstance(direct, Mapping):
        return direct
    alias = {
        "color_image2": "front_camera",
        "color_image1": "wrist_camera",
    }.get(image_key)
    candidate = trajectory_camera_info.get(alias) if alias is not None else None
    return candidate if isinstance(candidate, Mapping) else None
