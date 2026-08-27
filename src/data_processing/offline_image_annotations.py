from __future__ import annotations

from typing import Mapping

import numpy as np

from src.common.image_annotations import (
    draw_grasp_annotation_on_image,
    draw_guidance_point_on_image,
)
from src.common.guidance import camera_info_for_image
from src.eval.skill_annotation_util import (
    project_3d_to_2d,
    project_pose_to_grasp_annotation_2d,
)


IMAGE_ANNOTATION_MODES = (
    "none",
    "guidance-point",
    "guidance-point-colored",
    "grasp-part",
    "grasp-part-colored",
)
GRASP_SKILLS = {"pick", "place"}


def _annotation_for_camera(observation: Mapping, key: str, camera: str):
    annotation = observation.get(key)
    if not isinstance(annotation, Mapping):
        return None
    return annotation.get(camera)


def _project_saved_robot_base_annotation(
    observation: Mapping,
    trajectory_camera_info: Mapping | None,
    *,
    camera: str,
    kind: str,
):
    """Project canonical 3-D data when robot-base camera extrinsics are saved.

    Real and legacy pickles may only contain their already-computed 2-D
    annotations.  Those remain a supported fallback; migrated/new simulator
    pickles use this path so pickle-to-LMDB never depends on stale frame-specific
    pixels.
    """

    camera_info = camera_info_for_image(trajectory_camera_info, camera)
    if not isinstance(camera_info, Mapping) or "robot_base_to_camera" not in camera_info:
        return None
    if kind == "point":
        point = observation.get("guidance_point")
        return None if point is None else project_3d_to_2d(point, camera_info)
    if kind == "grasp":
        pose = observation.get("guidance_pose")
        if pose is None:
            return None
        width = observation.get("guidance_gripper_width")
        if width is not None:
            width = float(np.asarray(width).reshape(-1)[0])
        return project_pose_to_grasp_annotation_2d(
            pose,
            camera_info,
            gripper_width=width,
        )
    raise ValueError(f"Unknown annotation projection kind: {kind}")


def annotate_observation_image(
    observation: Mapping,
    mode: str,
    *,
    camera: str = "color_image2",
    trajectory_camera_info: Mapping | None = None,
):
    """Return a shallow observation copy with deterministic offline annotation."""
    if mode not in IMAGE_ANNOTATION_MODES:
        raise ValueError(
            f"Unknown image annotation mode {mode!r}; expected one of "
            f"{IMAGE_ANNOTATION_MODES}."
        )

    output = dict(observation)
    if mode == "none":
        return output

    image = observation.get(camera)
    if image is None:
        raise ValueError(
            f"Cannot apply image annotation mode {mode!r}: missing {camera!r}."
        )

    image = np.asarray(image)
    skill = observation.get("skill")
    if isinstance(skill, bytes):
        skill = skill.decode("utf-8")
    colored = mode.endswith("-colored")

    if mode.startswith("guidance-point"):
        point_2d = _project_saved_robot_base_annotation(
            observation,
            trajectory_camera_info,
            camera=camera,
            kind="point",
        )
        if point_2d is None:
            point_2d = _annotation_for_camera(
                observation, "guidance_point_2d", camera
            )
        output[camera] = draw_guidance_point_on_image(
            image,
            point_2d,
            skill=skill,
            use_skill_color=colored,
        )
        return output

    if skill in GRASP_SKILLS:
        grasp_2d = _project_saved_robot_base_annotation(
            observation,
            trajectory_camera_info,
            camera=camera,
            kind="grasp",
        )
        if grasp_2d is None:
            grasp_2d = _annotation_for_camera(
                observation, "grasp_annotation_2d", camera
            )
        output[camera] = draw_grasp_annotation_on_image(
            image,
            grasp_2d,
            skill=skill,
            use_skill_color=colored,
        )
    else:
        point_2d = _project_saved_robot_base_annotation(
            observation,
            trajectory_camera_info,
            camera=camera,
            kind="point",
        )
        if point_2d is None:
            point_2d = _annotation_for_camera(
                observation, "guidance_point_2d", camera
            )
        output[camera] = draw_guidance_point_on_image(
            image,
            point_2d,
            skill=skill,
            use_skill_color=colored,
        )
    return output
