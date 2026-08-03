from __future__ import annotations

from typing import Mapping

import numpy as np

from src.common.image_annotations import (
    draw_grasp_annotation_on_image,
    draw_guidance_point_on_image,
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


def annotate_observation_image(
    observation: Mapping,
    mode: str,
    *,
    camera: str = "color_image2",
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
        output[camera] = draw_guidance_point_on_image(
            image,
            _annotation_for_camera(observation, "guidance_point_2d", camera),
            skill=skill,
            use_skill_color=colored,
        )
        return output

    if skill in GRASP_SKILLS:
        output[camera] = draw_grasp_annotation_on_image(
            image,
            _annotation_for_camera(observation, "grasp_annotation_2d", camera),
            skill=skill,
            use_skill_color=colored,
        )
    else:
        output[camera] = draw_guidance_point_on_image(
            image,
            _annotation_for_camera(observation, "guidance_point_2d", camera),
            skill=skill,
            use_skill_color=colored,
        )
    return output
