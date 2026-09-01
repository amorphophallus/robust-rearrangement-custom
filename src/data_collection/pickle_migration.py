"""Migration of legacy FurnitureBench rollouts to the shared raw contract."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.data_collection.pickle_contract import (
    CANONICAL_IMAGE_SIZE,
    camera_calibration_to_robot_base,
    center_crop_camera_calibration,
    center_crop_grasp_mapping,
    center_crop_observation_images,
    center_crop_point_mapping,
    flattened_poses_to_robot_base,
    legacy_sim_local_to_robot_base_matrix,
    normalize_depth_meters,
    point_to_robot_base,
    pose_to_robot_base,
    robot_state_with_base_frame_aliases,
    validate_and_align_pickle_timeseries,
)


def canonicalize_furniturebench_trajectory(
    data,
    *,
    legacy_pos_actions_are_delta: bool = False,
):
    """Return a canonical 224px copy of one legacy FB trajectory.

    Historical FB writers converted absolute targets to delta actions but left
    ``action_type='pos'`` in some campaigns. That ambiguity is never guessed:
    callers must opt in after inspecting the source campaign.
    """
    if not isinstance(data, Mapping):
        raise TypeError("FurnitureBench trajectory must be a mapping.")
    if data.get("env", "FurnitureBench") != "FurnitureBench":
        raise ValueError("Only FurnitureBench trajectories can use this migration.")
    source_action_type = data.get("action_type")
    if source_action_type == "pos" and not legacy_pos_actions_are_delta:
        raise ValueError(
            "Legacy trajectory says action_type='pos'. Pass "
            "legacy_pos_actions_are_delta=True only after verifying that the "
            "stored values were already converted to deltas."
        )
    if source_action_type not in ("delta", "pos"):
        raise ValueError(f"Unsupported action_type {source_action_type!r}.")

    observations = data.get("observations")
    if not observations:
        raise ValueError("FurnitureBench trajectory has no observations.")
    original_first_state = observations[0]["robot_state"]
    sim_local_to_base = legacy_sim_local_to_robot_base_matrix(
        original_first_state
    )
    canonical_observations = []
    for observation in observations:
        observation = dict(observation)
        source_shapes = {
            f"color_image{camera_idx}": tuple(
                np.asarray(observation[f"color_image{camera_idx}"]).shape[:2]
            )
            for camera_idx in (1, 2)
        }
        color1, color2, depth1, depth2 = center_crop_observation_images(
            np.asarray(observation["color_image1"])[None],
            np.asarray(observation["color_image2"])[None],
            normalize_depth_meters(observation["depth_image1"])[None],
            normalize_depth_meters(observation["depth_image2"])[None],
            CANONICAL_IMAGE_SIZE,
        )
        observation["color_image1"] = color1[0]
        observation["color_image2"] = color2[0]
        observation["depth_image1"] = depth1[0]
        observation["depth_image2"] = depth2[0]
        observation["robot_state"] = robot_state_with_base_frame_aliases(
            observation["robot_state"]
        )
        observation["parts_poses"] = flattened_poses_to_robot_base(
            observation["parts_poses"], sim_local_to_base
        )
        for key in ("guidance_point", "guidance_point_clean"):
            observation[key] = point_to_robot_base(
                observation.get(key), sim_local_to_base
            )
        for key in ("guidance_pose", "guidance_pose_clean"):
            observation[key] = pose_to_robot_base(
                observation.get(key), sim_local_to_base
            )
        if "guidance_point_2d" in observation:
            observation["guidance_point_2d"] = center_crop_point_mapping(
                observation["guidance_point_2d"],
                source_shapes,
                CANONICAL_IMAGE_SIZE,
            )
        if "oracle_guidance_point_2d" in observation:
            observation["oracle_guidance_point_2d"] = center_crop_point_mapping(
                observation["oracle_guidance_point_2d"],
                source_shapes,
                CANONICAL_IMAGE_SIZE,
            )
        if "grasp_annotation_2d" in observation:
            observation["grasp_annotation_2d"] = center_crop_grasp_mapping(
                observation["grasp_annotation_2d"],
                source_shapes,
                CANONICAL_IMAGE_SIZE,
            )
        canonical_observations.append(observation)

    actions, rewards = validate_and_align_pickle_timeseries(
        canonical_observations, data["actions"], data.get("rewards", [])
    )
    camera_info = dict(data.get("camera_info") or {})
    front_camera = center_crop_camera_calibration(
        camera_info.get("front_camera"), CANONICAL_IMAGE_SIZE
    )
    camera_info["front_camera"] = camera_calibration_to_robot_base(
        front_camera, original_first_state
    )

    canonical = dict(data)
    canonical.update(
        {
            "env": "FurnitureBench",
            "observations": canonical_observations,
            "actions": actions.tolist(),
            "rewards": rewards.tolist(),
            "camera_info": camera_info,
            "action_type": "delta",
        }
    )
    return canonical


__all__ = ["canonicalize_furniturebench_trajectory"]
