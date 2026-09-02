"""Strict contract checks for offline-buffered Deoxys demonstrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


V6_BUFFERED_SCHEMA = "deoxys_furniturebench_raw_v6_offline_buffered"


class V6PickleContractError(ValueError):
    pass


def _fail(source: str, message: str) -> None:
    raise V6PickleContractError(f"{source}: {message}")


def _contains_vlm_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "vlm" in str(key).lower() or _contains_vlm_key(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        return any(_contains_vlm_key(child) for child in value)
    return False


def is_v6_buffered_trajectory(trajectory: Any) -> bool:
    return (
        isinstance(trajectory, Mapping)
        and isinstance(trajectory.get("metadata"), Mapping)
        and trajectory["metadata"].get("schema") == V6_BUFFERED_SCHEMA
    )


def _assert_projection_equal(stored: Any, expected: Any, source: str) -> None:
    if stored is None or expected is None:
        if stored is not expected:
            _fail(source, "nullability disagrees with same-frame reprojection")
        return
    stored_array = np.asarray(stored, dtype=np.float64).reshape(-1)
    expected_array = np.asarray(expected, dtype=np.float64).reshape(-1)
    if stored_array.shape != (2,) or expected_array.shape != (2,):
        _fail(source, "expected a 2-D pixel")
    if not np.all(np.isfinite(stored_array)) or not np.allclose(
        stored_array, expected_array, atol=1.0, rtol=0.0
    ):
        _fail(
            source,
            "does not match the same-frame calibrated 3-D guidance point "
            f"(stored={stored_array.tolist()}, projected={expected_array.tolist()})",
        )


def validate_v6_buffered_trajectory(
    trajectory: Any,
    *,
    source: str = "<pickle>",
    verify_projection: bool = True,
) -> dict[str, Any]:
    """Validate the production v6 timing, provenance, and geometry contract."""

    if not isinstance(trajectory, Mapping):
        _fail(source, "trajectory must be a mapping")
    metadata = trajectory.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != V6_BUFFERED_SCHEMA:
        _fail(source, f"expected metadata.schema={V6_BUFFERED_SCHEMA!r}")
    expected = {
        "env": "FurnitureBench",
        "annotation_source": "scripted",
        "image_annotation_mode": "none",
    }
    for key, value in expected.items():
        if trajectory.get(key) != value:
            _fail(source, f"expected {key}={value!r}, got {trajectory.get(key)!r}")
    if _contains_vlm_key(trajectory):
        _fail(source, "VLM metadata is forbidden in production scripted data")

    actions = np.asarray(trajectory.get("actions", []), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 8:
        _fail(source, f"actions must have nonempty shape (N,8), got {actions.shape}")
    frame_count = actions.shape[0]
    observations = trajectory.get("observations", [])
    arrays = {
        "observations": observations,
        "actions_original": trajectory.get("actions_original", []),
        "actions_absolute": trajectory.get("actions_absolute", []),
        "action_timing": trajectory.get("action_timing", []),
        "action_target_timestamps_ns": trajectory.get(
            "action_target_timestamps_ns", []
        ),
        "action_timestamps_ns": trajectory.get("action_timestamps_ns", []),
        "obs_valid": trajectory.get("obs_valid", []),
        "rewards": trajectory.get("rewards", []),
    }
    lengths = {key: len(value) for key, value in arrays.items()}
    if any(length != frame_count for length in lengths.values()):
        _fail(source, f"all v6 arrays must have N={frame_count} entries, got {lengths}")

    obs_valid = np.asarray(arrays["obs_valid"], dtype=np.bool_).reshape(-1)
    if not np.all(obs_valid):
        _fail(source, "obs_valid must be true for every v6 frame")
    targets = np.asarray(
        arrays["action_target_timestamps_ns"], dtype=np.int64
    ).reshape(-1)
    aliases = np.asarray(arrays["action_timestamps_ns"], dtype=np.int64).reshape(-1)
    if not np.array_equal(targets, aliases):
        _fail(source, "action_timestamps_ns must exactly alias the target-time grid")
    period_ns = int(metadata.get("action_period_ns", 0))
    frequency_hz = float(metadata.get("recording_frequency_hz", 0.0))
    if period_ns <= 0 or not np.isclose(frequency_hz, 10.0, atol=1e-6, rtol=0.0):
        _fail(source, "v6 production data must declare a 10 Hz positive-period grid")
    if not np.isclose(period_ns, 1e9 / frequency_hz, atol=1_000, rtol=0.0):
        _fail(source, "action_period_ns disagrees with recording_frequency_hz")
    if frame_count > 1 and not np.all(np.abs(np.diff(targets) - period_ns) <= 1_000):
        _fail(source, "action target timestamps are not a continuous 10 Hz grid")

    annotation = metadata.get("real_skill_annotation")
    if not isinstance(annotation, Mapping):
        _fail(source, "missing real_skill_annotation metadata")
    if annotation.get("mode") != "offline" or annotation.get("complete") is not True:
        _fail(source, "scripted annotation must be complete and offline")
    provenance = metadata.get("annotation_provenance")
    if not isinstance(provenance, Mapping):
        _fail(source, "missing annotation_provenance metadata")
    if (
        provenance.get("source") != "scripted"
        or provenance.get("stage") != "after_target_time_selection"
        or provenance.get("rgb_pixels_modified") is not False
    ):
        _fail(source, "annotation provenance does not satisfy the v6 contract")
    if not isinstance(metadata.get("offline_buffer_alignment"), Mapping):
        _fail(source, "missing offline buffer alignment audit")

    prompt_config = metadata.get("prompt_depth_anything")
    if not isinstance(prompt_config, Mapping) or prompt_config.get("online") is not False:
        _fail(source, "PromptDA must be configured offline")
    prompt_cameras = set(prompt_config.get("cameras", ()))
    if prompt_cameras != {"front", "wrist"}:
        _fail(source, "production RGB-D data requires offline PromptDA on both cameras")

    projection_annotator = None
    if verify_projection:
        if trajectory.get("task", trajectory.get("furniture")) != "one_leg":
            _fail(source, "v6 projection audit currently supports only one_leg")
        from src.eval.real_skill_annotation_util import RealSkillAnnotationSession

        projection_annotator = RealSkillAnnotationSession(
            "one_leg", trajectory.get("camera_info"), mode="offline"
        ).annotator

    for index, (observation, target_ns, timing) in enumerate(
        zip(observations, targets, arrays["action_timing"])
    ):
        path = f"{source}.observations[{index}]"
        if not isinstance(observation, Mapping):
            _fail(path, "observation must be a mapping")
        if int(observation.get("observation_target_wall_time_ns", -1)) != int(target_ns):
            _fail(path, "observation target time differs from action master grid")
        if int(timing.get("action_target_wall_time_ns", -1)) != int(target_ns):
            _fail(path, "action_timing target differs from action master grid")
        for key in (
            "color_image1",
            "color_image2",
            "depth_image1",
            "depth_image2",
            "depth_image1_realsense",
            "depth_image2_realsense",
            "robot_state",
            "parts_poses",
            "skill",
            "guidance_point_2d",
        ):
            if observation.get(key) is None:
                _fail(path, f"required field {key!r} is missing")
        if not isinstance(observation["guidance_point_2d"], Mapping):
            _fail(path, "guidance_point_2d must be a mapping")
        if projection_annotator is not None:
            point = observation.get("guidance_point_clean")
            if point is None:
                point = observation.get("guidance_point")
            pose = observation.get("guidance_pose_clean")
            if pose is None:
                pose = observation.get("guidance_pose")
            projected, _ = projection_annotator._camera_projections(
                observation,
                trajectory["camera_info"],
                point,
                pose,
                observation.get("guidance_gripper_width"),
            )
            for image_key in ("color_image1", "color_image2"):
                _assert_projection_equal(
                    observation["guidance_point_2d"].get(image_key),
                    projected.get(image_key),
                    f"{path}.guidance_point_2d.{image_key}",
                )

    return {
        "schema": V6_BUFFERED_SCHEMA,
        "frames": frame_count,
        "frequency_hz": frequency_hz,
        "projection_frames_checked": frame_count if verify_projection else 0,
    }
