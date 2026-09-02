"""Reconstruct the legacy real demonstrations on a nominal action-time grid.

This module is deliberately limited to the old ``raw_v2`` recorder contract.
New recordings must already contain their authoritative fixed-rate timeline in
the pickle and must never pass through this reconstruction path.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.data_processing.offline_image_annotations import annotate_observation_image
from src.eval.real_skill_annotation_util import RealSkillAnnotationSession
from src.real.align_pickles import (
    _camera_matches,
    build_aligned_observation,
)
from src.real.time_alignment import interpolate_quaternion_xyzw, interpolate_vector


LEGACY_TIMELINE_SCHEMA = "rr_legacy_real_uniform_timeline_v1"
TIMELINE_KEYS = (
    "obs_valid",
    "timeline_timestamp_ns",
    "source_action_index",
    "source_action_timestamp_ns",
    "action_source_recorded",
)


def build_uniform_action_grid(
    action_times_ns: Sequence[int],
    *,
    frequency_hz: float = 10.0,
    max_quantization_residual_ms: float = 75.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map every source action to a unique, monotonic fixed-rate grid slot."""

    times = np.asarray(action_times_ns, dtype=np.int64)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("action_times_ns must be a non-empty 1-D sequence")
    if np.any(np.diff(times) <= 0):
        raise ValueError("action_times_ns must increase strictly")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")

    period_ns = int(round(1_000_000_000.0 / float(frequency_hz)))
    source_slots = np.rint((times - times[0]) / period_ns).astype(np.int64)
    source_slots[0] = 0
    for index in range(1, len(source_slots)):
        source_slots[index] = max(source_slots[index], source_slots[index - 1] + 1)

    timeline_times = times[0] + np.arange(source_slots[-1] + 1, dtype=np.int64) * period_ns
    residual_ms = np.abs(times - timeline_times[source_slots]) / 1_000_000.0
    if float(np.max(residual_ms)) > float(max_quantization_residual_ms):
        raise ValueError(
            "legacy action-to-grid residual exceeds the configured limit: "
            f"max={float(np.max(residual_ms)):.3f} ms, "
            f"limit={float(max_quantization_residual_ms):.3f} ms"
        )
    return timeline_times, source_slots, residual_ms


def _interpolate_robot_state(
    action_times_ns: np.ndarray,
    states: Sequence[Mapping[str, Any]],
    target_time_ns: int,
) -> Dict[str, Any]:
    """Interpolate continuous legacy state fields and hold discrete fields."""

    right = int(np.searchsorted(action_times_ns, target_time_ns, side="left"))
    if right <= 0:
        return copy.deepcopy(states[0])
    if right >= len(states):
        return copy.deepcopy(states[-1])
    if int(action_times_ns[right]) == int(target_time_ns):
        return copy.deepcopy(states[right])

    left = right - 1
    pair_times = [int(action_times_ns[left]), int(action_times_ns[right])]
    result = copy.deepcopy(states[left])
    vector_fields = (
        "ee_pos",
        "ee_pos_vel",
        "ee_ori_vel",
        "joint_positions",
        "joint_velocities",
        "joint_torques",
        "ee_pos_original",
    )
    for field in vector_fields:
        if field in states[left] and field in states[right]:
            result[field] = interpolate_vector(
                pair_times,
                [states[left][field], states[right][field]],
                int(target_time_ns),
            )
    for field in ("ee_quat", "ee_quat_original"):
        if field in states[left] and field in states[right]:
            result[field] = interpolate_quaternion_xyzw(
                pair_times,
                [states[left][field], states[right][field]],
                int(target_time_ns),
            )

    # Matrices contain both translation and rotation.  The policy processor
    # primarily consumes ee_pos/ee_quat, but the wrist pose is required by the
    # geometry annotator on source frames.
    for field in ("ee_pose", "ee_pose_original", "wrist_pose"):
        if field not in states[left] or field not in states[right]:
            continue
        left_matrix = np.asarray(states[left][field], dtype=np.float64)
        right_matrix = np.asarray(states[right][field], dtype=np.float64)
        if left_matrix.shape != (4, 4) or right_matrix.shape != (4, 4):
            continue
        from scipy.spatial.transform import Rotation

        position = interpolate_vector(
            pair_times,
            [left_matrix[:3, 3], right_matrix[:3, 3]],
            int(target_time_ns),
        )
        quaternion = interpolate_quaternion_xyzw(
            pair_times,
            [
                Rotation.from_matrix(left_matrix[:3, :3]).as_quat(),
                Rotation.from_matrix(right_matrix[:3, :3]).as_quat(),
            ],
            int(target_time_ns),
        )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        matrix[:3, 3] = position
        result[field] = matrix

    # Gripper state is discrete for temporal reconstruction.  Do not create a
    # fictitious partially closed state across a grasp transition.
    if "gripper_width" in states[left]:
        result["gripper_width"] = states[left]["gripper_width"]
    return result


def _zero_visual_observation(
    template: Mapping[str, Any],
    *,
    robot_state: Mapping[str, Any],
    skill: Any,
    parts_poses: Any,
) -> Dict[str, Any]:
    observation: Dict[str, Any] = {
        "robot_state": copy.deepcopy(robot_state),
        "skill": skill,
        "parts_poses": np.asarray(parts_poses).copy(),
    }
    for key in ("color_image1", "color_image2", "depth_image1", "depth_image2"):
        value = template.get(key)
        if value is None:
            raise ValueError(f"legacy RGB-D source is missing {key}")
        observation[key] = np.zeros_like(np.asarray(value))
    return observation


def reconstruct_legacy_real_trajectory(
    data: Mapping[str, Any],
    *,
    frequency_hz: float = 10.0,
    max_quantization_residual_ms: float = 75.0,
    max_camera_residual_ms: float = 75.0,
    image_annotation_mode: str = "guidance-point-colored",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return one dense legacy trajectory and a JSON-safe reconstruction report."""

    observations = data.get("observations")
    actions = data.get("actions")
    if not isinstance(observations, list) or not observations or actions is None:
        raise ValueError("legacy trajectory requires observations and actions")
    action_count = len(actions)
    if len(observations) not in {action_count, action_count + 1}:
        raise ValueError("legacy trajectory must contain N or N+1 observations")
    source_observations = observations[:action_count]
    metadata = data.get("metadata", {})
    if not isinstance(metadata, Mapping) or metadata.get("schema") != "deoxys_furniturebench_raw_v2":
        raise ValueError("legacy-real-10hz only accepts deoxys_furniturebench_raw_v2")

    action_times = np.asarray(
        [int(observation["control_wall_time_ns"]) for observation in source_observations],
        dtype=np.int64,
    )
    timeline_times, source_slots, residual_ms = build_uniform_action_grid(
        action_times,
        frequency_hz=frequency_hz,
        max_quantization_residual_ms=max_quantization_residual_ms,
    )
    slot_to_source = {int(slot): index for index, slot in enumerate(source_slots)}

    front_matches = _camera_matches(
        source_observations, action_times.tolist(), "front", max_camera_residual_ms
    )
    wrist_matches = _camera_matches(
        source_observations, action_times.tolist(), "wrist", max_camera_residual_ms
    )
    valid_sources = set(front_matches) & set(wrist_matches)

    session = RealSkillAnnotationSession(
        str(data.get("furniture", data.get("task"))),
        data.get("camera_info"),
        mode="offline",
    )
    annotated_sources: Dict[int, Dict[str, Any]] = {}
    source_skills = []
    for source_index, source_observation in enumerate(source_observations):
        if source_index in valid_sources:
            working = build_aligned_observation(
                source_observations,
                source_index,
                front_matches[source_index],
                wrist_matches[source_index],
            )
        else:
            working = dict(source_observation)
            if isinstance(source_observation.get("robot_state"), Mapping):
                working["robot_state"] = copy.deepcopy(source_observation["robot_state"])
        session.annotate_observation(working)
        source_skills.append(working.get("skill"))
        if source_index in valid_sources:
            annotated_sources[source_index] = annotate_observation_image(
                working,
                image_annotation_mode,
                trajectory_camera_info=data.get("camera_info"),
            )

    source_actions = np.asarray(actions, dtype=np.float32)
    timeline_actions = np.zeros((len(timeline_times), 8), dtype=np.float32)
    timeline_actions[:, 6] = 1.0  # identity quaternion xyzw
    timeline_rewards = np.zeros(len(timeline_times), dtype=np.float32)
    source_rewards = np.asarray(
        data.get("rewards", np.zeros(action_count)), dtype=np.float32
    )[:action_count]
    source_action_index = np.full(len(timeline_times), -1, dtype=np.int32)
    source_action_timestamp = np.full(len(timeline_times), -1, dtype=np.int64)
    action_source_recorded = np.zeros(len(timeline_times), dtype=np.bool_)

    gripper_command = float(np.sign(source_actions[0, -1]))
    source_cursor = 0
    for slot in range(len(timeline_times)):
        while source_cursor + 1 < action_count and source_slots[source_cursor + 1] <= slot:
            source_cursor += 1
        gripper_command = float(np.sign(source_actions[source_cursor, -1]))
        timeline_actions[slot, -1] = gripper_command
        source_index = slot_to_source.get(slot)
        if source_index is None:
            continue
        timeline_actions[slot] = source_actions[source_index]
        timeline_rewards[slot] = source_rewards[source_index]
        source_action_index[slot] = source_index
        source_action_timestamp[slot] = action_times[source_index]
        action_source_recorded[slot] = True

    source_states = [observation["robot_state"] for observation in source_observations]
    dense_observations = []
    obs_valid = np.zeros(len(timeline_times), dtype=np.bool_)
    last_skill = source_skills[0]
    for slot, target_time in enumerate(timeline_times):
        source_index = slot_to_source.get(slot)
        if source_index is not None:
            last_skill = source_skills[source_index]
        if source_index is not None and source_index in annotated_sources:
            observation = annotated_sources[source_index]
            obs_valid[slot] = True
        else:
            nearest_index = int(np.clip(np.searchsorted(action_times, target_time), 0, action_count - 1))
            if nearest_index > 0 and abs(int(target_time) - int(action_times[nearest_index - 1])) <= abs(
                int(action_times[nearest_index]) - int(target_time)
            ):
                nearest_index -= 1
            observation = _zero_visual_observation(
                source_observations[0],
                robot_state=_interpolate_robot_state(
                    action_times, source_states, int(target_time)
                ),
                skill=last_skill,
                parts_poses=source_observations[nearest_index].get("parts_poses", []),
            )
        dense_observations.append(observation)

    output = dict(data)
    output["observations"] = dense_observations
    output["actions"] = timeline_actions
    output["rewards"] = timeline_rewards
    output.pop("actions_original", None)
    output["obs_valid"] = obs_valid
    output["timeline_timestamp_ns"] = timeline_times
    output["source_action_index"] = source_action_index
    output["source_action_timestamp_ns"] = source_action_timestamp
    output["action_source_recorded"] = action_source_recorded
    output_metadata = copy.deepcopy(dict(metadata))
    output_metadata.update(
        {
            "schema": LEGACY_TIMELINE_SCHEMA,
            "timeline_frequency_hz": float(frequency_hz),
            "timeline_period_ns": int(round(1_000_000_000.0 / frequency_hz)),
            "legacy_salvage": True,
            "source_annotation_source": data.get("annotation_source"),
            "annotation_implementation": "real_skill_annotation_util",
            "image_annotation_mode": image_annotation_mode,
        }
    )
    output["metadata"] = output_metadata
    session.update_trajectory_metadata(output)
    output["annotation_source"] = "scripted"

    report = {
        "schema": LEGACY_TIMELINE_SCHEMA,
        "source_actions": action_count,
        "timeline_steps": len(timeline_times),
        "synthetic_noop_steps": int(len(timeline_times) - action_count),
        "valid_observations": int(np.count_nonzero(obs_valid)),
        "quantization_residual_ms_p95": float(np.percentile(residual_ms, 95)),
        "quantization_residual_ms_max": float(np.max(residual_ms)),
    }
    return output, report
