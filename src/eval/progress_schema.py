from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


TASK_PROGRESS_SCHEMA: dict[str, dict[str, list[str]]] = {
    "one_leg": {
        "skill_states": [
            "top-leg-pick",
            "top-leg-push",
            "leg-top-pick",
            "leg-top-place",
            "leg-top-insert",
            "leg-top-screw",
        ],
        "assembly_steps": [
            "top-leg",
        ],
    },
    "round_table": {
        "skill_states": [
            "top-leg-push",
            "leg-top-pick",
            "leg-top-place",
            "leg-top-insert",
            "leg-top-screw",
            "base-leg-pick",
            "base-leg-place",
            "base-leg-insert",
            "base-leg-screw",
        ],
        "assembly_steps": [
            "top-leg",
            "leg-base",
        ],
    },
    "lamp": {
        "skill_states": [
            "base-bulb-push",
            "bulb-base-pick",
            "bulb-base-place",
            "bulb-base-insert",
            "bulb-base-screw",
            "hood-base-pick",
            "hood-base-place",
        ],
        "assembly_steps": [
            "base-bulb",
            "base-hood",
        ],
    },
}


def get_task_progress_labels(task_name: Optional[str], kind: str) -> list[str]:
    if task_name is None:
        return []
    return list(TASK_PROGRESS_SCHEMA.get(str(task_name), {}).get(kind, ()))


def normalize_progress_counts(
    counts: Optional[dict[str, int]],
    expected_labels: Iterable[str],
) -> dict[str, int]:
    normalized = {}
    raw_counts = counts or {}

    for label in expected_labels:
        normalized[str(label)] = int(raw_counts.get(label, 0))

    for key, value in raw_counts.items():
        key = str(key)
        if key in normalized:
            continue
        normalized[key] = int(value)

    return normalized


def normalize_annotation_label(label):
    if label is None:
        return None
    if hasattr(label, "item"):
        label = label.item()
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    return str(label)


def ordered_unique_non_null(labels):
    ordered_labels = []
    seen = set()
    for label in labels:
        normalized_label = normalize_annotation_label(label)
        if normalized_label is None or normalized_label in seen:
            continue
        seen.add(normalized_label)
        ordered_labels.append(normalized_label)
    return ordered_labels


def increment_ordered_counter(counter: dict[str, int], label: str):
    counter[label] = counter.get(label, 0) + 1


def accumulate_episode_skill_stats(
    state_labels,
    step_labels,
    success: bool,
    state_counts: dict[str, int],
    skill_completion_counts: dict[str, int],
    step_counts: dict[str, int],
    step_completion_counts: dict[str, int],
):
    ordered_states = ordered_unique_non_null(state_labels)
    for state_label in ordered_states:
        increment_ordered_counter(state_counts, state_label)
    for state_label in ordered_states[:-1]:
        increment_ordered_counter(skill_completion_counts, state_label)
    if success and ordered_states:
        increment_ordered_counter(skill_completion_counts, ordered_states[-1])

    ordered_steps = ordered_unique_non_null(step_labels)
    for step_label in ordered_steps:
        increment_ordered_counter(step_counts, step_label)
    for step_label in ordered_steps[:-1]:
        increment_ordered_counter(step_completion_counts, step_label)
    if success and ordered_steps:
        increment_ordered_counter(step_completion_counts, ordered_steps[-1])


def compute_success_rates(
    reached_counts: dict[str, int],
    completion_counts: dict[str, int],
) -> dict[str, float]:
    success_rates = {}
    for label, reached in reached_counts.items():
        completed = completion_counts.get(label, 0)
        success_rates[label] = completed / reached if reached > 0 else 0.0
    return success_rates


TRACKING_TOTAL_POS_SCALE_M = 0.01
TRACKING_TOTAL_ORI_SCALE_DEG = 5.0


def _as_float_array(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> Optional[np.ndarray]:
    quat = _as_float_array(quat_xyzw)
    if quat is None or quat.shape[-1] != 4:
        return None
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 1e-12:
        return None

    x, y, z, w = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _extract_ee_pose_from_robot_state(robot_state) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if robot_state is None:
        return None
    if not isinstance(robot_state, dict):
        return None

    # Guidance poses are projected/drawn in sim-local coordinates. Prefer the
    # matching EE pose when rollout saved it; fall back for old rollouts.
    ee_pos = _as_float_array(robot_state.get("ee_pos_sim", robot_state.get("ee_pos")))
    ee_quat = _as_float_array(robot_state.get("ee_quat_sim", robot_state.get("ee_quat")))
    if ee_pos is None or ee_quat is None:
        return None
    ee_pos = ee_pos.reshape(-1)
    ee_quat = ee_quat.reshape(-1)
    if ee_pos.shape[0] != 3 or ee_quat.shape[0] != 4:
        return None
    if not np.isfinite(ee_pos).all() or not np.isfinite(ee_quat).all():
        return None
    return ee_pos, ee_quat


def _tracking_error_at_frame(
    robot_state,
    target_pose,
    *,
    pos_scale_m: float = TRACKING_TOTAL_POS_SCALE_M,
    ori_scale_deg: float = TRACKING_TOTAL_ORI_SCALE_DEG,
) -> Optional[dict[str, float]]:
    ee_pose = _extract_ee_pose_from_robot_state(robot_state)
    target = _as_float_array(target_pose)
    if ee_pose is None or target is None or target.shape != (4, 4):
        return None
    if not np.isfinite(target).all():
        return None

    ee_pos, ee_quat = ee_pose
    ee_rot = _quat_xyzw_to_matrix(ee_quat)
    if ee_rot is None:
        return None

    target_pos = target[:3, 3]
    target_rot = target[:3, :3]
    pos_m = float(np.linalg.norm(ee_pos - target_pos))

    rot_delta = target_rot @ ee_rot.T
    cos_angle = (float(np.trace(rot_delta)) - 1.0) / 2.0
    cos_angle = min(1.0, max(-1.0, cos_angle))
    ori_deg = float(math.degrees(math.acos(cos_angle)))
    total = float(pos_m / pos_scale_m + ori_deg / ori_scale_deg)

    return {
        "pos_m": pos_m,
        "ori_deg": ori_deg,
        "total": total,
    }


def compute_episode_tracking_errors(
    robot_states,
    skill_states,
    target_poses,
    *,
    pos_scale_m: float = TRACKING_TOTAL_POS_SCALE_M,
    ori_scale_deg: float = TRACKING_TOTAL_ORI_SCALE_DEG,
) -> dict[str, dict[str, float]]:
    """Compute per-skill final-ee-vs-target error, keeping min for repeated skills."""
    n_frames = min(len(robot_states or []), len(skill_states or []), len(target_poses or []))
    if n_frames <= 0:
        return {}

    per_skill: dict[str, dict[str, float]] = {}
    segment_label = normalize_annotation_label(skill_states[0])
    segment_start = 0

    def close_segment(end_idx_exclusive: int):
        nonlocal segment_label, segment_start
        if segment_label is None:
            return
        final_idx = end_idx_exclusive - 1
        if final_idx < segment_start:
            return
        error = _tracking_error_at_frame(
            robot_states[final_idx],
            target_poses[final_idx],
            pos_scale_m=pos_scale_m,
            ori_scale_deg=ori_scale_deg,
        )
        if error is None:
            return
        previous = per_skill.get(segment_label)
        if previous is None or error["total"] < previous["total"]:
            per_skill[segment_label] = error

    for frame_idx in range(1, n_frames):
        current_label = normalize_annotation_label(skill_states[frame_idx])
        if current_label == segment_label:
            continue
        close_segment(frame_idx)
        segment_label = current_label
        segment_start = frame_idx

    close_segment(n_frames)
    return per_skill


def accumulate_tracking_error_records(
    accumulator: dict[str, list[dict[str, float]]],
    episode_errors: dict[str, dict[str, float]],
) -> None:
    for skill_state, error in episode_errors.items():
        accumulator.setdefault(skill_state, []).append(error)


def _summarize_tracking_error_list(errors: list[dict[str, float]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"count": len(errors)}
    if not errors:
        return {
            **summary,
            "mean_pos_m": 0.0,
            "mean_ori_deg": 0.0,
            "mean_total": 0.0,
            "min_pos_m": 0.0,
            "min_ori_deg": 0.0,
            "min_total": 0.0,
            "max_pos_m": 0.0,
            "max_ori_deg": 0.0,
            "max_total": 0.0,
        }

    for field, output_name in (
        ("pos_m", "pos_m"),
        ("ori_deg", "ori_deg"),
        ("total", "total"),
    ):
        values = np.asarray([float(error[field]) for error in errors], dtype=np.float64)
        summary[f"mean_{output_name}"] = float(values.mean())
        summary[f"min_{output_name}"] = float(values.min())
        summary[f"max_{output_name}"] = float(values.max())
    return summary


def build_tracking_error_summary(
    accumulator: dict[str, list[dict[str, float]]],
    *,
    expected_labels: Iterable[str] = (),
    pos_scale_m: float = TRACKING_TOTAL_POS_SCALE_M,
    ori_scale_deg: float = TRACKING_TOTAL_ORI_SCALE_DEG,
) -> dict:
    by_skill: dict[str, dict[str, float | int]] = {}
    all_labels = list(expected_labels)
    for label in accumulator.keys():
        if label not in all_labels:
            all_labels.append(label)

    all_errors: list[dict[str, float]] = []
    for label in all_labels:
        errors = accumulator.get(label, [])
        by_skill[label] = _summarize_tracking_error_list(errors)
        all_errors.extend(errors)

    return {
        "pos_scale_m": float(pos_scale_m),
        "ori_scale_deg": float(ori_scale_deg),
        "overall": _summarize_tracking_error_list(all_errors),
        "by_skill": by_skill,
    }
