from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_clean_train_noise_eval import (
    CONDITIONS,
    GRASP_NOISE_LEVELS,
    POINT_NOISE_LEVELS,
    REPO_ROOT,
    _rollout_group_dirs,
)
from src.eval.progress_schema import (
    _tracking_error_at_frame,
    build_tracking_error_summary,
    build_tracking_workspace_filter_summary,
    get_task_progress_labels,
    new_tracking_workspace_counts,
    record_tracking_workspace_status,
    tracking_target_workspace_status,
)
from src.common.eepose import ROBOT_BASE, SIM_LOCAL
from src.common.guidance import (
    normalize_guidance_frame,
    robot_base_to_sim_local_from_state,
    transform_guidance_pose,
)


TASKS = ("one_leg", "round_table", "lamp")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "reports"
    / "data"
    / "fresh36"
    / "annotation_noise_clean_train_tracking_saved8.json"
)


def _skill_type(skill_state: str) -> str:
    return str(skill_state).split("-")[-1]


def _observation_segments(
    observations: list[dict[str, Any]], task: str
) -> list[tuple[str, str, int]]:
    """Map stored generic skill segments back to the task's ordered semantic states."""
    expected_states = get_task_progress_labels(task, "skill_states")
    expected_types = [_skill_type(state) for state in expected_states]
    raw_segments: list[tuple[str, int]] = []
    for frame_idx, observation in enumerate(observations):
        skill = observation.get("skill")
        if skill is None:
            continue
        skill = str(skill)
        if raw_segments and raw_segments[-1][0] == skill:
            raw_segments[-1] = (skill, frame_idx)
        else:
            raw_segments.append((skill, frame_idx))

    segments: list[tuple[str, str, int]] = []
    current_index: int | None = None
    for skill, final_idx in raw_segments:
        candidates = [
            idx for idx, expected_type in enumerate(expected_types) if expected_type == skill
        ]
        if not candidates:
            raise ValueError(f"{task}: stored skill {skill!r} is not in progress schema")
        if current_index is None:
            selected = candidates[0]
        elif current_index + 1 in candidates:
            selected = current_index + 1
        else:
            selected = min(
                candidates,
                key=lambda idx: (
                    abs(idx - current_index),
                    0 if idx <= current_index else 1,
                    idx,
                ),
            )
        current_index = selected
        segments.append((expected_states[selected], skill, final_idx))
    return segments


def _episode_tracking_errors(
    trajectory: dict[str, Any],
    *,
    task: str,
    metric_type: str,
    workspace_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    observations = trajectory.get("observations") or []
    frame = trajectory.get("guidance_frame")
    if frame is None:
        sample_state = observations[0].get("robot_state", {}) if observations else {}
        frame = SIM_LOCAL if "ee_pos_sim" in sample_state else ROBOT_BASE
    frame = normalize_guidance_frame(frame)
    per_state: dict[str, dict[str, float]] = {}
    for semantic_state, _, final_idx in _observation_segments(observations, task):
        observation = observations[final_idx]
        target_pose = observation.get("guidance_pose")
        if target_pose is not None and frame == SIM_LOCAL:
            robot_to_sim = robot_base_to_sim_local_from_state(
                observation["robot_state"]
            )
            target_pose = transform_guidance_pose(
                target_pose, np.linalg.inv(robot_to_sim)
            )
        workspace_status = tracking_target_workspace_status(target_pose)
        record_tracking_workspace_status(workspace_counts, workspace_status)
        if workspace_status != "inside":
            continue
        error = _tracking_error_at_frame(
            observation.get("robot_state"),
            target_pose,
            metric_type=metric_type,
        )
        if error is None:
            continue
        selection_key = "pos_m" if metric_type == "position" else "total"
        previous = per_state.get(semantic_state)
        if previous is None or error[selection_key] < previous[selection_key]:
            per_state[semantic_state] = error
    return per_state


def _load_trajectory(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected dict trajectory, got {type(payload).__name__}")
    return payload


def _summarize_group(
    *,
    paths: list[Path],
    task: str,
    metric_type: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    skill_type_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    workspace_counts = new_tracking_workspace_counts()
    for path in paths:
        trajectory = _load_trajectory(path)
        episode_errors = _episode_tracking_errors(
            trajectory,
            task=task,
            metric_type=metric_type,
            workspace_counts=workspace_counts,
        )
        for semantic_state, error in episode_errors.items():
            state_records[semantic_state].append(error)
            skill_type_records[_skill_type(semantic_state)].append(error)
        del trajectory
        gc.collect()

    state_summary = build_tracking_error_summary(
        state_records,
        expected_labels=get_task_progress_labels(task, "skill_states"),
        metric_type=metric_type,
    )
    skill_type_summary = build_tracking_error_summary(
        skill_type_records,
        expected_labels=("push", "pick", "place", "insert", "screw"),
        metric_type=metric_type,
    )
    return (
        state_summary,
        skill_type_summary["by_skill"],
        build_tracking_workspace_filter_summary(workspace_counts),
    )


def recompute_saved_tracking(
    *,
    output_path: Path,
    task_group: str = "one_leg+round_table+lamp",
    randomness: str = "low",
    expected_rollouts: int = 8,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        noise_levels = (
            POINT_NOISE_LEVELS if condition.family == "point" else GRASP_NOISE_LEVELS
        )
        metric_type = "position" if condition.family == "point" else "pose"
        for noise in noise_levels:
            rollout_dirs = _rollout_group_dirs(
                task_group=task_group,
                randomness=randomness,
                condition=condition,
                noise=noise,
            )
            dirs_by_task = {path.parts[-7]: path for path in rollout_dirs}
            for task in TASKS:
                rollout_dir = dirs_by_task[task]
                paths = sorted(rollout_dir.glob("success/*.pkl")) + sorted(
                    rollout_dir.glob("failure/*.pkl")
                )
                paths = sorted(paths)
                if len(paths) != expected_rollouts:
                    raise RuntimeError(
                        f"{condition.condition_id}/{noise.noise_id}/{task}: "
                        f"found {len(paths)} pickles, expected {expected_rollouts} in {rollout_dir}"
                    )
                print(
                    f"[{len(groups) + 1:02d}/75] {condition.condition_id} "
                    f"{noise.noise_id} {task}: {len(paths)} rollouts",
                    flush=True,
                )
                tracking_error, by_skill_type, workspace_filter = _summarize_group(
                    paths=paths,
                    task=task,
                    metric_type=metric_type,
                )
                groups.append(
                    {
                        "condition": condition.condition,
                        "condition_id": condition.condition_id,
                        "family": condition.family,
                        "noise_id": noise.noise_id,
                        "noise_label": noise.noise_label,
                        "pos_std_mm": noise.pos_std_m * 1000.0,
                        "ori_std_deg": noise.ori_std_deg,
                        "task": task,
                        "rollout_count": len(paths),
                        "metric_type": metric_type,
                        "tracking_error": tracking_error,
                        "by_skill_type": by_skill_type,
                        "workspace_filter": workspace_filter,
                        "rollout_files": [str(path.relative_to(REPO_ROOT)) for path in paths],
                    }
                )

    payload = {
        "schema_version": 2,
        "source": "saved_rollout_pickles",
        "task_group": task_group,
        "randomness": randomness,
        "rollouts_per_task_setting": expected_rollouts,
        "target_pose": "displayed_noisy_guidance_pose",
        "workspace_filter": (
            "Only final-segment guidance targets inside the Panda robot-base point "
            "workspace contribute tracking error. Outside targets are counted and excluded."
        ),
        "position_metric": "3d_euclidean_robot_base_m",
        "orientation_metric": "so3_geodesic_deg",
        "total_metric": "pos_m/0.01 + ori_deg/5",
        "semantic_state_reconstruction": (
            "Stored pickles contain generic skill labels only. Consecutive skill segments "
            "are aligned to the task's ordered semantic-state schema; repeated inferred "
            "semantic states retain the minimum position or total error."
        ),
        "groups": groups,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-group", default="one_leg+round_table+lamp")
    parser.add_argument("--randomness", default="low")
    parser.add_argument("--expected-rollouts", type=int, default=8)
    args = parser.parse_args()
    recompute_saved_tracking(
        output_path=args.output,
        task_group=args.task_group,
        randomness=args.randomness,
        expected_rollouts=args.expected_rollouts,
    )


if __name__ == "__main__":
    main()
