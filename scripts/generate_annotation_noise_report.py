from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TASK_ORDER = ["one_leg", "round_table", "lamp"]
CONDITION_ORDER = [
    "rgbd+GP",
    "rgbd+colored GP",
    "rgbd+GP+skill",
    "rgbd+grasp-part",
    "rgbd+grasp-part-colored",
]
FAMILY_ORDER = ["point", "grasp-part"]
SKILL_TYPE_ORDER = ["push", "pick", "place", "insert", "screw"]
CONDITION_MARKERS = dict(zip(CONDITION_ORDER, ("o", "s", "^", "D", "P")))
CONDITION_LINESTYLES = dict(zip(CONDITION_ORDER, ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_path_part(value: str) -> str:
    safe = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", safe)
    return safe.strip("._") or "unknown"


def _dedupe_latest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("condition_id")), str(row.get("noise_id")))
        current = latest.get(key)
        if current is None or str(row.get("started_at", "")) > str(
            current.get("started_at", "")
        ):
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            FAMILY_ORDER.index(row["family"]),
            CONDITION_ORDER.index(row["condition"]),
            float(row["pos_std_mm"]),
            float(row["ori_std_deg"]),
        ),
    )


def _parse_iso_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _resolve_summary_path(run_row: dict[str, Any]) -> Path | None:
    summary_json = str(run_row.get("summary_json", "") or "").strip()
    if summary_json:
        summary_path = Path(summary_json)
        if summary_path.exists():
            return summary_path

    task_group = str(run_row.get("task_group", "") or "").strip()
    checkpoint_name = str(run_row.get("checkpoint_name", "") or "").strip()
    if not task_group or not checkpoint_name:
        return None

    log_dir = (
        Path("logs")
        / "evaluate_model"
        / _safe_path_part(task_group)
        / _safe_path_part(checkpoint_name)
    )
    if not log_dir.exists():
        return None

    candidates = sorted(log_dir.glob("*.json"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        return None

    started_ts = _parse_iso_seconds(run_row.get("started_at"))
    ended_ts = _parse_iso_seconds(run_row.get("ended_at"))
    if started_ts is None and ended_ts is None:
        return candidates[-1]

    matched: list[Path] = []
    for path in candidates:
        mtime = path.stat().st_mtime
        if started_ts is not None and mtime < started_ts - 120:
            continue
        if ended_ts is not None and mtime > ended_ts + 120:
            continue
        matched.append(path)
    if matched:
        return matched[-1]
    return candidates[-1]




def _skill_type_from_state(skill_state: str) -> str:
    token = str(skill_state).split("-")[-1]
    return token


def _weighted_tracking_overall(
    per_task: dict[str, Any], *, family: str
) -> dict[str, Any]:
    total_count = 0
    pos_sum = 0.0
    ori_sum = 0.0
    total_sum = 0.0
    metric_type = "position" if family == "point" else "pose"
    for task_payload in per_task.values():
        tracking = task_payload.get("tracking_error") or {}
        if family != "point":
            metric_type = tracking.get("metric_type", metric_type)
        overall = tracking.get("overall", {})
        count = int(overall.get("count", 0))
        if count <= 0:
            continue
        total_count += count
        pos_sum += float(overall.get("mean_pos_m", 0.0)) * count
        if metric_type == "pose":
            ori_sum += float(overall.get("mean_ori_deg", 0.0)) * count
            total_sum += float(overall.get("mean_total", 0.0)) * count
    if total_count <= 0:
        return {
            "metric_type": metric_type,
            "count": 0,
            "mean_pos_m": 0.0,
            "mean_ori_deg": None if metric_type == "position" else 0.0,
            "mean_total": None if metric_type == "position" else 0.0,
        }
    return {
        "metric_type": metric_type,
        "count": total_count,
        "mean_pos_m": pos_sum / total_count,
        "mean_ori_deg": None if metric_type == "position" else ori_sum / total_count,
        "mean_total": None if metric_type == "position" else total_sum / total_count,
    }


def _tracking_coverage_complete(task_payload: dict[str, Any]) -> bool:
    """Require explicit evidence that every rollout contributed tracking history."""
    expected = int(task_payload.get("n_rollouts", 0) or 0)
    tracking = task_payload.get("tracking_error") or {}
    workspace_filter = tracking.get("workspace_filter") or {}
    workspace_counts_match = int(
        workspace_filter.get("final_segment_count", -1)
    ) == sum(
        int(workspace_filter.get(key, 0))
        for key in (
            "included_segment_count",
            "excluded_outside_workspace_count",
            "missing_or_invalid_target_count",
        )
    )
    return (
        expected > 0
        and tracking.get("complete") is True
        and int(tracking.get("episode_count", -1)) == expected
        and int(tracking.get("expected_episode_count", expected)) == expected
        and int(tracking.get("incomplete_episode_count", -1)) == 0
        and workspace_filter.get("coordinate_frame") == "robot_base_m"
        and workspace_counts_match
    )


def _build_rows(
    manifest_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    overall_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    per_step_rows: list[dict[str, Any]] = []
    skill_type_rows: list[dict[str, Any]] = []

    for run_row in manifest_rows:
        if run_row.get("status") != "ok":
            continue
        summary_path = _resolve_summary_path(run_row)
        if summary_path is None or not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text())
        per_task = payload.get("per_task", {})
        task_tracking_complete = {
            task: _tracking_coverage_complete(per_task.get(task, {}))
            for task in TASK_ORDER
        }
        tracking_complete = all(task_tracking_complete.values())
        tracking_overall = _weighted_tracking_overall(
            per_task, family=str(run_row["family"])
        )
        overall_row = {
            "condition": run_row["condition"],
            "condition_id": run_row["condition_id"],
            "family": run_row["family"],
            "noise_id": run_row["noise_id"],
            "noise_label": run_row["noise_label"],
            "pos_std_mm": float(run_row["pos_std_mm"]),
            "ori_std_deg": float(run_row["ori_std_deg"]),
            "apply_to": run_row["apply_to"],
            "n_success": int(payload.get("n_success", 0)),
            "n_rollouts": int(payload.get("n_rollouts", 0)),
            "success_rate": float(payload.get("success_rate", 0.0) or 0.0),
            "track_pos_cm": float(tracking_overall["mean_pos_m"]) * 100.0,
            "tracking_metric_type": tracking_overall["metric_type"],
            "track_ori_deg": tracking_overall["mean_ori_deg"],
            "track_total": tracking_overall["mean_total"],
            "skill_state_count": int(tracking_overall["count"]),
            "tracking_complete": tracking_complete,
            "tracking_rollouts_per_task": int(
                run_row.get("tracking_rollouts_per_task", 0)
            ),
            "summary_json": str(summary_path),
        }
        for task in TASK_ORDER:
            task_payload = per_task.get(task, {})
            overall_row[f"{task}_success_rate"] = float(
                task_payload.get("success_rate", 0.0) or 0.0
            )
            task_tracking_payload = task_payload.get("tracking_error") or {}
            metric_type = task_tracking_payload.get(
                "metric_type", "position" if run_row["family"] == "point" else "pose"
            )
            task_tracking = task_tracking_payload.get("overall", {})
            task_rows.append(
                {
                    "condition": run_row["condition"],
                    "condition_id": run_row["condition_id"],
                    "family": run_row["family"],
                    "noise_id": run_row["noise_id"],
                    "noise_label": run_row["noise_label"],
                    "pos_std_mm": float(run_row["pos_std_mm"]),
                    "ori_std_deg": float(run_row["ori_std_deg"]),
                    "task": task,
                    "n_success": int(task_payload.get("n_success", 0)),
                    "n_rollouts": int(task_payload.get("n_rollouts", 0)),
                    "success_rate": float(task_payload.get("success_rate", 0.0) or 0.0),
                    "track_pos_cm": float(task_tracking.get("mean_pos_m", 0.0)) * 100.0,
                    "tracking_metric_type": metric_type,
                    "track_ori_deg": (
                        float(task_tracking.get("mean_ori_deg", 0.0))
                        if metric_type == "pose"
                        else None
                    ),
                    "track_total": (
                        float(task_tracking.get("mean_total", 0.0))
                        if metric_type == "pose"
                        else None
                    ),
                    "tracking_count": int(task_tracking.get("count", 0)),
                    "tracking_complete": task_tracking_complete[task],
                    "tracking_workspace_filter": task_tracking_payload.get(
                        "workspace_filter"
                    ),
                }
            )
        overall_rows.append(overall_row)

        skill_type_accumulator: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "count": 0,
                "pos_sum_cm": 0.0,
                "ori_sum_deg": 0.0,
                "total_sum": 0.0,
                "reached": 0,
                "completed": 0,
            }
        )

        for task in TASK_ORDER:
            task_payload = per_task.get(task, {})
            by_skill = (task_payload.get("tracking_error") or {}).get("by_skill", {})
            state_counts = task_payload.get("skill_state_counts", {})
            completion_counts = task_payload.get("skill_completion_counts", {})
            success_rates = task_payload.get("skill_success_rates", {})
            skill_states = list(state_counts)
            skill_states.extend(
                skill_state for skill_state in by_skill if skill_state not in state_counts
            )
            for skill_state in skill_states:
                stats = by_skill.get(skill_state, {})
                count = int(stats.get("count", 0))
                reached = int(state_counts.get(skill_state, 0))
                completed = int(completion_counts.get(skill_state, 0))
                if count <= 0 and reached <= 0:
                    continue
                per_step_rows.append(
                    {
                        "condition": run_row["condition"],
                        "condition_id": run_row["condition_id"],
                        "family": run_row["family"],
                        "noise_id": run_row["noise_id"],
                        "noise_label": run_row["noise_label"],
                        "pos_std_mm": float(run_row["pos_std_mm"]),
                        "ori_std_deg": float(run_row["ori_std_deg"]),
                        "task": task,
                        "skill_state": skill_state,
                        "skill_type": _skill_type_from_state(skill_state),
                        "reached_count": reached,
                        "completed_count": completed,
                        "skill_success_rate": float(success_rates.get(skill_state, 0.0) or 0.0),
                        "track_pos_cm": (
                            float(stats.get("mean_pos_m", 0.0)) * 100.0
                            if count > 0
                            else None
                        ),
                        "track_ori_deg": (
                            float(stats.get("mean_ori_deg", 0.0))
                            if count > 0 and run_row["family"] != "point"
                            else None
                        ),
                        "track_total": (
                            float(stats.get("mean_total", 0.0))
                            if count > 0 and run_row["family"] != "point"
                            else None
                        ),
                        "tracking_count": count,
                        "tracking_complete": task_tracking_complete[task],
                    }
                )
                skill_type = _skill_type_from_state(skill_state)
                bucket = skill_type_accumulator[skill_type]
                bucket["count"] += count
                if count > 0:
                    bucket["pos_sum_cm"] += float(stats.get("mean_pos_m", 0.0)) * 100.0 * count
                    if run_row["family"] != "point":
                        bucket["ori_sum_deg"] += float(stats.get("mean_ori_deg", 0.0)) * count
                        bucket["total_sum"] += float(stats.get("mean_total", 0.0)) * count
                bucket["reached"] += reached
                bucket["completed"] += completed

        for skill_type in SKILL_TYPE_ORDER:
            bucket = skill_type_accumulator.get(skill_type)
            if not bucket or (bucket["count"] <= 0 and bucket["reached"] <= 0):
                continue
            skill_type_rows.append(
                {
                    "condition": run_row["condition"],
                    "condition_id": run_row["condition_id"],
                    "family": run_row["family"],
                    "noise_id": run_row["noise_id"],
                    "noise_label": run_row["noise_label"],
                    "pos_std_mm": float(run_row["pos_std_mm"]),
                    "ori_std_deg": float(run_row["ori_std_deg"]),
                    "skill_type": skill_type,
                    "n_skill_states": int(bucket["count"]),
                    "reached_count": int(bucket["reached"]),
                    "completed_count": int(bucket["completed"]),
                    "skill_success_rate": (
                        float(bucket["completed"]) / float(bucket["reached"])
                        if bucket["reached"] > 0
                        else 0.0
                    ),
                    "track_pos_cm": (
                        float(bucket["pos_sum_cm"]) / float(bucket["count"])
                        if bucket["count"] > 0
                        else None
                    ),
                    "track_ori_deg": (
                        float(bucket["ori_sum_deg"]) / float(bucket["count"])
                        if bucket["count"] > 0 and run_row["family"] != "point"
                        else None
                    ),
                    "track_total": (
                        float(bucket["total_sum"]) / float(bucket["count"])
                        if bucket["count"] > 0 and run_row["family"] != "point"
                        else None
                    ),
                    "tracking_complete": tracking_complete,
                }
            )

    overall_rows.sort(
        key=lambda row: (
            FAMILY_ORDER.index(row["family"]),
            CONDITION_ORDER.index(row["condition"]),
            row["pos_std_mm"],
            row["ori_std_deg"],
        )
    )
    task_rows.sort(
        key=lambda row: (
            TASK_ORDER.index(row["task"]),
            CONDITION_ORDER.index(row["condition"]),
            row["pos_std_mm"],
            row["ori_std_deg"],
        )
    )
    per_step_rows.sort(
        key=lambda row: (
            FAMILY_ORDER.index(row["family"]),
            CONDITION_ORDER.index(row["condition"]),
            row["pos_std_mm"],
            TASK_ORDER.index(row["task"]),
            row["skill_state"],
        )
    )
    skill_type_rows.sort(
        key=lambda row: (
            FAMILY_ORDER.index(row["family"]),
            CONDITION_ORDER.index(row["condition"]),
            row["pos_std_mm"],
            SKILL_TYPE_ORDER.index(row["skill_type"]),
        )
    )
    return overall_rows, task_rows, per_step_rows, skill_type_rows


def _weighted_saved_tracking(
    summaries: list[dict[str, Any]], *, family: str
) -> dict[str, Any]:
    metric_type = "position" if family == "point" else "pose"
    count = sum(int(summary.get("count", 0)) for summary in summaries)
    if count <= 0:
        return {
            "count": 0,
            "mean_pos_m": None,
            "mean_ori_deg": None,
            "mean_total": None,
        }

    def weighted(key: str) -> float:
        return sum(
            float(summary.get(key, 0.0)) * int(summary.get("count", 0))
            for summary in summaries
        ) / count

    return {
        "count": count,
        "mean_pos_m": weighted("mean_pos_m"),
        "mean_ori_deg": weighted("mean_ori_deg") if metric_type == "pose" else None,
        "mean_total": weighted("mean_total") if metric_type == "pose" else None,
    }


def _set_tracking_fields(
    row: dict[str, Any],
    summary: dict[str, Any],
    *,
    family: str,
    source: str,
    rollout_count: int,
) -> None:
    count = int(summary.get("count", 0))
    mean_pos_m = summary.get("mean_pos_m")
    row.update(
        {
            "track_pos_cm": (
                float(mean_pos_m) * 100.0 if count > 0 and mean_pos_m is not None else None
            ),
            "track_ori_deg": (
                float(summary["mean_ori_deg"])
                if count > 0
                and family != "point"
                and summary.get("mean_ori_deg") is not None
                else None
            ),
            "track_total": (
                float(summary["mean_total"])
                if count > 0
                and family != "point"
                and summary.get("mean_total") is not None
                else None
            ),
            "tracking_count": count,
            "tracking_complete": True,
            "tracking_source": source,
            "tracking_rollouts_per_task": rollout_count,
        }
    )


def _invalidate_unfiltered_tracking(row: dict[str, Any]) -> None:
    # Keep legacy values available for the explicitly caveated Shuffle plots, but
    # do not publish them as workspace-filtered metrics in tables or analysis.
    row.update(
        {
            "tracking_complete": False,
            "tracking_source": "full_evaluator_36_pre_workspace_filter",
            "tracking_unavailable_reason": "workspace filtering unavailable",
        }
    )


def _apply_saved_tracking(
    *,
    overall_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    per_step_rows: list[dict[str, Any]],
    skill_type_rows: list[dict[str, Any]],
    saved_tracking_path: Path,
) -> dict[str, Any]:
    payload = json.loads(saved_tracking_path.read_text())
    groups = payload.get("groups") or []
    group_lookup = {
        (str(group["condition_id"]), str(group["noise_id"]), str(group["task"])): group
        for group in groups
    }
    expected_keys = {
        (str(row["condition_id"]), str(row["noise_id"]), str(row["task"]))
        for row in task_rows
        if row["noise_id"] != "shuffle"
    }
    missing = sorted(expected_keys - set(group_lookup))
    if missing:
        raise ValueError(
            f"Saved tracking input is missing {len(missing)} task settings: {missing[:3]}"
        )

    for row in task_rows:
        if row["noise_id"] == "shuffle":
            row["tracking_source"] = "full_evaluator_36"
            row["tracking_rollouts_per_task"] = int(row["n_rollouts"])
            if not row.get("tracking_workspace_filter"):
                _invalidate_unfiltered_tracking(row)
            continue
        group = group_lookup[
            (str(row["condition_id"]), str(row["noise_id"]), str(row["task"]))
        ]
        _set_tracking_fields(
            row,
            group["tracking_error"]["overall"],
            family=str(row["family"]),
            source="saved_rollouts_8",
            rollout_count=int(group["rollout_count"]),
        )
        row["tracking_workspace_filter"] = group.get("workspace_filter")

    for row in per_step_rows:
        if row["noise_id"] == "shuffle":
            row["tracking_source"] = "full_evaluator_36"
            row["tracking_rollouts_per_task"] = 36
            _invalidate_unfiltered_tracking(row)
            continue
        group = group_lookup[
            (str(row["condition_id"]), str(row["noise_id"]), str(row["task"]))
        ]
        summary = (group["tracking_error"].get("by_skill") or {}).get(
            str(row["skill_state"]), {"count": 0}
        )
        _set_tracking_fields(
            row,
            summary,
            family=str(row["family"]),
            source="saved_rollouts_8",
            rollout_count=int(group["rollout_count"]),
        )

    for row in skill_type_rows:
        if row["noise_id"] == "shuffle":
            row["tracking_source"] = "full_evaluator_36"
            row["tracking_rollouts_per_task"] = 36
            _invalidate_unfiltered_tracking(row)
            continue
        matching_groups = [
            group_lookup[(str(row["condition_id"]), str(row["noise_id"]), task)]
            for task in TASK_ORDER
        ]
        summaries = [
            (group.get("by_skill_type") or {}).get(str(row["skill_type"]), {"count": 0})
            for group in matching_groups
        ]
        _set_tracking_fields(
            row,
            _weighted_saved_tracking(summaries, family=str(row["family"])),
            family=str(row["family"]),
            source="saved_rollouts_8",
            rollout_count=int(payload["rollouts_per_task_setting"]),
        )
        row["n_skill_states"] = row.pop("tracking_count")

    task_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        task_lookup[(str(row["condition_id"]), str(row["noise_id"]))].append(row)
    for row in overall_rows:
        matching = task_lookup[(str(row["condition_id"]), str(row["noise_id"]))]
        if row["noise_id"] == "shuffle" and not all(
            bool(task_row.get("tracking_complete", False)) for task_row in matching
        ):
            _invalidate_unfiltered_tracking(row)
            continue
        summaries = [
            {
                "count": task_row["tracking_count"],
                "mean_pos_m": (
                    float(task_row["track_pos_cm"]) / 100.0
                    if task_row["track_pos_cm"] is not None
                    else 0.0
                ),
                "mean_ori_deg": task_row["track_ori_deg"] or 0.0,
                "mean_total": task_row["track_total"] or 0.0,
            }
            for task_row in matching
        ]
        weighted = _weighted_saved_tracking(summaries, family=str(row["family"]))
        _set_tracking_fields(
            row,
            weighted,
            family=str(row["family"]),
            source=(
                "full_evaluator_36"
                if row["noise_id"] == "shuffle"
                else "saved_rollouts_8"
            ),
            rollout_count=(
                36
                if row["noise_id"] == "shuffle"
                else int(payload["rollouts_per_task_setting"])
            ),
        )
        row["skill_state_count"] = row.pop("tracking_count")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    seen_fields = set()
    for row in rows:
        for field in row:
            if field in seen_fields:
                continue
            seen_fields.add(field)
            fieldnames.append(field)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _task_skill_type_rows(
    per_step_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_step_rows:
        grouped[(row["condition"], row["task"], row["noise_id"])].append(row)

    output: list[dict[str, Any]] = []
    for (condition, task, noise_id), rows in grouped.items():
        for skill_type in SKILL_TYPE_ORDER:
            candidates = [row for row in rows if row["skill_type"] == skill_type]
            if not candidates:
                continue
            reached = sum(int(row["reached_count"]) for row in candidates)
            completed = sum(int(row["completed_count"]) for row in candidates)
            tracking_count = sum(int(row["tracking_count"]) for row in candidates)

            def weighted_tracking(key: str) -> Optional[float]:
                if tracking_count <= 0:
                    return None
                return sum(
                    float(row[key]) * int(row["tracking_count"])
                    for row in candidates
                    if row[key] is not None
                ) / tracking_count

            first = candidates[0]
            output.append(
                {
                    "condition": condition,
                    "condition_id": first["condition_id"],
                    "family": first["family"],
                    "noise_id": noise_id,
                    "noise_label": first["noise_label"],
                    "pos_std_mm": float(first["pos_std_mm"]),
                    "ori_std_deg": float(first["ori_std_deg"]),
                    "task": task,
                    "skill_type": skill_type,
                    "reached_count": reached,
                    "completed_count": completed,
                    "skill_success_rate": completed / reached if reached else 0.0,
                    "track_pos_cm": weighted_tracking("track_pos_cm"),
                    "track_ori_deg": (
                        weighted_tracking("track_ori_deg")
                        if first["family"] != "point"
                        else None
                    ),
                    "track_total": (
                        weighted_tracking("track_total")
                        if first["family"] != "point"
                        else None
                    ),
                    "tracking_count": tracking_count,
                    "tracking_complete": all(
                        bool(row.get("tracking_complete", False))
                        for row in candidates
                    ),
                }
            )

    output.sort(
        key=lambda row: (
            CONDITION_ORDER.index(row["condition"]),
            TASK_ORDER.index(row["task"]),
            SKILL_TYPE_ORDER.index(row["skill_type"]),
            row["pos_std_mm"],
            row["ori_std_deg"],
        )
    )
    return output




def _wilson_interval(n_success: int, n_rollouts: int) -> tuple[float, float]:
    if n_rollouts <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = n_success / n_rollouts
    denominator = 1.0 + z * z / n_rollouts
    center = (p + z * z / (2.0 * n_rollouts)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / n_rollouts + z * z / (4.0 * n_rollouts**2))
        / denominator
    )
    return center - radius, center + radius




def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(str(row.get(key, "")) for key, _ in columns)
            + " |"
        )
    return "\n".join([header, sep, *body])




def _format_optional(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _result_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "--"
    success_rate = float(row.get("success_rate", row.get("skill_success_rate", 0.0)))
    n_success = int(row.get("n_success", row.get("completed_count", 0)))
    n_trials = int(row.get("n_rollouts", row.get("reached_count", 0)))
    if not bool(row.get("tracking_complete", False)):
        reason = row.get("tracking_unavailable_reason", "legacy partial")
        return (
            f"SR {100.0 * success_rate:.1f}% ({n_success}/{n_trials})"
            f"<br>Tracking unavailable ({reason})"
        )
    tracking_count = int(row.get("tracking_count", row.get("n_skill_states", 0)))
    tracking_fields = f"P {_format_optional(row['track_pos_cm'])}"
    if row.get("family") != "point" and row.get("tracking_metric_type") != "position":
        tracking_fields += (
            f" / O {_format_optional(row['track_ori_deg'])}"
            f" / T {_format_optional(row['track_total'])}"
        )
    return (
        f"SR {100.0 * success_rate:.1f}% ({n_success}/{n_trials})"
        f"<br>{tracking_fields}"
        f" (n={tracking_count})"
    )


def _noise_ids(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row["noise_id"]) for row in rows}
    ordered = [noise_id for noise_id in ("n0", "n1", "n2", "n3", "n4") if noise_id in present]
    if "shuffle" in present:
        ordered.append("shuffle")
    return ordered


def _matrix_table(
    rows: list[dict[str, Any]],
    *,
    group_fields: list[tuple[str, str]],
) -> str:
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = tuple(str(row[field]) for field, _ in group_fields)
        grouped[key][str(row["noise_id"])] = row

    output_rows = []
    for key in sorted(
        grouped,
        key=lambda values: tuple(
            CONDITION_ORDER.index(values[0]) if idx == 0 else values[idx]
            for idx in range(len(values))
        ),
    ):
        row = {field: value for (field, _), value in zip(group_fields, key)}
        for noise_id in _noise_ids(rows):
            row[noise_id] = _result_cell(grouped[key].get(noise_id))
        output_rows.append(row)

    return _markdown_table(
        output_rows,
        [*group_fields, *((noise_id, noise_id) for noise_id in _noise_ids(rows))],
    )


def _task_overall_table(task_rows: list[dict[str, Any]]) -> str:
    by_condition_task: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in task_rows:
        by_condition_task[(row["condition"], row["task"])][row["noise_id"]] = row

    formatted_rows = []
    for condition in CONDITION_ORDER:
        formatted = {"condition": condition}
        for task in TASK_ORDER:
            cells = []
            for noise_id in _noise_ids(task_rows):
                result = by_condition_task[(condition, task)].get(noise_id)
                cells.append(f"{noise_id}: {_result_cell(result)}")
            formatted[task] = "<br><br>".join(cells)
        formatted_rows.append(formatted)
    return _markdown_table(
        formatted_rows,
        [
            ("condition", "Condition"),
            ("one_leg", "one_leg"),
            ("round_table", "round_table"),
            ("lamp", "lamp"),
        ],
    )


def _categorical_endpoint_offsets(
    rows: list[dict[str, Any]], metric_key: str, *, spacing: float = 0.035
) -> dict[str, float]:
    """Offset only conditions whose n4-to-Shuffle segments exactly overlap."""
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[str(row["condition"])][str(row["noise_id"])] = row

    endpoint_groups: dict[tuple[float, float], list[str]] = defaultdict(list)
    offsets = {condition: 0.0 for condition in CONDITION_ORDER}
    for condition in CONDITION_ORDER:
        n4 = by_condition.get(condition, {}).get("n4")
        shuffled = by_condition.get(condition, {}).get("shuffle")
        if n4 is None or shuffled is None:
            continue
        n4_value = n4.get(metric_key)
        shuffled_value = shuffled.get(metric_key)
        if n4_value is None or shuffled_value is None:
            continue
        endpoint_groups[(float(n4_value), float(shuffled_value))].append(condition)

    for conditions in endpoint_groups.values():
        if len(conditions) <= 1:
            continue
        center = 0.5 * (len(conditions) - 1)
        for idx, condition in enumerate(conditions):
            offsets[condition] = (idx - center) * spacing
    return offsets


def _plot_success_vs_noise(
    task_rows: list[dict[str, Any]],
    figure_path: Path,
    *,
    title: str = "Clean-Train -> Noisy-Eval Success Curves",
    show_confidence: bool = False,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19, 4.8))
    grid = fig.add_gridspec(1, 6, width_ratios=[4, 1.4] * 3, wspace=0.08)
    shared_axis = None
    colors = {condition: f"C{idx}" for idx, condition in enumerate(CONDITION_ORDER)}
    legend_handles = []
    for task_idx, task in enumerate(TASK_ORDER):
        axis = fig.add_subplot(grid[0, 2 * task_idx], sharey=shared_axis)
        if shared_axis is None:
            shared_axis = axis
        shuffle_axis = fig.add_subplot(grid[0, 2 * task_idx + 1], sharey=shared_axis)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            if row["task"] != task:
                continue
            grouped[row["condition"]].append(row)
        category_offsets = _categorical_endpoint_offsets(
            [row for rows in grouped.values() for row in rows],
            "success_rate",
        )
        for condition in CONDITION_ORDER:
            marker = CONDITION_MARKERS[condition]
            linestyle = CONDITION_LINESTYLES[condition]
            rows = grouped.get(condition, [])
            regular = sorted(
                [row for row in rows if row["noise_id"] != "shuffle"],
                key=lambda item: item["pos_std_mm"],
            )
            x_values = [row["pos_std_mm"] for row in regular]
            y_values = [100.0 * row["success_rate"] for row in regular]
            if show_confidence:
                intervals = [
                    _wilson_interval(int(row["n_success"]), int(row["n_rollouts"]))
                    for row in regular
                ]
                y_error = [
                    [value - 100.0 * interval[0] for value, interval in zip(y_values, intervals)],
                    [100.0 * interval[1] - value for value, interval in zip(y_values, intervals)],
                ]
                line = axis.errorbar(
                    x_values,
                    y_values,
                    yerr=y_error,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=6,
                    markeredgewidth=1.2,
                    capsize=2,
                    label=condition,
                    color=colors[condition],
                )[0]
            else:
                line = axis.plot(
                    x_values,
                    y_values,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=6,
                    markeredgewidth=1.2,
                    label=condition,
                    color=colors[condition],
                )[0]
            if task_idx == 0:
                legend_handles.append(line)
            max_noise = next((row for row in rows if row["noise_id"] == "n4"), None)
            shuffled = next((row for row in rows if row["noise_id"] == "shuffle"), None)
            if max_noise is not None and shuffled is not None:
                offset = category_offsets[condition]
                endpoint_x = [offset, 1.0 + offset]
                endpoint_rows = [max_noise, shuffled]
                endpoint_values = [100.0 * row["success_rate"] for row in endpoint_rows]
                if show_confidence:
                    intervals = [
                        _wilson_interval(int(row["n_success"]), int(row["n_rollouts"]))
                        for row in endpoint_rows
                    ]
                    shuffle_axis.errorbar(
                        endpoint_x,
                        endpoint_values,
                        yerr=[
                            [value - 100.0 * interval[0] for value, interval in zip(endpoint_values, intervals)],
                            [100.0 * interval[1] - value for value, interval in zip(endpoint_values, intervals)],
                        ],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.5,
                        markersize=6,
                        markeredgewidth=1.2,
                        capsize=2,
                        color=colors[condition],
                    )
                else:
                    shuffle_axis.plot(
                        endpoint_x,
                        endpoint_values,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.5,
                        markersize=6,
                        markeredgewidth=1.2,
                        color=colors[condition],
                    )
        axis.set_xlabel("Per-axis Position Noise Std (mm)")
        axis.set_title(task)
        axis.grid(True, alpha=0.3)
        shuffle_axis.set_facecolor("#f4f4f4")
        shuffle_axis.set_xticks([0, 1], ["n4", "Shuffle"], rotation=35, ha="right")
        shuffle_axis.set_xlim(-0.25, 1.25)
        shuffle_axis.grid(True, alpha=0.3)
        shuffle_axis.tick_params(labelleft=False)
    shared_axis.set_ylabel("Success Rate (%)")
    fig.legend(legend_handles, CONDITION_ORDER, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(title)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _plot_tracking_vs_noise(task_rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19, 13))
    grid = fig.add_gridspec(
        3,
        2 * len(TASK_ORDER),
        width_ratios=[4, 1.4] * len(TASK_ORDER),
        hspace=0.22,
        wspace=0.08,
    )
    colors = {condition: f"C{idx}" for idx, condition in enumerate(CONDITION_ORDER)}
    metrics = [
        ("track_pos_cm", "Position Error (cm)"),
        ("track_ori_deg", "Orientation Error (deg)"),
        ("track_total", "Total Error"),
    ]
    for metric_idx, (metric_key, metric_label) in enumerate(metrics):
        shared_axis = None
        for task_idx, task in enumerate(TASK_ORDER):
            axis = fig.add_subplot(
                grid[metric_idx, 2 * task_idx], sharey=shared_axis
            )
            if shared_axis is None:
                shared_axis = axis
            shuffle_axis = fig.add_subplot(
                grid[metric_idx, 2 * task_idx + 1], sharey=shared_axis
            )
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in task_rows:
                if row["task"] == task:
                    grouped[row["condition"]].append(row)
            category_offsets = _categorical_endpoint_offsets(
                [row for rows in grouped.values() for row in rows],
                metric_key,
            )
            for condition in CONDITION_ORDER:
                if metric_key != "track_pos_cm" and condition in CONDITION_ORDER[:3]:
                    continue
                all_rows = grouped.get(condition, [])
                marker = CONDITION_MARKERS[condition]
                linestyle = CONDITION_LINESTYLES[condition]
                rows = sorted(
                    [
                        row
                        for row in all_rows
                        if row["noise_id"] != "shuffle"
                        and row.get(metric_key) is not None
                    ],
                    key=lambda item: item["pos_std_mm"],
                )
                if rows:
                    axis.plot(
                        [row["pos_std_mm"] for row in rows],
                        [row[metric_key] for row in rows],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=2,
                        markersize=6,
                        markeredgewidth=1.2,
                        label=condition,
                        color=colors[condition],
                    )
                n4 = next(
                    (row for row in all_rows if row["noise_id"] == "n4"), None
                )
                shuffled = next(
                    (row for row in all_rows if row["noise_id"] == "shuffle"), None
                )
                if (
                    n4 is not None
                    and shuffled is not None
                    and n4.get(metric_key) is not None
                    and shuffled.get(metric_key) is not None
                ):
                    offset = category_offsets[condition]
                    shuffle_axis.plot(
                        [offset, 1.0 + offset],
                        [float(n4[metric_key]), float(shuffled[metric_key])],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.5,
                        markersize=6,
                        markeredgewidth=1.2,
                        color=colors[condition],
                    )
            if metric_idx == 0:
                axis.set_title(task)
            if metric_idx == len(metrics) - 1:
                axis.set_xlabel("Per-axis Position Noise Std (mm)")
            if task_idx == 0:
                axis.set_ylabel(metric_label)
            axis.set_xticks([0, 3, 6, 12, 24])
            axis.grid(True, alpha=0.3)
            if task_idx > 0:
                axis.tick_params(labelleft=False)
            shuffle_axis.set_facecolor("#f4f4f4")
            shuffle_axis.set_xticks(
                [0, 1], ["n4", "Shuffle"], rotation=35, ha="right"
            )
            shuffle_axis.set_xlim(-0.25, 1.25)
            shuffle_axis.grid(True, alpha=0.3)
            shuffle_axis.tick_params(labelleft=False)
            if metric_idx == 0:
                shuffle_axis.set_title("Categorical", fontsize=8)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[condition],
            marker=CONDITION_MARKERS[condition],
            linestyle=CONDITION_LINESTYLES[condition],
            linewidth=2,
        )
        for condition in CONDITION_ORDER
    ]
    fig.legend(
        legend_handles,
        CONDITION_ORDER,
        loc="lower center",
        ncol=5,
        fontsize=8,
    )
    fig.suptitle(
        "Clean-Train -> Noisy-Eval Tracking Error Curves\n"
        "n0-n4: workspace-filtered saved-8; Shuffle: legacy unfiltered full-36"
    )
    fig.savefig(figure_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)




def _plot_task_skill_condition_grids(
    task_skill_rows: list[dict[str, Any]],
    figures_dir: Path,
    *,
    include_tracking: bool,
) -> list[tuple[str, str, Path]]:
    """Plot 5 skill rows by 3 task columns, with conditions as curves."""
    metric_specs = [
        ("skill_success_rate", "Success Rate (%)", False),
        ("track_pos_cm", "Position Error (cm)", False),
        ("track_ori_deg", "Orientation Error (deg)", True),
        ("track_total", "Total Error", True),
    ]
    if not include_tracking:
        metric_specs = metric_specs[:1]
    colors = {condition: f"C{idx}" for idx, condition in enumerate(CONDITION_ORDER)}
    generated: list[tuple[str, str, Path]] = []
    figures_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_label, pose_only in metric_specs:
        figure_path = figures_dir / f"annotation_noise_clean_train_skill_{metric_key}.png"
        fig = plt.figure(figsize=(20, 22))
        grid = fig.add_gridspec(
            len(SKILL_TYPE_ORDER),
            2 * len(TASK_ORDER),
            width_ratios=[4, 1.15] * len(TASK_ORDER),
            hspace=0.28,
            wspace=0.10,
        )

        for skill_idx, skill_type in enumerate(SKILL_TYPE_ORDER):
            for task_idx, task in enumerate(TASK_ORDER):
                axis = fig.add_subplot(grid[skill_idx, 2 * task_idx])
                shuffle_axis = fig.add_subplot(
                    grid[skill_idx, 2 * task_idx + 1], sharey=axis
                )
                candidates = [
                    row
                    for row in task_skill_rows
                    if row["task"] == task and row["skill_type"] == skill_type
                ]
                category_offsets = _categorical_endpoint_offsets(candidates, metric_key)

                for condition in CONDITION_ORDER:
                    if pose_only and condition in CONDITION_ORDER[:3]:
                        continue
                    rows = [
                        row for row in candidates if row["condition"] == condition
                    ]
                    marker = CONDITION_MARKERS[condition]
                    linestyle = CONDITION_LINESTYLES[condition]
                    regular = sorted(
                        [
                            row
                            for row in rows
                            if row["noise_id"] != "shuffle"
                            and row[metric_key] is not None
                        ],
                        key=lambda row: row["pos_std_mm"],
                    )
                    if not regular:
                        continue
                    y_values = [float(row[metric_key]) for row in regular]
                    if metric_key == "skill_success_rate":
                        y_values = [100.0 * value for value in y_values]
                    axis.plot(
                        [float(row["pos_std_mm"]) for row in regular],
                        y_values,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.8,
                        markersize=4.5,
                        markeredgewidth=1.0,
                        color=colors[condition],
                    )

                    n4 = next((row for row in rows if row["noise_id"] == "n4"), None)
                    shuffled = next(
                        (row for row in rows if row["noise_id"] == "shuffle"), None
                    )
                    if (
                        n4 is not None
                        and shuffled is not None
                        and n4[metric_key] is not None
                        and shuffled[metric_key] is not None
                    ):
                        offset = category_offsets[condition]
                        endpoints = [float(n4[metric_key]), float(shuffled[metric_key])]
                        if metric_key == "skill_success_rate":
                            endpoints = [100.0 * value for value in endpoints]
                        shuffle_axis.plot(
                            [offset, 1.0 + offset],
                            endpoints,
                            marker=marker,
                            linestyle=linestyle,
                            linewidth=1.3,
                            markersize=4.5,
                            markeredgewidth=1.0,
                            color=colors[condition],
                        )

                if metric_key == "skill_success_rate":
                    axis.set_ylim(0.0, 105.0)
                axis.set_xticks([0, 3, 6, 12, 24])
                axis.grid(True, alpha=0.3)
                shuffle_axis.set_facecolor("#f4f4f4")
                shuffle_axis.set_xticks(
                    [0, 1], ["n4", "Shuffle"], rotation=35, ha="right"
                )
                shuffle_axis.set_xlim(-0.25, 1.25)
                shuffle_axis.grid(True, alpha=0.3)
                shuffle_axis.tick_params(labelleft=False)
                if skill_idx == 0:
                    axis.set_title(task)
                    shuffle_axis.set_title("Categorical", fontsize=8)
                if skill_idx == len(SKILL_TYPE_ORDER) - 1:
                    axis.set_xlabel("Position Noise Std/axis (mm)")
                if task_idx == 0:
                    axis.set_ylabel(f"{skill_type}\n{metric_label}")

        legend_conditions = (
            CONDITION_ORDER[3:] if pose_only else CONDITION_ORDER
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=colors[condition],
                marker=CONDITION_MARKERS[condition],
                linestyle=CONDITION_LINESTYLES[condition],
                linewidth=2,
            )
            for condition in legend_conditions
        ]
        fig.legend(
            legend_handles,
            legend_conditions,
            loc="lower center",
            ncol=len(legend_conditions),
            fontsize=9,
        )
        sample_note = (
            "\nn0-n4 tracking: 8 saved rollouts/task/setting; "
            "Shuffle: legacy unfiltered full-36"
            if metric_key != "skill_success_rate"
            else ""
        )
        fig.suptitle(
            f"Task-Skill Average: {metric_label} (5 Skills x 3 Tasks){sample_note}"
        )
        fig.savefig(figure_path, dpi=180, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        generated.append((metric_key, metric_label, figure_path))

    return generated


def _best_tolerance_rows(overall_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overall_rows:
        grouped[row["condition"]].append(row)
    rows = []
    for condition, candidates in grouped.items():
        valid = [
            row
            for row in candidates
            if row["noise_id"] != "shuffle" and row["success_rate"] >= threshold
        ]
        if not valid:
            rows.append(
                {
                    "condition": condition,
                    "success_threshold": f"{100.0 * threshold:.0f}%",
                    "max_pos_std_mm": "none",
                    "max_ori_std_deg": "none",
                }
            )
            continue
        best = max(valid, key=lambda row: (row["pos_std_mm"], row["ori_std_deg"]))
        rows.append(
            {
                "condition": condition,
                "success_threshold": f"{100.0 * threshold:.0f}%",
                "max_pos_std_mm": f"{best['pos_std_mm']:.0f}",
                "max_ori_std_deg": f"{best['ori_std_deg']:.1f}",
            }
        )
    rows.sort(key=lambda row: CONDITION_ORDER.index(row["condition"]))
    return rows


def _endpoint_comparison_rows(overall_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in overall_rows:
        grouped[row["condition"]][row["noise_id"]] = row

    comparison_rows = []
    for condition in CONDITION_ORDER:
        n0 = grouped.get(condition, {}).get("n0")
        n4 = grouped.get(condition, {}).get("n4")
        if n0 is None or n4 is None:
            continue
        shuffled = grouped.get(condition, {}).get("shuffle")
        success_delta_pp = 100.0 * (n4["success_rate"] - n0["success_rate"])
        tracking_key = "track_pos_cm" if n0["family"] == "point" else "track_total"
        tracking_delta = n4[tracking_key] - n0[tracking_key]
        tracking_name = "Position (cm)" if n0["family"] == "point" else "Total"
        shuffle_success = "pending"
        shuffle_success_delta = "pending"
        shuffle_tracking = "pending"
        shuffle_tracking_delta = "pending"
        if shuffled is not None:
            shuffle_success = f"{100.0 * shuffled['success_rate']:.1f}%"
            shuffle_success_delta = (
                f"{100.0 * (shuffled['success_rate'] - n4['success_rate']):+.1f} pp"
            )
            if bool(shuffled.get("tracking_complete", False)):
                shuffle_tracking = f"{shuffled[tracking_key]:.2f}"
                shuffle_tracking_delta = (
                    f"{shuffled[tracking_key] - n4[tracking_key]:+.2f}"
                )
            else:
                shuffle_tracking = "unavailable"
                shuffle_tracking_delta = "unavailable"
        comparison_rows.append(
            {
                "condition": condition,
                "n0_success": f"{100.0 * n0['success_rate']:.1f}%",
                "n4_success": f"{100.0 * n4['success_rate']:.1f}%",
                "success_delta": f"{success_delta_pp:+.1f} pp",
                "tracking_metric": tracking_name,
                "n0_tracking": f"{n0[tracking_key]:.2f}",
                "n4_tracking": f"{n4[tracking_key]:.2f}",
                "tracking_delta": f"{tracking_delta:+.2f}",
                "shuffle_success": shuffle_success,
                "shuffle_success_delta": shuffle_success_delta,
                "shuffle_tracking": shuffle_tracking,
                "shuffle_tracking_delta": shuffle_tracking_delta,
            }
        )
    return comparison_rows


def _shuffle_summary_rows(overall_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in overall_rows:
        grouped[str(row["condition"])][str(row["noise_id"])] = row

    rows: list[dict[str, str]] = []
    for condition in CONDITION_ORDER:
        n0 = grouped.get(condition, {}).get("n0")
        shuffled = grouped.get(condition, {}).get("shuffle")
        if n0 is None or shuffled is None:
            continue
        tracking_valid = bool(shuffled.get("tracking_complete", False))
        family = str(shuffled["family"])
        rows.append(
            {
                "condition": condition,
                "n0_success": f"{100.0 * n0['success_rate']:.1f}%",
                "shuffle_success": f"{100.0 * shuffled['success_rate']:.1f}%",
                "success_delta": (
                    f"{100.0 * (shuffled['success_rate'] - n0['success_rate']):+.1f} pp"
                ),
                "track_pos_cm": (
                    f"{float(shuffled['track_pos_cm']):.2f}" if tracking_valid else "unavailable"
                ),
                "track_ori_deg": (
                    f"{float(shuffled['track_ori_deg']):.2f}"
                    if tracking_valid and family != "point"
                    else "N/A"
                ),
                "track_total": (
                    f"{float(shuffled['track_total']):.2f}"
                    if tracking_valid and family != "point"
                    else "N/A"
                ),
                "tracking_count": (
                    str(int(shuffled["skill_state_count"]))
                    if tracking_valid
                    else "0"
                ),
            }
        )
    return rows


def _shuffle_version_comparison_rows(
    current_rows: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    current = {
        (str(row["condition"]), str(row["noise_id"])): row for row in current_rows
    }
    previous = {
        (str(row["condition"]), str(row["noise_id"])): row for row in previous_rows
    }
    rows: list[dict[str, str]] = []
    totals = {"n0_success": 0, "n0_rollouts": 0, "old_success": 0, "old_rollouts": 0,
              "new_success": 0, "new_rollouts": 0}
    for condition in CONDITION_ORDER:
        n0 = current.get((condition, "n0"))
        old = previous.get((condition, "shuffle"))
        new = current.get((condition, "shuffle"))
        if n0 is None or old is None or new is None:
            continue
        for prefix, value in (("n0", n0), ("old", old), ("new", new)):
            totals[f"{prefix}_success"] += int(value["n_success"])
            totals[f"{prefix}_rollouts"] += int(value["n_rollouts"])
        rows.append(
            {
                "condition": condition,
                "n0": f"{100.0 * n0['success_rate']:.1f}%",
                "old_shuffle": f"{100.0 * old['success_rate']:.1f}%",
                "old_delta": f"{100.0 * (old['success_rate'] - n0['success_rate']):+.1f} pp",
                "new_shuffle": f"{100.0 * new['success_rate']:.1f}%",
                "new_delta": f"{100.0 * (new['success_rate'] - n0['success_rate']):+.1f} pp",
            }
        )
    if rows:
        rates = {
            prefix: totals[f"{prefix}_success"] / totals[f"{prefix}_rollouts"]
            for prefix in ("n0", "old", "new")
        }
        rows.append(
            {
                "condition": "All conditions",
                "n0": f"{100.0 * rates['n0']:.1f}%",
                "old_shuffle": f"{100.0 * rates['old']:.1f}%",
                "old_delta": f"{100.0 * (rates['old'] - rates['n0']):+.1f} pp",
                "new_shuffle": f"{100.0 * rates['new']:.1f}%",
                "new_delta": f"{100.0 * (rates['new'] - rates['n0']):+.1f} pp",
            }
        )
    return rows


def _shuffle_conclusion_lines(overall_rows: list[dict[str, Any]]) -> list[str]:
    grouped = {
        (str(row["condition"]), str(row["noise_id"])): row for row in overall_rows
    }
    pairs = [
        (grouped.get((condition, "n4")), grouped.get((condition, "shuffle")))
        for condition in CONDITION_ORDER
    ]
    pairs = [(n4, shuffled) for n4, shuffled in pairs if n4 and shuffled]
    if not pairs:
        return ["- Shuffle 数据尚未完成。"]
    n4_success = sum(int(n4["n_success"]) for n4, _ in pairs)
    n4_rollouts = sum(int(n4["n_rollouts"]) for n4, _ in pairs)
    shuffle_success = sum(int(shuffled["n_success"]) for _, shuffled in pairs)
    shuffle_rollouts = sum(int(shuffled["n_rollouts"]) for _, shuffled in pairs)
    n4_rate = n4_success / n4_rollouts
    shuffle_rate = shuffle_success / shuffle_rollouts
    deltas = {
        str(n4["condition"]): 100.0
        * (float(shuffled["success_rate"]) - float(n4["success_rate"]))
        for n4, shuffled in pairs
    }
    declining = [condition for condition, delta in deltas.items() if delta < 0.0]
    tracking_valid = all(
        bool(shuffled.get("tracking_complete", False)) for _, shuffled in pairs
    )
    return [
        (
            f"- 五个 condition 合计：n4 为 `{n4_success}/{n4_rollouts} = {100.0 * n4_rate:.1f}%`，"
            f"强 Shuffle 为 `{shuffle_success}/{shuffle_rollouts} = {100.0 * shuffle_rate:.1f}%`，"
            f"变化 `{100.0 * (shuffle_rate - n4_rate):+.1f} pp`。"
        ),
        (
            f"- `{len(declining)}/{len(deltas)}` 个 condition 从 n4 到 Shuffle 下降："
            + "、".join(
                f"{condition} `{deltas[condition]:+.1f} pp`" for condition in declining
            )
            + f"；只有 rgbd+GP+skill 回升 `{deltas['rgbd+GP+skill']:+.1f} pp`。"
        ),
        (
            "- 图像与深度不变时，破坏 guidance 的 semantic-state 对应关系会在大多数 condition "
            "上降低成功率，说明 policy 并非只依赖 RGBD；语义正确的 guidance 仍提供有效信息。"
            "Shuffle 后仍保留约一半成功率，则说明模型也能使用视觉或低维输入 fallback，guidance "
            "不是唯一的信息源。"
        ),
        *(
            []
            if tracking_valid
            else [
                "- 旧 Shuffle full-36 summary 没有 workspace exclusion 明细，且受完成态坐标早退 bug "
                "影响，因此其 tracking error 已撤下；仅保留不依赖 tracking 的成功率结果。"
            ]
        ),
        (
            "- task-level 正负波动仍较大，且 clean 与 Shuffle rollout 未按相同 reset seed 配对；"
            "这些波动不能解释为错误 guidance 带来的真实提升。"
        ),
    ]


def _linear_response(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sum_xx = sum((value - mean_x) ** 2 for value in xs)
    sum_yy = sum((value - mean_y) ** 2 for value in ys)
    sum_xy = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys)
    )
    slope = sum_xy / sum_xx if sum_xx else 0.0
    correlation = sum_xy / math.sqrt(sum_xx * sum_yy) if sum_xx and sum_yy else 0.0
    return slope, correlation, correlation**2


def _tracking_response_rows(
    overall_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overall_rows:
        if row["noise_id"] != "shuffle":
            grouped[str(row["condition"])].append(row)

    response_rows: list[dict[str, str]] = []
    for condition in CONDITION_ORDER:
        candidates = sorted(
            grouped[condition], key=lambda row: int(str(row["noise_id"])[1:])
        )
        family = str(candidates[0]["family"])
        metric_specs = [
            (
                "Position",
                "track_pos_cm",
                [float(row["pos_std_mm"]) / 10.0 for row in candidates],
                "cm/cm",
            )
        ]
        if family != "point":
            metric_specs.extend(
                [
                    (
                        "Orientation",
                        "track_ori_deg",
                        [float(row["ori_std_deg"]) for row in candidates],
                        "deg/deg",
                    ),
                    (
                        "Total",
                        "track_total",
                        [
                            float(row["pos_std_mm"]) / 10.0
                            + float(row["ori_std_deg"]) / 5.0
                            for row in candidates
                        ],
                        "total/unit",
                    ),
                ]
            )

        for metric, metric_key, noise_values, slope_unit in metric_specs:
            tracking_values = [float(row[metric_key]) for row in candidates]
            slope, correlation, r_squared = _linear_response(
                noise_values, tracking_values
            )
            response_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "n0": f"{tracking_values[0]:.2f}",
                    "n4": f"{tracking_values[-1]:.2f}",
                    "delta": f"{tracking_values[-1] - tracking_values[0]:+.2f}",
                    "slope": f"{slope:.3f} {slope_unit}",
                    "pearson_r": f"{correlation:.3f}",
                    "r_squared": f"{r_squared:.3f}",
                }
            )
    return response_rows


def _tracking_interpretation_lines(
    overall_rows: list[dict[str, Any]],
) -> list[str]:
    grouped = {
        (str(row["condition"]), str(row["noise_id"])): row for row in overall_rows
    }
    numeric_position_deltas = []
    numeric_position_correlations = []
    orientation_deltas = []
    for condition in CONDITION_ORDER:
        numeric = [
            grouped[(condition, noise_id)]
            for noise_id in ("n0", "n1", "n2", "n3", "n4")
        ]
        position_values = [float(row["track_pos_cm"]) for row in numeric]
        _, correlation, _ = _linear_response(
            [float(row["pos_std_mm"]) / 10.0 for row in numeric],
            position_values,
        )
        numeric_position_deltas.append(position_values[-1] - position_values[0])
        numeric_position_correlations.append((correlation, condition))
        if str(numeric[0]["family"]) != "point":
            orientation_deltas.append(
                float(numeric[-1]["track_ori_deg"])
                - float(numeric[0]["track_ori_deg"])
            )

    weakest_correlation, weakest_condition = min(numeric_position_correlations)
    strongest_correlation, strongest_condition = max(numeric_position_correlations)
    return [
        (
            f"- 从 n0 到 n4，五个 condition 的平均 position tracking error 均增加 "
            f"`{min(numeric_position_deltas):.2f}-{max(numeric_position_deltas):.2f} cm`；"
            f"grasp-part 的 orientation error 增加 "
            f"`{min(orientation_deltas):.2f}-{max(orientation_deltas):.2f} deg`。"
        ),
        (
            f"- Position error 与 position noise 的相关性均为正；最强为 "
            f"`{strongest_condition}` (`r={strongest_correlation:.3f}`)，最弱为 "
            f"`{weakest_condition}` (`r={weakest_correlation:.3f}`)。弱相关曲线主要受 "
            "saved-8 小样本及 task/skill 极值影响，不宜解释为真实非单调响应。"
        ),
        (
            "- 这里的 target 是 noisy guidance，而不是真实 semantic target。若 policy 完全跟随 "
            "noisy guidance，tracking error 应保持低且近似平坦；tracking error 随噪声增加，"
            "同时成功率没有同步下降，说明动作仍受 RGBD、skill 或 clean semantic prior 约束，"
            "只部分跟随甚至主动拒绝偏移 guidance。"
        ),
        (
            "- 旧 Shuffle tracking 不满足 workspace 过滤要求，仅作为曲线中的 legacy 参考端点，"
            "不进入 tracking 表格、拟合或正式结论；需要用修复后的 evaluator 重跑后才能正式比较 "
            "n4 与 Shuffle tracking。"
        ),
    ]


def _numeric_success_stability_rows(
    task_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    numeric_ids = ("n0", "n1", "n2", "n3", "n4")
    for condition in CONDITION_ORDER:
        condition_rows = [row for row in task_rows if row["condition"] == condition]

        def task_rate(task: str, noise_id: str) -> float:
            row = next(
                row
                for row in condition_rows
                if row["task"] == task and row["noise_id"] == noise_id
            )
            return 100.0 * float(row["success_rate"])

        overall_values = []
        for noise_id in numeric_ids:
            candidates = [row for row in condition_rows if row["noise_id"] == noise_id]
            overall_values.append(
                100.0
                * sum(int(row["n_success"]) for row in candidates)
                / sum(int(row["n_rollouts"]) for row in candidates)
            )
        mean = sum(overall_values) / len(overall_values)
        std = math.sqrt(
            sum((value - mean) ** 2 for value in overall_values)
            / len(overall_values)
        )
        task_ranges = {
            task: max(task_rate(task, noise_id) for noise_id in numeric_ids)
            - min(task_rate(task, noise_id) for noise_id in numeric_ids)
            for task in TASK_ORDER
        }
        n4 = sum(
            int(row["n_success"])
            for row in condition_rows
            if row["noise_id"] == "n4"
        ) / sum(
            int(row["n_rollouts"])
            for row in condition_rows
            if row["noise_id"] == "n4"
        )
        shuffled = sum(
            int(row["n_success"])
            for row in condition_rows
            if row["noise_id"] == "shuffle"
        ) / sum(
            int(row["n_rollouts"])
            for row in condition_rows
            if row["noise_id"] == "shuffle"
        )
        rows.append(
            {
                "condition": condition,
                "overall_range": f"{max(overall_values) - min(overall_values):.1f} pp",
                "overall_std": f"{std:.1f} pp",
                "one_leg_range": f"{task_ranges['one_leg']:.1f} pp",
                "round_table_range": f"{task_ranges['round_table']:.1f} pp",
                "lamp_range": f"{task_ranges['lamp']:.1f} pp",
                "n4_to_shuffle": f"{100.0 * (shuffled - n4):+.1f} pp",
            }
        )
    return rows


def _tracking_workspace_exclusion_rows(
    saved_tracking_payload: dict[str, Any],
) -> list[dict[str, str]]:
    """Aggregate n0-n4 saved-rollout workspace exclusions by condition and task."""
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"inside": 0, "outside": 0, "invalid": 0}
    )
    condition_names: dict[str, str] = {}
    for group in saved_tracking_payload.get("groups") or []:
        condition_id = str(group["condition_id"])
        task = str(group["task"])
        condition_names[condition_id] = str(group["condition"])
        workspace_filter = group.get("workspace_filter") or {}
        bucket = grouped[(condition_id, task)]
        bucket["inside"] += int(
            workspace_filter.get("included_segment_count", 0)
        )
        bucket["outside"] += int(
            workspace_filter.get("excluded_outside_workspace_count", 0)
        )
        bucket["invalid"] += int(
            workspace_filter.get("missing_or_invalid_target_count", 0)
        )

    def format_cell(condition_id: str, task: str) -> str:
        counts = grouped[(condition_id, task)]
        finite_targets = counts["inside"] + counts["outside"]
        percentage = (
            100.0 * counts["outside"] / finite_targets if finite_targets else 0.0
        )
        return f"{counts['outside']}/{finite_targets} ({percentage:.2f}%)"

    rows = []
    for condition in CONDITION_ORDER:
        condition_id = next(
            (
                condition_id
                for condition_id, name in condition_names.items()
                if name == condition
            ),
            None,
        )
        if condition_id is None:
            continue
        rows.append(
            {
                "condition": condition,
                **{
                    task: format_cell(condition_id, task)
                    for task in TASK_ORDER
                },
                "invalid": str(
                    sum(grouped[(condition_id, task)]["invalid"] for task in TASK_ORDER)
                ),
            }
        )
    return rows


def _main_conclusion_lines(
    overall_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    per_step_rows: list[dict[str, Any]],
) -> list[str]:
    grouped = {
        (str(row["condition"]), str(row["noise_id"])): row for row in overall_rows
    }

    def task_rate(condition: str, task: str, noise_id: str) -> float:
        row = next(
            row
            for row in task_rows
            if row["condition"] == condition
            and row["task"] == task
            and row["noise_id"] == noise_id
        )
        return 100.0 * float(row["success_rate"])

    gp_skill_n4 = grouped[("rgbd+GP+skill", "n4")]
    gp_skill_shuffle = grouped[("rgbd+GP+skill", "shuffle")]
    n4_success = sum(
        int(grouped[(condition, "n4")]["n_success"]) for condition in CONDITION_ORDER
    )
    n4_rollouts = sum(
        int(grouped[(condition, "n4")]["n_rollouts"]) for condition in CONDITION_ORDER
    )
    shuffle_success = sum(
        int(grouped[(condition, "shuffle")]["n_success"])
        for condition in CONDITION_ORDER
    )
    shuffle_rollouts = sum(
        int(grouped[(condition, "shuffle")]["n_rollouts"])
        for condition in CONDITION_ORDER
    )
    numeric_position_correlations = {}
    for condition in ("rgbd+GP", "rgbd+colored GP"):
        rows = [
            grouped[(condition, noise_id)]
            for noise_id in ("n0", "n1", "n2", "n3", "n4")
        ]
        _, correlation, _ = _linear_response(
            [float(row["pos_std_mm"]) / 10.0 for row in rows],
            [float(row["track_pos_cm"]) for row in rows],
        )
        numeric_position_correlations[condition] = correlation

    gp_round_table = {
        str(row["noise_id"]): row
        for row in task_rows
        if row["condition"] == "rgbd+GP"
        and row["task"] == "round_table"
        and row["noise_id"] in {"n0", "n1", "n2", "n3", "n4"}
    }
    gp_first_screw = {
        str(row["noise_id"]): row
        for row in per_step_rows
        if row["condition"] == "rgbd+GP"
        and row["task"] == "round_table"
        and row["skill_state"] == "leg-top-screw"
        and row["noise_id"] in {"n0", "n1", "n2", "n3", "n4"}
    }
    gp_step_lookup = {
        (str(row["skill_state"]), str(row["noise_id"])): row
        for row in per_step_rows
        if row["condition"] == "rgbd+GP" and row["task"] == "round_table"
    }
    gp_round_table_n0 = gp_round_table["n0"]
    gp_round_table_n4 = gp_round_table["n4"]
    gp_screw_rates = " -> ".join(
        f"{100.0 * float(gp_first_screw[noise_id]['skill_success_rate']):.1f}%"
        for noise_id in ("n0", "n1", "n2", "n3", "n4")
    )
    gp_n0_interval = _wilson_interval(
        int(gp_round_table_n0["n_success"]), int(gp_round_table_n0["n_rollouts"])
    )
    gp_n4_interval = _wilson_interval(
        int(gp_round_table_n4["n_success"]), int(gp_round_table_n4["n_rollouts"])
    )
    shuffle_deltas = {
        condition: 100.0
        * (
            float(grouped[(condition, "shuffle")]["success_rate"])
            - float(grouped[(condition, "n4")]["success_rate"])
        )
        for condition in CONDITION_ORDER
    }
    declining_shuffle_conditions = [
        condition for condition in CONDITION_ORDER if shuffle_deltas[condition] < 0.0
    ]
    return [
        (
            "- **`rgbd+colored GP` 是数值噪声下最稳定的 condition。** n0-n4 overall "
            "range/std 为 `3.7/1.4 pp`，略优于 `rgbd+GP` 的 `4.6/1.6 pp`；"
            "one_leg 与 round_table 的 task 内 range 分别为 `5.6/8.3 pp`，明显小于 "
            "GP 的 `19.4/16.7 pp`，lamp 均为 `11.1 pp`。两者 n4 overall 都是 "
            "`55.6%`，因此结论为“colored GP 最稳定，GP 在总体上仍稳定但对 task/reset "
            "randomization 更敏感”。"
        ),
        (
            "- **现有 tracking 不支持推断 GP 与 colored GP 使用了不同机制。** 排除 workspace "
            "外的 guidance target 后，两者 position tracking 与噪声均保持正相关：GP "
            f"`r={numeric_position_correlations['rgbd+GP']:.3f}`，colored GP "
            f"`r={numeric_position_correlations['rgbd+colored GP']:.3f}`。原 colored GP "
            "round_table n2 极值来自单个部件飞出工作空间的仿真失败，并非噪声响应。由于两个 "
            "checkpoint 不同且 tracking 只有 saved-8，当前只能比较经验稳定性，不能据此解释内部机制。"
        ),
        (
            "- **`rgbd+GP` 在 round_table 上随噪声上升的现象集中在第一段 screw，并非所有 skill "
            "都受益。** task success 从 "
            f"`{gp_round_table_n0['n_success']}/{gp_round_table_n0['n_rollouts']} = "
            f"{100.0 * float(gp_round_table_n0['success_rate']):.1f}%` 增至 "
            f"`{gp_round_table_n4['n_success']}/{gp_round_table_n4['n_rollouts']} = "
            f"{100.0 * float(gp_round_table_n4['success_rate']):.1f}%`。但 `leg-top-pick` "
            f"完成数保持 `{gp_step_lookup[('leg-top-pick', 'n0')]['completed_count']}/36 -> "
            f"{gp_step_lookup[('leg-top-pick', 'n4')]['completed_count']}/36`，`leg-top-place` "
            f"反而从 `{gp_step_lookup[('leg-top-place', 'n0')]['completed_count']}/"
            f"{gp_step_lookup[('leg-top-place', 'n0')]['reached_count']}` 降为 "
            f"`{gp_step_lookup[('leg-top-place', 'n4')]['completed_count']}/"
            f"{gp_step_lookup[('leg-top-place', 'n4')]['reached_count']}`；主要变化是 "
            f"`leg-top-screw` 条件完成率按 n0-n4 呈 `{gp_screw_rates}`，完成数从 "
            f"`{gp_first_screw['n0']['completed_count']}/{gp_first_screw['n0']['reached_count']}` "
            f"增至 `{gp_first_screw['n4']['completed_count']}/{gp_first_screw['n4']['reached_count']}`。"
            "其他 condition 的同一 step 不呈现一致单调提升，因此不是 round_table 对噪声的普遍收益。"
            "所有幅度共用 annotation noise seed 0，在相同 env/phase 上相当于将同一噪声方向按 std "
            "放大，可能恰好补偿该 checkpoint 在 screw 上的局部动作偏差；同时各 setting 的 reset "
            "未配对。n0/n4 的 95% Wilson 区间分别为 "
            f"`[{100.0 * gp_n0_interval[0]:.1f}, {100.0 * gp_n0_interval[1]:.1f}]%` 与 "
            f"`[{100.0 * gp_n4_interval[0]:.1f}, {100.0 * gp_n4_interval[1]:.1f}]%`，明显重叠，"
            "所以该曲线应解释为 seed/checkpoint-specific 局部补偿加抽样波动，而不是更大噪声能提高成功率。"
            "要验证是否存在真实的有益偏移，需要固定一组 paired reset seeds，并对多个 annotation "
            "noise seeds 汇总均值。"
        ),
        (
            f"- **`rgbd+GP+skill` 对连续数值噪声不稳定，但可能更容易拒绝明显错误的 guidance。** "
            f"它的 n0-n4 range 为 `13.9 pp`，n4 overall 为 "
            f"`{100.0 * gp_skill_n4['success_rate']:.1f}%`，低于 GP/colored GP；但它是唯一从 "
            f"n4 到 Shuffle 回升的模型：`{100.0 * gp_skill_n4['success_rate']:.1f}% -> "
            f"{100.0 * gp_skill_shuffle['success_rate']:.1f}%` (`+5.6 pp`, "
            f"`{gp_skill_n4['n_success']} -> {gp_skill_shuffle['n_success']}`/108)。回升几乎全部来自 "
            f"one_leg (`{task_rate('rgbd+GP+skill', 'one_leg', 'n4'):.1f}% -> "
            f"{task_rate('rgbd+GP+skill', 'one_leg', 'shuffle'):.1f}%`)；round_table/lamp 仅变化 "
            "`-2.8/+2.8 pp`。一种解释是 n4 仍是“可信但偏移”的点，模型会被误导；Shuffle "
            "与 one-hot skill/场景明显冲突，触发对 guidance 的 gating，退回 RGBD+skill 路径。"
            "由于两组 reset 未配对且只差 6 次成功，这不是统计显著性证据。"
        ),
        (
            "- **Grasp annotation 总体可容忍噪声，但单 task 波动更大。** grasp-part 与 colored "
            "grasp-part 的 n0-n4 overall range 为 `9.3/13.0 pp`；最大 task range 都达到 "
            "`27.8 pp`（分别出现在 round_table/lamp）。同时 position/orientation tracking 与噪声"
            "保持正相关，更符合“仍围绕视觉语义完成动作，但对零件初始随机化较敏感”的解释。"
        ),
        (
            f"- **Strong Shuffle 表明语义正确的 guidance 仍然有用，policy 并非只依赖 RGBD。** "
            f"五个 condition 合计从 n4 的 `{n4_success}/{n4_rollouts} = "
            f"{100.0 * n4_success / n4_rollouts:.1f}%` 到 Shuffle 的 "
            f"`{shuffle_success}/{shuffle_rollouts} = "
            f"{100.0 * shuffle_success / shuffle_rollouts:.1f}%`，只变化 "
            f"`{100.0 * (shuffle_success / shuffle_rollouts - n4_success / n4_rollouts):+.1f} pp`；"
            f"其中 `{len(declining_shuffle_conditions)}/5` 个 condition 下降："
            + "、".join(
                f"{condition} `{shuffle_deltas[condition]:+.1f} pp`"
                for condition in declining_shuffle_conditions
            )
            + f"，只有 rgbd+GP+skill 回升 `{shuffle_deltas['rgbd+GP+skill']:+.1f} pp`。"
            "图像与深度保持不变而 semantic-state guidance 被置乱后，大多数模型同向下降，"
            "支持正确 guidance 确实参与决策；Shuffle 后成功率仍约为一半，则说明模型同时保留了"
            "视觉/低维 fallback，而不是 guidance 是唯一输入。由于每组只有 108 rollout 且 reset "
            "未配对，单个 condition 的 3.7-7.4 pp 降幅仍应视为趋势证据。"
        ),
    ]




def generate_report(
    *,
    manifest_path: Path,
    report_path: Path,
    figures_dir: Path,
    data_dir: Path,
    previous_shuffle_manifest_path: Path | None = None,
    saved_tracking_path: Path | None = None,
) -> None:
    manifest_rows = _dedupe_latest(_read_jsonl(manifest_path))
    overall_rows, task_rows, per_step_rows, skill_type_rows = _build_rows(manifest_rows)
    saved_tracking_payload = None
    if saved_tracking_path is not None:
        if not saved_tracking_path.exists():
            raise FileNotFoundError(saved_tracking_path)
        saved_tracking_payload = _apply_saved_tracking(
            overall_rows=overall_rows,
            task_rows=task_rows,
            per_step_rows=per_step_rows,
            skill_type_rows=skill_type_rows,
            saved_tracking_path=saved_tracking_path,
        )
    task_skill_rows = _task_skill_type_rows(per_step_rows)
    numeric_tracking_rows = [
        row for row in task_rows if row["noise_id"] != "shuffle"
    ]
    tracking_complete = bool(numeric_tracking_rows) and all(
        bool(row.get("tracking_complete", False)) for row in numeric_tracking_rows
    )

    overall_csv = data_dir / "annotation_noise_clean_train_overall.csv"
    task_csv = data_dir / "annotation_noise_clean_train_by_task.csv"
    skill_type_csv = data_dir / "annotation_noise_clean_train_skill_type.csv"
    per_step_csv = data_dir / "annotation_noise_clean_train_per_step.csv"
    task_skill_csv = data_dir / "annotation_noise_clean_train_task_skill_type.csv"
    tracking_response_csv = data_dir / "annotation_noise_clean_train_tracking_response.csv"
    tracking_exclusions_csv = (
        data_dir / "annotation_noise_clean_train_tracking_workspace_exclusions.csv"
    )
    _write_csv(overall_csv, overall_rows)
    _write_csv(task_csv, task_rows)
    _write_csv(skill_type_csv, skill_type_rows)
    _write_csv(per_step_csv, per_step_rows)
    _write_csv(task_skill_csv, task_skill_rows)

    success_fig = figures_dir / "annotation_noise_clean_train_success_vs_noise.png"
    tracking_fig = figures_dir / "annotation_noise_clean_train_tracking_vs_noise.png"
    _plot_success_vs_noise(task_rows, success_fig)
    if tracking_complete:
        _plot_tracking_vs_noise(task_rows, tracking_fig)
    task_skill_grids = _plot_task_skill_condition_grids(
        task_skill_rows,
        figures_dir,
        include_tracking=tracking_complete,
    )
    success_fig_ref = Path(os.path.relpath(success_fig, report_path.parent)).as_posix()
    tracking_fig_ref = Path(os.path.relpath(tracking_fig, report_path.parent)).as_posix()
    task_skill_grid_refs = [
        (
            metric_key,
            metric_label,
            Path(os.path.relpath(path, report_path.parent)).as_posix(),
        )
        for metric_key, metric_label, path in task_skill_grids
    ]

    skill_figure_report_lines = [
        "### 1.3 Task-Skill Average（5 Skills × 3 Tasks）",
        "",
        (
                "每张总图为 5 行 skill × 3 列 task：行顺序为 `push/pick/place/insert/screw`，列顺序为 `one_leg/round_table/lamp`；每个子图中的曲线表示不同 condition。不同单位的 success、position、orientation 和 total 分图展示；GP 系列不定义 orientation/total。Tracking 子图中 n0-n4 为 workspace-filtered saved-8，Shuffle 为 legacy unfiltered full-36 参考端点。"
            if tracking_complete
            else "总图为 5 行 skill × 3 列 task：行顺序为 `push/pick/place/insert/screw`，列顺序为 `one_leg/round_table/lamp`；成功率完整发布。有效 Shuffle tracking 在表格中展示，旧 n0-n4 tracking 不进入曲线。"
        ),
        "",
    ]
    for metric_key, metric_label, figure_ref in task_skill_grid_refs:
        skill_figure_report_lines.extend(
            [
                f"#### {metric_label}",
                "",
                f"![Task-skill {metric_key}]({figure_ref})",
                "",
            ]
        )
    if not tracking_complete:
        skill_figure_report_lines.extend(
            [
                "#### Tracking Error",
                "",
                "> Tracking 图暂不发布：旧 summary 没有完整 per-rollout guidance history 覆盖证明，需用修复后的 evaluator 补跑。",
                "",
            ]
        )

    best_80_rows = _best_tolerance_rows(overall_rows, threshold=0.80)
    best_60_rows = _best_tolerance_rows(overall_rows, threshold=0.60)
    endpoint_rows = _endpoint_comparison_rows(overall_rows)
    shuffle_summary_rows = _shuffle_summary_rows(overall_rows)
    previous_overall_rows: list[dict[str, Any]] = []
    if previous_shuffle_manifest_path is not None and previous_shuffle_manifest_path.exists():
        previous_manifest_rows = _dedupe_latest(
            _read_jsonl(previous_shuffle_manifest_path)
        )
        previous_overall_rows, _, _, _ = _build_rows(previous_manifest_rows)
    shuffle_version_rows = _shuffle_version_comparison_rows(
        overall_rows, previous_overall_rows
    )
    shuffle_conclusion_lines = _shuffle_conclusion_lines(overall_rows)
    tracking_response_rows = (
        _tracking_response_rows(overall_rows) if tracking_complete else []
    )
    tracking_interpretation_lines = (
        _tracking_interpretation_lines(overall_rows) if tracking_complete else []
    )
    stability_rows = _numeric_success_stability_rows(task_rows)
    main_conclusion_lines = _main_conclusion_lines(
        overall_rows, task_rows, per_step_rows
    )
    tracking_workspace_exclusion_rows = (
        _tracking_workspace_exclusion_rows(saved_tracking_payload)
        if saved_tracking_payload is not None
        else []
    )
    _write_csv(tracking_response_csv, tracking_response_rows)
    _write_csv(tracking_exclusions_csv, tracking_workspace_exclusion_rows)

    report_lines = [
        "# 打点噪声鲁棒性实验：Clean Train -> Noisy Eval",
        "",
        *(
            [
                "> [!NOTE]",
                "> **成功率与 tracking 的样本量不同。** n0-n4 成功率使用每个 task/setting 的 36 个 rollout；n0-n4 tracking 从每个 task/setting 已保存的 8 个 rollout pickle 重新计算，用作整体 tracking 趋势的估计。Tracking 图照常画出 Shuffle 并与 n4 连线，但该端点来自旧 full-36 summary，无法事后执行 workspace 过滤，只作为 legacy 参考，不进入正式 tracking 表格、拟合或结论。",
                "",
            ]
            if saved_tracking_payload is not None
            else []
        ),
        *(
            []
            if tracking_complete
            else [
                "> [!WARNING]",
                "> **Tracking 仅部分通过完整性验收。** 旧 evaluator 产生的 n0-n4 tracking history 不完整，统一标记为 `legacy partial`；修复后重跑且满足 `episode_count=36`、`incomplete_episode_count=0`、`complete=true` 的 Shuffle tracking 可作为正式结果单独展示，但不与旧 tracking 连成曲线。",
                "",
            ]
        ),
        "## 1. 结果图",
        "",
        "### 1.1 Task Overall Success",
        "",
        f"![Success vs Noise]({success_fig_ref})",
        "",
        "### 1.2 Task Overall Tracking",
        "",
        *(
            [f"![Tracking vs Noise]({tracking_fig_ref})"]
            if tracking_complete
            else [
                "> 暂不发布完整曲线：n0-n4 缺少完整 per-rollout tracking coverage；有效 Shuffle tracking 见第 4.1 节。"
            ]
        ),
        "",
        "> 图中灰色分类区的 Shuffle tracking 来自 legacy unfiltered full-36 summary；其纵轴与 n0-n4 共享，但不共享数值噪声横轴。",
        "",
        *skill_figure_report_lines,
        "### 1.4 主要结论",
        "",
        *main_conclusion_lines,
        "",
        *(
            [
                "#### Tracking workspace 排除统计",
                "",
                "仅统计用于 n0-n4 tracking 曲线的 saved-8 rollout。单元格为 `workspace 外 final skill segment 数 / 具有有限 guidance target 的 final skill segment 总数`；同一 semantic state 重复进入时，过滤发生在取最小 tracking error 之前。`Invalid Target` 单列记录缺失或非有限 target，不与 workspace 外数据混合。",
                "",
                _markdown_table(
                    tracking_workspace_exclusion_rows,
                    [
                        ("condition", "Condition"),
                        ("one_leg", "one_leg Excluded"),
                        ("round_table", "round_table Excluded"),
                        ("lamp", "lamp Excluded"),
                        ("invalid", "Invalid Target"),
                    ],
                ),
                "",
                "历史排除项主要包含两类：任务完成后的 annotation 早退分支返回了错误 frame 的缓存坐标，以及家具部件被物理仿真抛出工作空间。前者已改为缓存并返回实际用于绘图的 robot-base noisy/clean guidance；该表反映历史数据质量，不应解释为 condition 本身的失败率。",
                "",
            ]
            if tracking_workspace_exclusion_rows
            else []
        ),
        "#### n0-n4 成功率稳定性",
        "",
        "Range/std 越小表示对数值噪声幅度越稳定；task range 是同一 task 在 n0-n4 间的最大成功率差。",
        "",
        _markdown_table(
            stability_rows,
            [
                ("condition", "Condition"),
                ("overall_range", "Overall Range"),
                ("overall_std", "Overall Std"),
                ("one_leg_range", "one_leg Range"),
                ("round_table_range", "round_table Range"),
                ("lamp_range", "lamp Range"),
                ("n4_to_shuffle", "n4->Shuffle"),
            ],
        ),
        "",
        "## 2. 实验设置",
        "",
        "- 只包含有空间 guidance 的 5 个 condition：`rgbd+GP`、`rgbd+colored GP`、`rgbd+GP+skill`、`rgbd+grasp-part`、`rgbd+grasp-part-colored`。",
        "- 训练 checkpoint 均为 clean-train 模型；本轮只评测 clean train -> noisy/shuffled eval。",
        "- 每个 condition、noise level、task 的成功率使用 36 个 rollout，randomness 为 low。",
        *(
            [
                "- n0-n4 tracking 使用每个 condition、noise level、task 已保存的 8 个 rollout；旧 Shuffle full-36 tracking 不含 workspace exclusion 明细，仅在 tracking 图中作为 legacy 参考端点，表格中仍标记为 unavailable。",
                "- 旧 pickle 仅保存通用 skill 标签；重算时将连续 skill 阶段按 task 的有序状态机 schema 对齐回 semantic state，同一推断 semantic state 多次进入时仍保留最小误差。",
            ]
            if saved_tracking_payload is not None
            else []
        ),
        "- point 的 position noise 为 xyz 每轴独立的 Gaussian std，并逐轴裁剪到 ±2σ；n0-n4 为 0/3/6/12/24 mm per axis。",
        "- grasp-part 使用相同 position noise，并耦合 0/2.5/5/10/20 deg orientation noise。",
        "- n1-n4 均使用 annotation noise seed 0；相同 env/phase 的标准高斯方向相同，仅按 noise std 缩放。因此单条幅度曲线仍包含 noise-seed-specific 效应，不等价于对零均值噪声分布取期望。",
        "- Shuffle 优先从同 task、同 skill type 的其他 semantic state 选择 donor；若不存在，则从同 task 的任意其他 semantic state 选择，禁止回退到当前 state。",
        "- tracking error 比较每个连续 skill 阶段最后一帧 final EE pose 与实际画出的 noisy/shuffled guidance pose。",
        "- tracking 只接收 robot-base workspace `x=[0.300, 0.800] m, y=[-0.550, 0.550] m, z=[0.000, 0.400] m` 内的 guidance target；workspace 外 target 会被计数并排除。z 下界取桌面高度，因为 guidance 是物体表面点而不是 EE origin。",
        "- 同一 episode 多次进入同一 semantic skill state 时，point 保留最小 position error，grasp-part 保留最小 total error。",
        "- point 只报告 position tracking；grasp-part 报告 position/orientation/total，其中 `total = pos_m / 0.01 + ori_deg / 5`。",
        f"- 当前完成组数：`{len(overall_rows)}`；task-level 数据行：`{len(task_rows)}`。",
        f"- n0-n4 Tracking 可发布覆盖：`{sum(bool(row.get('tracking_complete', False)) for row in numeric_tracking_rows)}/{len(numeric_tracking_rows)}` 个 task/setting；来源为 saved-8 重算。Shuffle 正式 tracking 覆盖为 `0/15`；图中 legacy 端点等待按新 workspace 规则重跑后替换。",
        "",
        "## 3. Task Overall 表",
        "",
        (
            "每个 task 大列内给出成功率和适用 tracking 指标；GP 显示 `P`，grasp-part 显示 `P/O/T`。括号中的 tracking `n` 是进入汇总的 skill-state 记录数，不是 rollout 数。"
            if tracking_complete
            else "每个 task 大列内的 36-rollout 成功率有效；通过 coverage 验收的 Shuffle tracking 正常展示，旧 n0-n4 tracking 标记为 `legacy partial`。"
        ),
        "",
        _task_overall_table(task_rows),
        "",
        "## 4. 端点结果",
        "",
        "跨三个 task 汇总 n0、n4 与 Shuffle；Shuffle 差值以 n4 为基准。",
        "",
        _markdown_table(
            endpoint_rows,
            [
                ("condition", "Condition"),
                ("n0_success", "n0 Success"),
                ("n4_success", "n4 Success"),
                ("success_delta", "n0->n4 Success Delta"),
                ("shuffle_success", "Shuffle Success"),
                ("shuffle_success_delta", "n4->Shuffle Success Delta"),
                *(
                    [
                        ("tracking_metric", "Tracking Metric"),
                        ("n0_tracking", "n0 Tracking"),
                        ("n4_tracking", "n4 Tracking"),
                        ("tracking_delta", "n0->n4 Tracking Delta"),
                        ("shuffle_tracking", "Shuffle Tracking"),
                        ("shuffle_tracking_delta", "n4->Shuffle Tracking Delta"),
                    ]
                    if tracking_complete
                    else []
                ),
            ],
        ),
        "",
        "### 4.1 Shuffle 成功率结果",
        "",
        "成功率以 n0 为基准；Shuffle 成功率来自完整 36 rollout。旧 tracking summary 不含 workspace exclusion 明细，相关列标记为 unavailable。",
        "",
        _markdown_table(
            shuffle_summary_rows,
            [
                ("condition", "Condition"),
                ("n0_success", "n0 Success"),
                ("shuffle_success", "Shuffle Success"),
                ("success_delta", "Delta"),
                ("track_pos_cm", "Shuffle Pos (cm)"),
                ("track_ori_deg", "Shuffle Ori (deg)"),
                ("track_total", "Shuffle Total"),
                ("tracking_count", "Tracked States"),
            ],
        ),
        "",
        "### 4.2 Fallback 修复前后",
        "",
        "旧 Shuffle 允许回退到当前 semantic state；新 Shuffle 禁止 same-state donor，并在同类型无候选时从任意其他 skill state 取点。仅比较成功率。",
        "",
        _markdown_table(
            shuffle_version_rows,
            [
                ("condition", "Condition"),
                ("n0", "n0"),
                ("old_shuffle", "Old Shuffle"),
                ("old_delta", "Old - n0"),
                ("new_shuffle", "Strong Shuffle"),
                ("new_delta", "Strong - n0"),
            ],
        ),
        "",
        "### 4.3 结论",
        "",
        *shuffle_conclusion_lines,
        "",
        "### 4.4 Numeric-Noise Tracking Response",
        "",
        "本节只拟合 n0-n4 的 saved-8 tracking 估计；8-rollout 的抽样波动会直接影响 slope、Pearson r 和 R^2，应结合曲线而不是单独作为显著性结论。",
        "",
        *(
            [
                _markdown_table(
                    tracking_response_rows,
                    [
                        ("condition", "Condition"),
                        ("metric", "Metric"),
                        ("n0", "n0"),
                        ("n4", "n4"),
                        ("delta", "Delta"),
                        ("slope", "Slope"),
                        ("pearson_r", "Pearson r"),
                        ("r_squared", "R^2"),
                    ],
                )
            ]
            if tracking_complete
            else [
                "> 暂不计算 slope、Pearson r 或 tracking/noise 比例；输入 tracking 数据未通过覆盖率验收。"
            ]
        ),
        "",
        "### 4.5 Tracking 解释",
        "",
        *tracking_interpretation_lines,
        "",
        "## 5. 成功率阈值对应的最大数值噪声",
        "",
        "### 5.1 `success_rate >= 80%`",
        "",
        _markdown_table(
            best_80_rows,
            [
                ("condition", "Condition"),
                ("success_threshold", "Threshold"),
                ("max_pos_std_mm", "Max Pos Std/axis (mm)"),
                ("max_ori_std_deg", "Max Ori Std (deg)"),
            ],
        ),
        "",
        "### 5.2 `success_rate >= 60%`",
        "",
        _markdown_table(
            best_60_rows,
            [
                ("condition", "Condition"),
                ("success_threshold", "Threshold"),
                ("max_pos_std_mm", "Max Pos Std/axis (mm)"),
                ("max_ori_std_deg", "Max Ori Std (deg)"),
            ],
        ),
        "",
        "## 6. Skill Average 表",
        "",
        (
            "跨三个 task 汇总同类 skill；point 每格为 `SR/P`，grasp-part 为 `SR/P/O/T`。"
            if tracking_complete
            else "跨三个 task 汇总同类 skill；有效 Shuffle tracking 正常展示，旧 n0-n4 tracking 显示 `legacy partial`。"
        ),
        "",
        _matrix_table(
            skill_type_rows,
            group_fields=[("condition", "Condition"), ("skill_type", "Skill Type")],
        ),
        "",
        "## 7. Per-Step 表",
        "",
        *(
            []
            if tracking_complete
            else ["分步成功率有效；有效 Shuffle tracking 正常展示，旧 n0-n4 tracking 标记为 `legacy partial`。", ""]
        ),
        _matrix_table(
            per_step_rows,
            group_fields=[
                ("condition", "Condition"),
                ("task", "Task"),
                ("skill_state", "Skill State"),
            ],
        ),
        "",
        "## 8. 原始导出",
        "",
        f"- overall csv: `{overall_csv}`",
        f"- by-task csv: `{task_csv}`",
        f"- task-skill csv: `{task_skill_csv}`",
        f"- cross-task skill-type csv: `{skill_type_csv}`",
        f"- per-step csv: `{per_step_csv}`",
        f"- tracking response csv: `{tracking_response_csv}`",
        f"- tracking workspace exclusions csv: `{tracking_exclusions_csv}`",
        *(
            [f"- saved-8 tracking json: `{saved_tracking_path}`"]
            if saved_tracking_path is not None
            else []
        ),
        f"- manifest jsonl: `{manifest_path}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("logs/annotation_noise_clean_train_fresh36_manifest.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/annotation_noise_clean_train_fresh36.md"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures/fresh36"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("reports/data/fresh36"),
    )
    parser.add_argument(
        "--previous-shuffle-manifest",
        type=Path,
        default=Path(
            "logs/archive_manifests/"
            "annotation_noise_clean_train_fresh36_manifest_before_any_skill_shuffle_20260810.jsonl"
        ),
    )
    parser.add_argument(
        "--saved-tracking",
        type=Path,
        default=Path(
            "reports/data/fresh36/"
            "annotation_noise_clean_train_tracking_saved8.json"
        ),
    )
    args = parser.parse_args()
    generate_report(
        manifest_path=args.manifest,
        report_path=args.report,
        figures_dir=args.figures_dir,
        data_dir=args.data_dir,
        previous_shuffle_manifest_path=args.previous_shuffle_manifest,
        saved_tracking_path=args.saved_tracking,
    )


if __name__ == "__main__":
    main()
