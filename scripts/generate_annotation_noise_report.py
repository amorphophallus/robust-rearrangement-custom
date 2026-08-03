from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re
from typing import Any

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


def _resolve_task_log_payload(
    run_row: dict[str, Any],
    task_name: str,
    checkpoint_name: str,
) -> dict[str, Any] | None:
    log_dir = (
        Path("logs")
        / "evaluate_model"
        / _safe_path_part(task_name)
        / _safe_path_part(checkpoint_name)
    )
    if not log_dir.exists():
        return None

    candidates = sorted(log_dir.glob("*.json"), key=lambda path: (path.stat().st_mtime, path.name))
    if not candidates:
        return None

    started_ts = _parse_iso_seconds(run_row.get("started_at"))
    ended_ts = _parse_iso_seconds(run_row.get("ended_at"))
    matched: list[Path] = []
    for path in candidates:
        mtime = path.stat().st_mtime
        if started_ts is not None and mtime < started_ts - 120:
            continue
        if ended_ts is not None and mtime > ended_ts + 120:
            continue
        matched.append(path)
    target = matched[-1] if matched else candidates[-1]
    return json.loads(target.read_text())


def _enrich_per_task_payloads(
    run_row: dict[str, Any],
    per_task: dict[str, Any],
) -> dict[str, Any]:
    enriched: dict[str, Any] = {}
    checkpoint_name = str(run_row.get("checkpoint_name", "") or "").strip()
    for task_name, task_payload in per_task.items():
        current_payload = task_payload
        tracking_error = (
            current_payload.get("tracking_error")
            if isinstance(current_payload, dict)
            else None
        )
        if isinstance(current_payload, dict) and tracking_error:
            enriched[task_name] = current_payload
            continue
        resolved = _resolve_task_log_payload(
            run_row=run_row,
            task_name=task_name,
            checkpoint_name=str(
                (current_payload or {}).get("checkpoint_name", checkpoint_name)
            ),
        )
        enriched[task_name] = resolved if resolved is not None else current_payload
    return enriched


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _format_overall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        formatted.append(
            {
                "condition": row["condition"],
                "noise": row["noise_label"],
                "pos_std_mm": f"{row['pos_std_mm']:.0f}",
                "ori_std_deg": f"{row['ori_std_deg']:.1f}",
                "success": f"{100.0 * row['success_rate']:.2f}% ({row['n_success']}/{row['n_rollouts']})",
                "track_pos_cm": f"{row['track_pos_cm']:.2f}",
                "track_ori_deg": f"{row['track_ori_deg']:.2f}",
                "track_total": f"{row['track_total']:.2f}",
                "count": str(row["skill_state_count"]),
            }
        )
    return formatted


def _format_skill_type_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        formatted.append(
            {
                "condition": row["condition"],
                "noise": row["noise_label"],
                "skill_type": row["skill_type"],
                "success": f"{100.0 * row['skill_success_rate']:.2f}% ({row['completed_count']}/{row['reached_count']})",
                "track_pos_cm": _format_optional(row["track_pos_cm"]),
                "track_ori_deg": _format_optional(row["track_ori_deg"]),
                "track_total": _format_optional(row["track_total"]),
            }
        )
    return formatted


def _format_per_step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        formatted.append(
            {
                "condition": row["condition"],
                "noise": row["noise_label"],
                "task": row["task"],
                "skill_state": row["skill_state"],
                "success": f"{100.0 * row['skill_success_rate']:.2f}% ({row['completed_count']}/{row['reached_count']})",
                "track_pos_cm": _format_optional(row["track_pos_cm"]),
                "track_ori_deg": _format_optional(row["track_ori_deg"]),
                "track_total": _format_optional(row["track_total"]),
            }
        )
    return formatted


def _format_optional(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _result_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "--"
    success_rate = float(row.get("success_rate", row.get("skill_success_rate", 0.0)))
    n_success = int(row.get("n_success", row.get("completed_count", 0)))
    n_trials = int(row.get("n_rollouts", row.get("reached_count", 0)))
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


def _plot_success_vs_noise(task_rows: list[dict[str, Any]], figure_path: Path) -> None:
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
        for condition in CONDITION_ORDER:
            rows = grouped.get(condition, [])
            regular = sorted(
                [row for row in rows if row["noise_id"] != "shuffle"],
                key=lambda item: item["pos_std_mm"],
            )
            line = axis.plot(
                [row["pos_std_mm"] for row in regular],
                [100.0 * row["success_rate"] for row in regular],
                marker="o",
                linewidth=2,
                label=condition,
                color=colors[condition],
            )[0]
            if task_idx == 0:
                legend_handles.append(line)
            clean = next((row for row in rows if row["noise_id"] == "n0"), None)
            shuffled = next((row for row in rows if row["noise_id"] == "shuffle"), None)
            if clean is not None and shuffled is not None:
                shuffle_axis.plot(
                    [0, 1],
                    [100.0 * clean["success_rate"], 100.0 * shuffled["success_rate"]],
                    marker="o",
                    linestyle="--",
                    linewidth=1.5,
                    color=colors[condition],
                )
        axis.set_xlabel("Position Noise Std (mm)")
        axis.set_title(task)
        axis.grid(True, alpha=0.3)
        shuffle_axis.set_xticks([0, 1], ["Clean", "Shuffle"], rotation=35, ha="right")
        shuffle_axis.grid(True, alpha=0.3)
        shuffle_axis.tick_params(labelleft=False)
    shared_axis.set_ylabel("Success Rate (%)")
    fig.legend(legend_handles, CONDITION_ORDER, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Clean-Train -> Noisy-Eval Success Curves")
    fig.savefig(figure_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _plot_tracking_vs_noise(task_rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(19, 13))
    grid = fig.add_gridspec(3, 6, width_ratios=[4, 1.4] * 3, hspace=0.2, wspace=0.08)
    colors = {condition: f"C{idx}" for idx, condition in enumerate(CONDITION_ORDER)}
    metrics = [
        ("track_pos_cm", "Position Error (cm)"),
        ("track_ori_deg", "Orientation Error (deg)"),
        ("track_total", "Total Error"),
    ]
    for metric_idx, (metric_key, metric_label) in enumerate(metrics):
        shared_axis = None
        for task_idx, task in enumerate(TASK_ORDER):
            axis = fig.add_subplot(grid[metric_idx, 2 * task_idx], sharey=shared_axis)
            if shared_axis is None:
                shared_axis = axis
            shuffle_axis = fig.add_subplot(
                grid[metric_idx, 2 * task_idx + 1], sharey=shared_axis
            )
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in task_rows:
                if row["task"] == task:
                    grouped[row["condition"]].append(row)
            for condition in CONDITION_ORDER:
                if metric_key != "track_pos_cm" and condition in CONDITION_ORDER[:3]:
                    continue
                all_rows = grouped.get(condition, [])
                rows = sorted(
                    [row for row in all_rows if row["noise_id"] != "shuffle"],
                    key=lambda item: item["pos_std_mm"],
                )
                axis.plot(
                    [row["pos_std_mm"] for row in rows],
                    [row[metric_key] for row in rows],
                    marker="o",
                    linewidth=2,
                    label=condition,
                    color=colors[condition],
                )
                clean = next((row for row in all_rows if row["noise_id"] == "n0"), None)
                shuffled = next(
                    (row for row in all_rows if row["noise_id"] == "shuffle"), None
                )
                if clean is not None and shuffled is not None:
                    shuffle_axis.plot(
                        [0, 1],
                        [clean[metric_key], shuffled[metric_key]],
                        marker="o",
                        linestyle="--",
                        linewidth=1.5,
                        color=colors[condition],
                    )
            if metric_idx == 0:
                axis.set_title(task)
            if metric_idx == len(metrics) - 1:
                axis.set_xlabel("Position Noise Std (mm)")
            if task_idx == 0:
                axis.set_ylabel(metric_label)
            axis.grid(True, alpha=0.3)
            shuffle_axis.set_xticks([0, 1], ["Clean", "Shuffle"], rotation=35, ha="right")
            shuffle_axis.grid(True, alpha=0.3)
            shuffle_axis.tick_params(labelleft=False)
    legend_handles = [
        Line2D([0], [0], color=colors[condition], marker="o", linewidth=2)
        for condition in CONDITION_ORDER
    ]
    fig.legend(
        legend_handles,
        CONDITION_ORDER,
        loc="lower center",
        ncol=5,
        fontsize=8,
    )
    fig.suptitle("Clean-Train -> Noisy-Eval Tracking Error Curves")
    fig.savefig(figure_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


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
        success_delta_pp = 100.0 * (n4["success_rate"] - n0["success_rate"])
        tracking_key = "track_pos_cm" if n0["family"] == "point" else "track_total"
        tracking_delta = n4[tracking_key] - n0[tracking_key]
        tracking_name = "Position (cm)" if n0["family"] == "point" else "Total"
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
            }
        )
    return comparison_rows


def _interpretation_lines(
    overall_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> list[str]:
    overall = {
        (str(row["condition"]), str(row["noise_id"])): row
        for row in overall_rows
    }
    by_task = {
        (str(row["condition"]), str(row["task"]), str(row["noise_id"])): row
        for row in task_rows
    }

    one_leg_n4 = [
        float(by_task[(condition, "one_leg", "n4")]["success_rate"])
        for condition in CONDITION_ORDER
    ]
    point_endpoint_parts = []
    for condition in CONDITION_ORDER[:3]:
        n0 = overall[(condition, "n0")]
        n4 = overall[(condition, "n4")]
        point_endpoint_parts.append(
            f"`{condition}` SR {100.0 * (n4['success_rate'] - n0['success_rate']):+.1f} pp, "
            f"P {n4['track_pos_cm'] - n0['track_pos_cm']:+.2f} cm"
        )

    grasp_endpoint_parts = []
    for condition in CONDITION_ORDER[3:]:
        n0 = overall[(condition, "n0")]
        n4 = overall[(condition, "n4")]
        grasp_endpoint_parts.append(
            f"`{condition}` SR {100.0 * (n4['success_rate'] - n0['success_rate']):+.1f} pp, "
            f"O {n4['track_ori_deg'] - n0['track_ori_deg']:+.2f} deg, "
            f"T {n4['track_total'] - n0['track_total']:+.2f}"
        )

    round_table_clean_best = max(
        float(by_task[(condition, "round_table", "n0")]["success_rate"])
        for condition in CONDITION_ORDER
    )
    lamp_clean_best = max(
        float(by_task[(condition, "lamp", "n0")]["success_rate"])
        for condition in CONDITION_ORDER
    )
    return [
        "### 3.1 主要观察",
        "",
        (
            f"- `one_leg` 在最大噪声下五个 condition 的成功率仍为 "
            f"{100.0 * min(one_leg_n4):.1f}%--{100.0 * max(one_leg_n4):.1f}%，"
            "说明简单任务对本轮最大测试噪声仍有较高容忍度。"
        ),
        "- point condition 的 n0 -> n4 跨任务端点变化："
        + "；".join(point_endpoint_parts)
        + "。成功率总体下降，但 tracking 退化幅度和单调性依 condition 而异。",
        "- grasp-part condition 的 n0 -> n4 端点变化："
        + "；".join(grasp_endpoint_parts)
        + "。两者 orientation/total tracking 都明显变差；colored 条件的成功率上升不能解释为噪声带来收益。",
        (
            f"- hard task 的 clean 上限本身有限：`round_table` 最好为 "
            f"{100.0 * round_table_clean_best:.1f}%，`lamp` 最好为 "
            f"{100.0 * lamp_clean_best:.1f}%。因此当前数据只能给出 condition/task-specific 容忍度，"
            "不能给出一个适用于所有任务的 VLM 打点精度阈值。"
        ),
        "",
        "### 3.2 结论边界",
        "",
        "- fresh rerun 中每个 task 的成功率与 tracking 都使用同一批 36 个 rollout；单次成败对应 2.78 个百分点。",
        "- 当前只使用一个 noise seed 和每个 checkpoint 一次评测，成功率曲线存在明显非单调波动；平台或拐点需要更多 rollout 和多个 seed 才能可靠确认。",
        "- 目前能够支持的结论是：clean-trained policy 的 guidance-noise 鲁棒性强烈依赖任务和 condition；grasp-part 的 6D 噪声会稳定增大 tracking error，而成功率退化在 hard task 上更明显。",
        "",
    ]


def generate_report(
    *,
    manifest_path: Path,
    report_path: Path,
    figures_dir: Path,
    data_dir: Path,
) -> None:
    manifest_rows = _dedupe_latest(_read_jsonl(manifest_path))
    overall_rows, task_rows, per_step_rows, skill_type_rows = _build_rows(manifest_rows)

    overall_csv = data_dir / "annotation_noise_clean_train_overall.csv"
    task_csv = data_dir / "annotation_noise_clean_train_by_task.csv"
    skill_type_csv = data_dir / "annotation_noise_clean_train_skill_type.csv"
    per_step_csv = data_dir / "annotation_noise_clean_train_per_step.csv"
    _write_csv(overall_csv, overall_rows)
    _write_csv(task_csv, task_rows)
    _write_csv(skill_type_csv, skill_type_rows)
    _write_csv(per_step_csv, per_step_rows)

    success_fig = figures_dir / "annotation_noise_clean_train_success_vs_noise.png"
    tracking_fig = figures_dir / "annotation_noise_clean_train_tracking_vs_noise.png"
    _plot_success_vs_noise(task_rows, success_fig)
    _plot_tracking_vs_noise(task_rows, tracking_fig)

    best_80_rows = _best_tolerance_rows(overall_rows, threshold=0.80)
    best_60_rows = _best_tolerance_rows(overall_rows, threshold=0.60)
    endpoint_rows = _endpoint_comparison_rows(overall_rows)

    report_lines = [
        "# 打点噪声鲁棒性实验：Clean Train -> Noisy Eval",
        "",
        "## 1. 设置",
        "",
        "- 只包含有空间 guidance 的 5 个 condition：`rgbd+GP`、`rgbd+colored GP`、`rgbd+GP+skill`、`rgbd+grasp-part`、`rgbd+grasp-part-colored`。",
        "- 训练 checkpoint 统一使用 clean-train 的现有模型；本轮不包含 noisy-train 曲线。",
        "- tracking error 定义为：每个 skill state 最后一帧 `final EE pose` 与实际画在图上的 noisy/shuffled `guidance pose` 的差，不计算 clean semantic pose 误差。",
        "- 同一 episode 多次进入同名 skill state 时，point 条件保留 position error 最小的一次；grasp-part 条件保留 total error 最小的一次。跨 episode、task 按有效 skill-state 记录数加权平均。",
        "- point 条件只报告 `P` (position error, cm)，不定义 orientation/total tracking；grasp-part 报告 `P/O/T`，其中 `total = pos_m / 0.01 + ori_deg / 5`。",
        "- n0-n4 全部从零重跑，每个 task 使用同一批 36 个 rollout 计算成功率、分步成功率和 tracking。",
        "- point guidance 的 x 轴为位置噪声；grasp-part 的位置噪声相同，同时 n1-n4 分别耦合 2.5/5/10/20 deg orientation noise。",
        "- 噪声档位：point 为 n0=0、n1=3、n2=6、n3=12、n4=24 mm；grasp-part 在相同位置档位上分别叠加 0/2.5/5/10/20 deg。",
        "- shuffled guidance 从同 task clean bank 中按 skill type 优先选择其他 semantic state；图中放在独立窄轴，不作为数值噪声档位。",
        f"- 当前完成组数：`{len(overall_rows)}`；task-level 数据行：`{len(task_rows)}`。",
        "",
        "## 2. 曲线",
        "",
        f"![Success vs Noise](./figures/{success_fig.name})",
        "",
        f"![Tracking vs Noise](./figures/{tracking_fig.name})",
        "",
        "## 3. 端点结果摘要",
        "",
        "下表比较跨三个 task 加权汇总后的 n0 与 n4；point 使用 position tracking，grasp-part 使用 total tracking。",
        "",
        _markdown_table(
            endpoint_rows,
            [
                ("condition", "Condition"),
                ("n0_success", "n0 Success"),
                ("n4_success", "n4 Success"),
                ("success_delta", "Success Delta"),
                ("tracking_metric", "Tracking Metric"),
                ("n0_tracking", "n0 Tracking"),
                ("n4_tracking", "n4 Tracking"),
                ("tracking_delta", "Tracking Delta"),
            ],
        ),
        "",
        *_interpretation_lines(overall_rows, task_rows),
        "## 4. Table 1: Task Overall",
        "",
        "每个 task 大列内给出 `SR` 和适用 tracking 指标；GP 只显示 `P`，grasp-part 显示 `P/O/T`。括号内 `n` 是 tracking 有效 skill-state 数。",
        "",
        _task_overall_table(task_rows),
        "",
        "## 5. 成功率阈值对应的最大噪声",
        "",
        "### 5.1 `success_rate >= 80%`",
        "",
        _markdown_table(
            best_80_rows,
            [
                ("condition", "Condition"),
                ("success_threshold", "Threshold"),
                ("max_pos_std_mm", "Max Pos Std (mm)"),
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
                ("max_pos_std_mm", "Max Pos Std (mm)"),
                ("max_ori_std_deg", "Max Ori Std (deg)"),
            ],
        ),
        "",
        "## 6. Skill Average 表",
        "",
        "跨三个 task 汇总同类 skill（push/pick/place/insert/screw）；point 每格为 `SR/P`，grasp-part 为 `SR/P/O/T`。",
        "",
        _matrix_table(
            skill_type_rows,
            group_fields=[("condition", "Condition"), ("skill_type", "Skill Type")],
        ),
        "",
        "## 7. Per-Step 表",
        "",
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
        f"- skill-type csv: `{skill_type_csv}`",
        f"- per-step csv: `{per_step_csv}`",
        f"- manifest jsonl: `{manifest_path}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/data/annotation_noise_clean_train_manifest.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/annotation_noise_clean_train_eval.md"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("reports/data"),
    )
    args = parser.parse_args()
    generate_report(
        manifest_path=args.manifest,
        report_path=args.report,
        figures_dir=args.figures_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
