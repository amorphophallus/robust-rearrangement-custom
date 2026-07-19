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


def _weighted_tracking_overall(per_task: dict[str, Any]) -> dict[str, float | int]:
    total_count = 0
    pos_sum = 0.0
    ori_sum = 0.0
    total_sum = 0.0
    for task_payload in per_task.values():
        overall = (task_payload.get("tracking_error") or {}).get("overall", {})
        count = int(overall.get("count", 0))
        if count <= 0:
            continue
        total_count += count
        pos_sum += float(overall.get("mean_pos_m", 0.0)) * count
        ori_sum += float(overall.get("mean_ori_deg", 0.0)) * count
        total_sum += float(overall.get("mean_total", 0.0)) * count
    if total_count <= 0:
        return {"count": 0, "mean_pos_m": 0.0, "mean_ori_deg": 0.0, "mean_total": 0.0}
    return {
        "count": total_count,
        "mean_pos_m": pos_sum / total_count,
        "mean_ori_deg": ori_sum / total_count,
        "mean_total": total_sum / total_count,
    }


def _build_rows(
    manifest_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall_rows: list[dict[str, Any]] = []
    per_step_rows: list[dict[str, Any]] = []
    skill_type_rows: list[dict[str, Any]] = []

    for run_row in manifest_rows:
        if run_row.get("status") != "ok":
            continue
        summary_path = _resolve_summary_path(run_row)
        if summary_path is None or not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text())
        per_task = _enrich_per_task_payloads(run_row, payload.get("per_task", {}))
        tracking_overall = _weighted_tracking_overall(per_task)
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
            "track_ori_deg": float(tracking_overall["mean_ori_deg"]),
            "track_total": float(tracking_overall["mean_total"]),
            "skill_state_count": int(tracking_overall["count"]),
            "summary_json": str(summary_path),
        }
        for task in TASK_ORDER:
            task_payload = per_task.get(task, {})
            overall_row[f"{task}_success_rate"] = float(
                task_payload.get("success_rate", 0.0) or 0.0
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
            for skill_state, stats in by_skill.items():
                count = int(stats.get("count", 0))
                if count <= 0:
                    continue
                reached = int(state_counts.get(skill_state, 0))
                completed = int(completion_counts.get(skill_state, 0))
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
                        "track_pos_cm": float(stats.get("mean_pos_m", 0.0)) * 100.0,
                        "track_ori_deg": float(stats.get("mean_ori_deg", 0.0)),
                        "track_total": float(stats.get("mean_total", 0.0)),
                        "tracking_count": count,
                    }
                )
                skill_type = _skill_type_from_state(skill_state)
                bucket = skill_type_accumulator[skill_type]
                bucket["count"] += count
                bucket["pos_sum_cm"] += float(stats.get("mean_pos_m", 0.0)) * 100.0 * count
                bucket["ori_sum_deg"] += float(stats.get("mean_ori_deg", 0.0)) * count
                bucket["total_sum"] += float(stats.get("mean_total", 0.0)) * count
                bucket["reached"] += reached
                bucket["completed"] += completed

        for skill_type in SKILL_TYPE_ORDER:
            bucket = skill_type_accumulator.get(skill_type)
            if not bucket or bucket["count"] <= 0:
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
                    "track_pos_cm": float(bucket["pos_sum_cm"]) / float(bucket["count"]),
                    "track_ori_deg": float(bucket["ori_sum_deg"]) / float(bucket["count"]),
                    "track_total": float(bucket["total_sum"]) / float(bucket["count"]),
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
    return overall_rows, per_step_rows, skill_type_rows


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
                "track_pos_cm": f"{row['track_pos_cm']:.2f}",
                "track_ori_deg": f"{row['track_ori_deg']:.2f}",
                "track_total": f"{row['track_total']:.2f}",
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
                "track_pos_cm": f"{row['track_pos_cm']:.2f}",
                "track_ori_deg": f"{row['track_ori_deg']:.2f}",
                "track_total": f"{row['track_total']:.2f}",
            }
        )
    return formatted


def _plot_success_vs_noise(overall_rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    family_rows = {"point": [], "grasp-part": []}
    for row in overall_rows:
        family_rows[row["family"]].append(row)

    for axis, family in zip(axes, FAMILY_ORDER):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in family_rows[family]:
            grouped[row["condition"]].append(row)
        for condition, rows in grouped.items():
            rows = sorted(rows, key=lambda item: item["pos_std_mm"])
            axis.plot(
                [row["pos_std_mm"] for row in rows],
                [100.0 * row["success_rate"] for row in rows],
                marker="o",
                linewidth=2,
                label=condition,
            )
        axis.set_xlabel("Position Noise Std (mm)")
        axis.set_title(
            "Point Guidance" if family == "point" else "Grasp-Part Guidance"
        )
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Success Rate (%)")
    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle("Clean-Train -> Noisy-Eval Success Curves")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_tracking_vs_noise(overall_rows: list[dict[str, Any]], figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = [
        ("track_pos_cm", "Tracking Pos Error (cm)"),
        ("track_ori_deg", "Tracking Ori Error (deg)"),
        ("track_total", "Tracking Total"),
    ]
    for axis, (metric_key, metric_title) in zip(axes, metric_specs):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in overall_rows:
            grouped[row["condition"]].append(row)
        for condition, rows in grouped.items():
            rows = sorted(rows, key=lambda item: item["pos_std_mm"])
            axis.plot(
                [row["pos_std_mm"] for row in rows],
                [row[metric_key] for row in rows],
                marker="o",
                linewidth=2,
                label=condition,
            )
        axis.set_xlabel("Position Noise Std (mm)")
        axis.set_title(metric_title)
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Clean-Train -> Noisy-Eval Tracking Curves")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _best_tolerance_rows(overall_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overall_rows:
        grouped[row["condition"]].append(row)
    rows = []
    for condition, candidates in grouped.items():
        valid = [row for row in candidates if row["success_rate"] >= threshold]
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


def generate_report(
    *,
    manifest_path: Path,
    report_path: Path,
    figures_dir: Path,
    data_dir: Path,
) -> None:
    manifest_rows = _dedupe_latest(_read_jsonl(manifest_path))
    overall_rows, per_step_rows, skill_type_rows = _build_rows(manifest_rows)

    overall_csv = data_dir / "annotation_noise_clean_train_overall.csv"
    skill_type_csv = data_dir / "annotation_noise_clean_train_skill_type.csv"
    per_step_csv = data_dir / "annotation_noise_clean_train_per_step.csv"
    _write_csv(overall_csv, overall_rows)
    _write_csv(skill_type_csv, skill_type_rows)
    _write_csv(per_step_csv, per_step_rows)

    success_fig = figures_dir / "annotation_noise_clean_train_success_vs_noise.png"
    tracking_fig = figures_dir / "annotation_noise_clean_train_tracking_vs_noise.png"
    _plot_success_vs_noise(overall_rows, success_fig)
    _plot_tracking_vs_noise(overall_rows, tracking_fig)

    best_80_rows = _best_tolerance_rows(overall_rows, threshold=0.80)
    best_60_rows = _best_tolerance_rows(overall_rows, threshold=0.60)

    report_lines = [
        "# 打点噪声鲁棒性实验：Clean Train -> Noisy Eval",
        "",
        "## 1. 设置",
        "",
        "- 只包含有空间 guidance 的 5 个 condition：`rgbd+GP`、`rgbd+colored GP`、`rgbd+GP+skill`、`rgbd+grasp-part`、`rgbd+grasp-part-colored`。",
        "- 训练 checkpoint 统一使用 clean-train 的现有模型；本轮不包含 noisy-train 曲线。",
        "- tracking error 统一定义为：每个 skill state 最后一帧 `final EE pose` 与当前画在图上的 `guidance pose` 的差。",
        "- `overall tracking error` 按所有 task 的全部 skill-state 计数加权平均。",
        "",
        "## 2. 曲线",
        "",
        f"![Success vs Noise](./figures/{success_fig.name})",
        "",
        f"![Tracking vs Noise](./figures/{tracking_fig.name})",
        "",
        "## 3. Overall 表",
        "",
        _markdown_table(
            _format_overall_rows(overall_rows),
            [
                ("condition", "Condition"),
                ("noise", "Noise"),
                ("pos_std_mm", "Pos Std (mm)"),
                ("ori_std_deg", "Ori Std (deg)"),
                ("success", "Success"),
                ("track_pos_cm", "Track Pos (cm)"),
                ("track_ori_deg", "Track Ori (deg)"),
                ("track_total", "Track Total"),
                ("count", "Count"),
            ],
        ),
        "",
        "## 4. 成功率阈值对应的最大噪声",
        "",
        "### 4.1 `success_rate >= 80%`",
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
        "### 4.2 `success_rate >= 60%`",
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
        "## 5. Skill-Type Average 表",
        "",
        _markdown_table(
            _format_skill_type_rows(skill_type_rows),
            [
                ("condition", "Condition"),
                ("noise", "Noise"),
                ("skill_type", "Skill Type"),
                ("success", "Success"),
                ("track_pos_cm", "Track Pos (cm)"),
                ("track_ori_deg", "Track Ori (deg)"),
                ("track_total", "Track Total"),
            ],
        ),
        "",
        "## 6. Per-Step Tracking 表",
        "",
        _markdown_table(
            _format_per_step_rows(per_step_rows),
            [
                ("condition", "Condition"),
                ("noise", "Noise"),
                ("task", "Task"),
                ("skill_state", "Skill State"),
                ("success", "Success"),
                ("track_pos_cm", "Track Pos (cm)"),
                ("track_ori_deg", "Track Ori (deg)"),
                ("track_total", "Track Total"),
            ],
        ),
        "",
        "## 7. 原始导出",
        "",
        f"- overall csv: `{overall_csv}`",
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
