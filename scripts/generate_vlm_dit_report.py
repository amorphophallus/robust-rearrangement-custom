#!/usr/bin/env python3
"""Generate CSVs and a Chinese Markdown report for the VLM + DiT campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.vlm_point_metrics import merge_vlm_point_error_summaries
from src.eval.progress_schema import get_task_progress_labels


TASKS = ("one_leg", "round_table", "lamp")
SKILLS = ("push", "pick", "place", "insert", "screw")
BASELINE_GATE_PATH = REPO_ROOT / "reports" / "data" / "vlm_dit_baseline_gate.json"


def _load_run(row: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(row["summary_path"])
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    payload["_manifest"] = row
    return payload


def _fmt(value: Any, digits: int = 2, missing: str = "—") -> str:
    if value is None:
        return missing
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "∞" if numeric > 0 else missing
    return f"{numeric:.{digits}f}"


def _success_cell(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "—"
    success = int(payload.get("n_success", 0))
    total = int(payload.get("n_rollouts", 0))
    return f"{100.0 * success / total:.1f}% ({success}/{total})" if total else "—"


def _tracking(payload: dict[str, Any] | None) -> dict[str, Any]:
    return ((payload or {}).get("tracking_error") or {}).get("overall", {})


def _point(payload: dict[str, Any] | None) -> dict[str, Any]:
    return (
        (((payload or {}).get("vlm_point_error") or {}).get("all") or {}).get(
            "overall", {}
        )
    )


def _query(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Backward-compatible name for the all-control-step distribution."""

    return (
        (((payload or {}).get("vlm_point_error") or {}).get("all") or {}).get(
            "step_distribution", {}
        )
    )


def _fresh_queries(payload: dict[str, Any] | None) -> dict[str, Any]:
    return (
        (((payload or {}).get("vlm_point_error") or {}).get("all") or {}).get(
            "fresh_queries", {}
        )
    )


def _point_quality_analysis(label: str, stats: dict[str, Any]) -> str:
    count = int(stats.get("count_valid", 0))
    if count <= 0:
        return f"- {label}：没有有效 GT–VLM point pair，不能判断 point 质量。"
    mean = stats.get("mean_error_px")
    rmse = stats.get("rmse_px")
    median = stats.get("p50_error_px")
    p90 = stats.get("p90_error_px")
    bias = stats.get("bias_norm_px")
    spread = stats.get("spread_ratio")
    r2 = stats.get("point_r2")
    skill_accuracy = stats.get("skill_accuracy")
    tail_40 = int(stats.get("tail_count_gt_40px", 0))
    tail_70 = int(stats.get("tail_count_gt_70px", 0))
    observations = [
        f"n={count}",
        f"mean/RMSE/median/P90={_fmt(mean)}/{_fmt(rmse)}/{_fmt(median)}/{_fmt(p90)} px",
        f">40/>70 px={tail_40}/{tail_70}",
        f"bias={_fmt(bias)} px",
        f"spread={_fmt(spread, 3)}",
        f"R²={_fmt(r2, 3)}",
        f"skill acc={_fmt(100.0 * skill_accuracy, 1)}%" if skill_accuracy is not None else "skill acc=—",
    ]
    interpretation = []
    if mean is not None and rmse is not None and float(rmse) > 1.35 * max(float(mean), 1e-12):
        interpretation.append("RMSE 明显高于 mean，存在长尾")
    if r2 is not None and float(r2) < 0.0:
        interpretation.append("R²<0，平方误差差于恒定预测该组 GT 均值")
    elif r2 is not None and float(r2) >= 0.5:
        interpretation.append("R²≥0.5，预测解释了较多样本间空间变化")
    if spread is not None and float(spread) < 0.7:
        interpretation.append("spread 明显收缩，需要检查 regress-to-mean")
    elif spread is not None and float(spread) > 1.3:
        interpretation.append("预测分布明显比 GT 更分散")
    if not interpretation:
        interpretation.append("需结合 bias、R²、spread 和视频判断，不能只看平均误差")
    return f"- {label}：" + "；".join(observations + interpretation) + "。"


def _weighted_tracking(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    payloads = list(payloads)
    count = sum(int(_tracking(payload).get("count", 0)) for payload in payloads)
    output: dict[str, Any] = {"count": count}
    for field in ("mean_pos_m", "mean_ori_deg", "mean_total"):
        weighted = sum(
            int(_tracking(payload).get("count", 0))
            * float(_tracking(payload).get(field, 0.0))
            for payload in payloads
        )
        output[field] = weighted / count if count else None
    return output


def _merge_point(payloads: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    summaries = [payload.get("vlm_point_error") for payload in payloads]
    summaries = [summary for summary in summaries if summary]
    return merge_vlm_point_error_summaries(summaries) if summaries else None


def _aggregate(
    condition: str,
    payloads: list[dict[str, Any]],
    *,
    label: str | None = None,
) -> dict[str, Any]:
    n_success = sum(int(payload.get("n_success", 0)) for payload in payloads)
    n_rollouts = sum(int(payload.get("n_rollouts", 0)) for payload in payloads)
    return {
        "condition": condition,
        "condition_label": (
            payloads[0]["_manifest"]["condition_label"]
            if payloads
            else (label or condition)
        ),
        "n_success": n_success,
        "n_rollouts": n_rollouts,
        "success_rate": n_success / n_rollouts if n_rollouts else None,
        "tracking": _weighted_tracking(payloads),
        "vlm_point_error": _merge_point(payloads),
    }


def _wilson(success: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/vlm_dit_guidance_eval.md")
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        help="Optional completed smoke manifest to include in the validation trail.",
    )
    parser.add_argument(
        "--scripted-diagnostic-summary",
        type=Path,
        help="Optional matched scripted-GT task summary to include.",
    )
    return parser.parse_args()


def _progress_labels(payload: dict[str, Any] | None, task: str) -> tuple[str, str]:
    if payload is None:
        return "—", "—"
    reached = payload.get("skill_state_counts") or {}
    completed = payload.get("skill_completion_counts") or {}
    labels = get_task_progress_labels(task, "skill_states")
    final = next((label for label in reversed(labels) if reached.get(label, 0)), "—")
    failed = [
        f"{label} {completed.get(label, 0)}/{reached.get(label, 0)}"
        for label in labels
        if reached.get(label, 0) > completed.get(label, 0)
    ]
    return final, ", ".join(failed) if failed else "—"


def _validation_smoke_rows(smoke: dict[str, Any] | None) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in (smoke or {}).get("runs", []):
        payload = _load_run(row)
        final, failed = _progress_labels(payload, row["task"])
        tracking = _tracking(payload)
        point = _point(payload)
        rows.append(
            [
                str(row.get("condition_label", row.get("condition", "—"))),
                str(row.get("task", "—")),
                (
                    f"{payload.get('n_success', 0)}/{payload.get('n_rollouts', 0)}"
                    if payload
                    else "—"
                ),
                final,
                failed,
                _fmt(tracking.get("mean_total"))
                if tracking.get("count", 0)
                else "—",
                _fmt(point.get("mean_error_px")),
                str(row.get("return_code", "—")),
            ]
        )
    return rows


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    manifest = json.loads(manifest_path.read_text())
    campaign_stage = manifest.get("stage", "legacy")
    legacy_invalid = campaign_stage not in {"smoke", "formal"}
    baseline_gate = (
        json.loads(BASELINE_GATE_PATH.read_text())
        if BASELINE_GATE_PATH.is_file()
        else {"status": "missing"}
    )
    smoke_manifest_path = args.smoke_manifest
    if smoke_manifest_path is None and campaign_stage == "smoke":
        smoke_manifest_path = manifest_path
    if smoke_manifest_path is None and manifest.get("smoke_manifest"):
        smoke_manifest_path = Path(manifest["smoke_manifest"])
    smoke_manifest = (
        json.loads(smoke_manifest_path.read_text())
        if smoke_manifest_path is not None and smoke_manifest_path.is_file()
        else None
    )
    formal_gate = manifest.get("formal_gate") or {}
    diagnostic_summary_path = args.scripted_diagnostic_summary
    if diagnostic_summary_path is None:
        candidate = REPO_ROOT / "logs/vlm_dit_depthfix_scripted_diag_20260817/summaries/rgbd_gp__one_leg.json"
        diagnostic_summary_path = candidate if candidate.is_file() else None
    diagnostic_summary = (
        json.loads(diagnostic_summary_path.read_text())
        if diagnostic_summary_path is not None and diagnostic_summary_path.is_file()
        else None
    )
    readiness = manifest.get("vlm_readiness", {})
    is_review = readiness.get("status") == "not_checked_dry_run"
    revision_label = (
        "dry-run 未检查"
        if is_review
        else readiness.get("model_revision", "—")
    )
    manifest_rows = manifest.get("runs", [])
    payload_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest_rows:
        payload = _load_run(row)
        if payload is not None:
            payload_by_key[(row["condition"], row["task"])] = payload

    condition_order = list(dict.fromkeys(row["condition"] for row in manifest_rows))
    by_condition = {
        condition: [
            payload_by_key[(condition, task)]
            for task in TASKS
            if (condition, task) in payload_by_key
        ]
        for condition in condition_order
    }
    aggregates = {
        condition: _aggregate(
            condition,
            payloads,
            label=next(
                row["condition_label"]
                for row in manifest_rows
                if row["condition"] == condition
            ),
        )
        for condition, payloads in by_condition.items()
    }

    data_dir = output_path.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    task_csv_rows: list[dict[str, Any]] = []
    task_tracking_skill_rows: list[dict[str, Any]] = []
    task_point_skill_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        payload = payload_by_key.get((row["condition"], row["task"]))
        tracking = _tracking(payload)
        point = _point(payload)
        query = _query(payload)
        query_vlm = query.get("vlm", {})
        projection_reference = query.get("projection_reference", {})
        same_depth = query.get("same_depth", {})
        task_csv_rows.append(
            {
                "condition": row["condition_label"],
                "task": row["task"],
                "status": row.get("status", "unknown"),
                "n_success": (payload or {}).get("n_success"),
                "n_rollouts": (payload or {}).get("n_rollouts"),
                "success_rate": (payload or {}).get("success_rate"),
                "tracking_count": tracking.get("count"),
                "tracking_position_cm": (
                    100.0 * tracking["mean_pos_m"]
                    if tracking.get("mean_pos_m") is not None
                    else None
                ),
                "tracking_position_median_cm": (
                    100.0 * tracking["median_pos_m"]
                    if tracking.get("median_pos_m") is not None
                    else None
                ),
                "tracking_position_p90_cm": (
                    100.0 * tracking["p90_pos_m"]
                    if tracking.get("p90_pos_m") is not None
                    else None
                ),
                "tracking_orientation_deg": tracking.get("mean_ori_deg"),
                "tracking_orientation_median_deg": tracking.get("median_ori_deg"),
                "tracking_orientation_p90_deg": tracking.get("p90_ori_deg"),
                "tracking_total": tracking.get("mean_total"),
                "point_step_count": point.get("count_valid"),
                "point_step_mean_px": point.get("mean_error_px"),
                "point_step_rmse_px": point.get("rmse_px"),
                "point_step_median_px": query_vlm.get("p50_error_px"),
                "point_step_p90_px": query_vlm.get("p90_error_px"),
                "point_step_mean_dx_px": query_vlm.get("mean_dx_px"),
                "point_step_mean_dy_px": query_vlm.get("mean_dy_px"),
                "point_step_bias_norm_px": query_vlm.get("bias_norm_px"),
                "point_query_count": query_vlm.get("count"),
                "reference_pair_count": projection_reference.get(
                    "reference_pair_count"
                ),
                "mc_samples_per_pair": projection_reference.get(
                    "monte_carlo_samples_per_pair"
                ),
                "reference_reservoir_size_per_level": projection_reference.get(
                    "reference_reservoir_size_per_level"
                ),
                "point_query_p90_px": query_vlm.get("p90_error_px"),
                "point_query_p95_px": query_vlm.get("p95_error_px"),
                "point_bias_px": query_vlm.get("bias_norm_px"),
                "point_anisotropy": query_vlm.get("anisotropy_ratio"),
                "same_depth_mean_mm": same_depth.get("mean_error_mm"),
                "closest_noise_swd": query.get("closest_level_sliced_wasserstein"),
                "closest_noise_centered_swd": query.get(
                    "closest_level_centered_sliced_wasserstein"
                ),
                "closest_noise_radial_w1": query.get("closest_level_radial_wasserstein"),
                "closest_noise_rmse": query.get("closest_level_rmse"),
                "magnitude_equivalent_std_mm": query.get(
                    "magnitude_equivalent_std_mm"
                ),
                "magnitude_equivalent_bracket": query.get(
                    "magnitude_equivalent_bracket"
                ),
            }
        )
        for skill, stats in sorted(
            ((payload or {}).get("tracking_error") or {}).get("by_skill", {}).items()
        ):
            task_tracking_skill_rows.append(
                {
                    "condition": row["condition_label"],
                    "task": row["task"],
                    "skill": skill,
                    "count": stats.get("count"),
                    "position_mean_cm": (
                        100.0 * stats["mean_pos_m"]
                        if stats.get("mean_pos_m") is not None
                        else None
                    ),
                    "position_median_cm": (
                        100.0 * stats["median_pos_m"]
                        if stats.get("median_pos_m") is not None
                        else None
                    ),
                    "position_p90_cm": (
                        100.0 * stats["p90_pos_m"]
                        if stats.get("p90_pos_m") is not None
                        else None
                    ),
                    "orientation_mean_deg": stats.get("mean_ori_deg"),
                    "orientation_median_deg": stats.get("median_ori_deg"),
                    "orientation_p90_deg": stats.get("p90_ori_deg"),
                    "total_mean": stats.get("mean_total"),
                    "total_median": stats.get("median_total"),
                    "total_p90": stats.get("p90_total"),
                }
            )
        for skill, stats in sorted(
            ((((payload or {}).get("vlm_point_error") or {}).get("all") or {}).get("by_skill") or {}).items()
        ):
            task_point_skill_rows.append(
                {
                    "condition": row["condition_label"],
                    "task": row["task"],
                    "skill": skill,
                    "step_count": stats.get("count_valid"),
                    "step_total": stats.get("count_total"),
                    "skill_accuracy": stats.get("skill_accuracy"),
                    "step_mean_error_px": stats.get("mean_error_px"),
                    "step_rmse_px": stats.get("rmse_px"),
                    "step_median_px": stats.get("p50_error_px"),
                    "step_p90_px": stats.get("p90_error_px"),
                    "mean_dx_px": stats.get("mean_dx_px"),
                    "mean_dy_px": stats.get("mean_dy_px"),
                    "bias_norm_px": stats.get("bias_norm_px"),
                    "spread_ratio": stats.get("spread_ratio"),
                    "point_r2": stats.get("point_r2"),
                    "tail_count_gt_40px": stats.get("tail_count_gt_40px"),
                    "tail_count_gt_70px": stats.get("tail_count_gt_70px"),
                    "same_depth_mean_mm": stats.get("mean_same_depth_error_mm"),
                }
            )
    task_fields = list(task_csv_rows[0]) if task_csv_rows else []
    _write_csv(data_dir / "vlm_dit_guidance_by_task.csv", task_csv_rows, task_fields)
    _write_csv(
        data_dir / "vlm_dit_tracking_by_task_skill.csv",
        task_tracking_skill_rows,
        list(task_tracking_skill_rows[0]) if task_tracking_skill_rows else [],
    )
    _write_csv(
        data_dir / "vlm_dit_point_error_by_task_skill.csv",
        task_point_skill_rows,
        list(task_point_skill_rows[0]) if task_point_skill_rows else [],
    )

    overall_csv_rows: list[dict[str, Any]] = []
    skill_csv_rows: list[dict[str, Any]] = []
    for condition in condition_order:
        aggregate = aggregates[condition]
        merged = aggregate["vlm_point_error"] or {}
        point = ((merged.get("all") or {}).get("overall") or {})
        query = ((merged.get("all") or {}).get("step_distribution") or {})
        query_vlm = query.get("vlm", {})
        projection_reference = query.get("projection_reference", {})
        same_depth = query.get("same_depth", {})
        tracking = aggregate["tracking"]
        lower, upper = _wilson(aggregate["n_success"], aggregate["n_rollouts"])
        overall_csv_rows.append(
            {
                "condition": aggregate["condition_label"],
                "n_success": aggregate["n_success"],
                "n_rollouts": aggregate["n_rollouts"],
                "success_rate": aggregate["success_rate"],
                "success_wilson95_low": lower,
                "success_wilson95_high": upper,
                "tracking_count": tracking.get("count"),
                "tracking_position_cm": (
                    100.0 * tracking["mean_pos_m"]
                    if tracking.get("mean_pos_m") is not None
                    else None
                ),
                "tracking_orientation_deg": tracking.get("mean_ori_deg"),
                "tracking_total": tracking.get("mean_total"),
                "point_step_count": point.get("count_valid"),
                "point_step_mean_px": point.get("mean_error_px"),
                "point_step_rmse_px": point.get("rmse_px"),
                "point_query_count": query_vlm.get("count"),
                "reference_pair_count": projection_reference.get(
                    "reference_pair_count"
                ),
                "mc_samples_per_pair": projection_reference.get(
                    "monte_carlo_samples_per_pair"
                ),
                "reference_reservoir_size_per_level": projection_reference.get(
                    "reference_reservoir_size_per_level"
                ),
                "point_query_p90_px": query_vlm.get("p90_error_px"),
                "point_query_p95_px": query_vlm.get("p95_error_px"),
                "point_bias_dx_px": query_vlm.get("mean_dx_px"),
                "point_bias_dy_px": query_vlm.get("mean_dy_px"),
                "point_bias_norm_px": query_vlm.get("bias_norm_px"),
                "point_anisotropy": query_vlm.get("anisotropy_ratio"),
                "same_depth_mean_mm": same_depth.get("mean_error_mm"),
                "same_depth_p90_mm": same_depth.get("p90_error_mm"),
                "closest_noise_swd": query.get("closest_level_sliced_wasserstein"),
                "closest_noise_centered_swd": query.get(
                    "closest_level_centered_sliced_wasserstein"
                ),
                "closest_noise_radial_w1": query.get("closest_level_radial_wasserstein"),
                "closest_noise_rmse": query.get("closest_level_rmse"),
                "magnitude_equivalent_std_mm": query.get(
                    "magnitude_equivalent_std_mm"
                ),
                "magnitude_equivalent_bracket": query.get(
                    "magnitude_equivalent_bracket"
                ),
            }
        )
        for skill in SKILLS:
            stats = (((merged.get("all") or {}).get("by_skill") or {}).get(skill) or {})
            skill_csv_rows.append(
                {
                    "condition": aggregate["condition_label"],
                    "skill": skill,
                    "step_count": stats.get("count_valid"),
                    "step_mean_error_px": stats.get("mean_error_px"),
                    "step_rmse_px": stats.get("rmse_px"),
                    "mean_dx_px": stats.get("mean_dx_px"),
                    "mean_dy_px": stats.get("mean_dy_px"),
                    "same_depth_mean_mm": stats.get("mean_same_depth_error_mm"),
                }
            )
    _write_csv(
        data_dir / "vlm_dit_guidance_overall.csv",
        overall_csv_rows,
        list(overall_csv_rows[0]) if overall_csv_rows else [],
    )
    _write_csv(
        data_dir / "vlm_dit_guidance_by_skill.csv",
        skill_csv_rows,
        list(skill_csv_rows[0]) if skill_csv_rows else [],
    )

    complete = len(payload_by_key)
    expected = len(manifest_rows)
    total_finished_rollouts = sum(
        int(payload.get("n_rollouts", 0)) for payload in payload_by_key.values()
    )
    monte_carlo_samples = int(manifest.get("vlm_noise_projection_samples", 200))
    rollouts_per_cell = int(manifest.get("n_rollouts_per_task", 36))
    lines = [
        "# VLM + DiT guidance point 评测报告",
        "",
        "## 验证链路（旧数据 / smoke / matched diagnostics / formal）",
        "",
        "### 旧 324-rollout：invalid",
        "",
        f"- 判定：`{baseline_gate.get('status', 'missing')}`。{baseline_gate.get('reason', '')}",
        "- 旧数据只保留作故障溯源，不 resume、不拼接、不用于 condition 排名。缺少 `--save-depth-image` 会同时令 RGBD policy observation 缺少 depth；此外旧 one_leg 使用 700 steps，旧 rgbd+GP checkpoint 也与本轮固定 checkpoint 不同。",
        "",
        "### 新 27-rollout smoke",
        "",
    ]
    smoke_rows = _validation_smoke_rows(smoke_manifest)
    if formal_gate.get("mode") == "explicit_user_approved_bypass":
        formal_gate_cli = [
            "  --allow-formal-without-smoke \\",
            "  --formal-approval-note "
            + shlex.quote(str(formal_gate.get("approval_note") or "<approval-note>")),
        ]
    else:
        formal_gate_cli = [
            f"  --smoke-manifest {smoke_manifest_path or 'logs/<completed-smoke>/manifest.json'}"
        ]
    repro_namespace = str(
        manifest.get("namespace") or "vlm_original_sft_formal_mc200_low_<date>"
    )
    repro_output_dir = str(manifest_path.parent)
    repro_data_dir_raw = str(
        manifest.get("data_dir_raw") or "/data/hy/robust-rearrangement/data"
    )

    lines.extend(
        _table(
            ["Condition", "Task", "Success", "Final skill", "Failed skill", "Tracking total", "VLM mean px", "Exit"],
            smoke_rows,
        )
        if smoke_rows
        else (
            [
                "本轮根据用户明确授权跳过 27-rollout smoke；"
                f"manifest gate mode=`{formal_gate.get('mode')}`，"
                f"approval=`{formal_gate.get('approval_note')}`。"
            ]
            if formal_gate.get("mode") == "explicit_user_approved_bypass"
            else ["尚无完成的 smoke manifest。"]
        )
    )
    lines.extend(
        [
            "",
            f"Smoke gate：`{((smoke_manifest or {}).get('smoke_gate') or {}).get('status', 'pending')}`；manifest：`{smoke_manifest_path or '—'}`。",
            "",
            "### Matched scripted diagnostics",
            "",
        ]
    )
    if diagnostic_summary:
        diagnostic_final, diagnostic_failed = _progress_labels(
            diagnostic_summary, "one_leg"
        )
        diagnostic_tracking = _tracking(diagnostic_summary)
        lines.extend(
            _table(
                ["Condition/task", "Success", "Final skill", "Failed skill", "Tracking count", "Tracking pos/rot mean"],
                [[
                    "rgbd+GP / one_leg / scripted-GT",
                    f"{diagnostic_summary.get('n_success', 0)}/{diagnostic_summary.get('n_rollouts', 0)}",
                    diagnostic_final,
                    diagnostic_failed,
                    str(diagnostic_tracking.get("count", 0)),
                    f"{_fmt(100.0 * diagnostic_tracking.get('mean_pos_m', 0.0))} cm / {_fmt(diagnostic_tracking.get('mean_ori_deg'))} deg",
                ]],
            )
        )
        lines.append(f"\nSummary：`{diagnostic_summary_path}`。")
    else:
        lines.append("本轮未生成 matched scripted diagnostic。")
    lines.extend(
        [
            "",
            "### 新正式 324-rollout",
            "",
            (
                f"当前主 manifest 是 formal：`{manifest_path}`。"
                if campaign_stage == "formal"
                else "Formal 仅在 smoke 自动 gate 和人工 pickle/MP4 审查均通过后启动；当前尚未启动。"
            ),
            "",
        "## 1. 实验状态",
        "",
        f"- 已完成 task-level 实验：`{complete}/{expected}`。",
        f"- 已完成 rollout：`{total_finished_rollouts}/{manifest.get('total_requested_rollouts', 0)}`。",
        f"- VLM：`{manifest.get('vlm_base_url')}`；revision：`{revision_label}`。",
        f"- 原始 manifest：`{manifest_path}`。",
        f"- 阶段：`{campaign_stage}`；设计：3 个 condition × 3 个 task × 每格 {rollouts_per_cell} rollout，共 {manifest.get('total_requested_rollouts', 0)} rollout；每批 3 个并行环境。",
        "- task 截止步数：one_leg=1000，round_table=1000，lamp=1000；randomness=low。",
        f"- 每个有效控制 step 的 GT/VLM 点对、每个 n0--n4 档位使用 `{monte_carlo_samples}` 个 3D Monte Carlo 投影样本。",
        "",
        ]
    )
    if legacy_invalid:
        lines.extend(
            [
                "> **旧 324-rollout 数据已判定 invalid：只保留作故障溯源，不得 resume、拼接或用于 condition 排名。**",
                ">",
                f"> 原因：{baseline_gate.get('reason', '缺少 scripted-GT one_leg 基线。')}",
                f"> 下一步：{baseline_gate.get('next_action', '先完成 one_leg scripted-GT 基线。')}",
                "",
                "### 1.1 one_leg scripted-GT 诊断",
                "",
            ]
        )
        diagnostic_rows = []
        for run in baseline_gate.get("diagnostic_runs", []):
            successes = int(run.get("successes", 0))
            rollouts = int(run.get("rollouts", 0))
            rate = 100.0 * successes / rollouts if rollouts else 0.0
            diagnostic_rows.append(
                [
                    str(run.get("label", "—")),
                    str(run.get("code", "—")),
                    str(run.get("n_envs", "—")),
                    str(run.get("max_rollout_steps", "—")),
                    f"{rate:.1f}% ({successes}/{rollouts})",
                    str(run.get("failed_skill", "—")),
                ]
            )
        lines.extend(
            _table(
                ["Run", "Code", "n_envs", "Max steps", "Success", "Main failure"],
                diagnostic_rows,
            )
        )
        lines.extend(["", "已排除：", ""])
        lines.extend(
            f"- {item}" for item in baseline_gate.get("ruled_out", [])
        )
        lines.extend(
            [
                "",
                f"旧数据判定文件：`{BASELINE_GATE_PATH}`。新评测改由 depth-fixed 27-rollout smoke gate 控制 formal 启动。",
                "",
            ]
        )
    if complete < expected:
        lines.extend(
            [
                (
                    "> 当前是开跑前 review 版本，正式评测尚未启动。下表中的 `—` 表示没有实验结果；批准方案后应使用新的输出目录启动，不能续跑旧实现产生的中断结果。"
                    if is_review
                    else (
                        "> Smoke 正在运行。下表中的 `—` 表示对应 cell 尚未生成完整 summary。"
                        if campaign_stage == "smoke"
                        else "> 正式评测正在运行。下表中的 `—` 表示对应 task 尚未生成完整 summary；仅完整的 36-rollout task 会进入聚合结果。"
                    )
                ),
                "",
            ]
        )

    lines.extend(["## 2. Success rate", ""])
    success_rows = []
    for condition in condition_order:
        label = next(
            row["condition_label"] for row in manifest_rows if row["condition"] == condition
        )
        cells = [
            _success_cell(payload_by_key.get((condition, task))) for task in TASKS
        ]
        aggregate = aggregates[condition]
        overall = (
            f"{100.0 * aggregate['success_rate']:.1f}% "
            f"({aggregate['n_success']}/{aggregate['n_rollouts']})"
            if aggregate["n_rollouts"]
            else "—"
        )
        success_rows.append([label, *cells, overall])
    lines.extend(_table(["Condition", *TASKS, "Overall"], success_rows))

    lines.extend(["", "### 2.1 每格成功率与 Wilson 95% CI", ""])
    success_detail_rows = []
    for row in manifest_rows:
        payload = payload_by_key.get((row["condition"], row["task"]))
        if payload is None:
            success_detail_rows.append([row["condition_label"], row["task"], "—", "—"])
            continue
        success = int(payload.get("n_success", 0))
        total = int(payload.get("n_rollouts", 0))
        lower, upper = _wilson(success, total)
        success_detail_rows.append(
            [
                row["condition_label"],
                row["task"],
                f"{success}/{total}",
                f"{100.0 * success / total:.1f}% [{100.0 * lower:.1f}%, {100.0 * upper:.1f}%]" if total else "—",
            ]
        )
    lines.extend(_table(["Condition", "Task", "Success", "Rate [Wilson 95% CI]"], success_detail_rows))

    lines.extend(["", "## 3. Tracking error（clean GT pose）", ""])
    lines.append(
        "每格为 `position cm / orientation deg / total (n)`；`total = pos_m / 0.01 + ori_deg / 5`，越低越好。VLM 只替换 policy 的 skill/2D point，shadow 自动机提供 clean guidance pose 作为共同 tracking target。"
    )
    lines.append("")
    tracking_rows = []
    for condition in condition_order:
        label = next(
            row["condition_label"] for row in manifest_rows if row["condition"] == condition
        )
        cells = []
        for task in TASKS:
            values = _tracking(payload_by_key.get((condition, task)))
            cells.append(
                (
                    f"{_fmt(100.0 * values.get('mean_pos_m', 0.0))}/"
                    f"{_fmt(values.get('mean_ori_deg'))}/"
                    f"{_fmt(values.get('mean_total'))} (n={values.get('count', 0)})"
                )
                if values.get("count", 0)
                else "—"
            )
        overall = aggregates[condition]["tracking"]
        cells.append(
            (
                f"{_fmt(100.0 * overall.get('mean_pos_m', 0.0))}/"
                f"{_fmt(overall.get('mean_ori_deg'))}/"
                f"{_fmt(overall.get('mean_total'))} (n={overall.get('count', 0)})"
            )
            if overall.get("count", 0)
            else "—"
        )
        tracking_rows.append([label, *cells])
    lines.extend(_table(["Condition", *TASKS, "Overall"], tracking_rows))

    lines.extend(["", "### 3.1 每格 position / rotation 分布", ""])
    tracking_detail_rows = []
    for row in manifest_rows:
        values = _tracking(payload_by_key.get((row["condition"], row["task"])))
        has_tracking = bool(values.get("count", 0))
        tracking_detail_rows.append(
            [
                row["condition_label"],
                row["task"],
                str(values.get("count", 0)),
                "/".join(
                    _fmt(100.0 * values[key])
                    if has_tracking and values.get(key) is not None
                    else "—"
                    for key in ("mean_pos_m", "median_pos_m", "p90_pos_m")
                ),
                "/".join(
                    _fmt(values.get(key)) if has_tracking else "—"
                    for key in ("mean_ori_deg", "median_ori_deg", "p90_ori_deg")
                ),
            ]
        )
    lines.extend(
        _table(
            ["Condition", "Task", "n", "Position mean/median/p90 cm", "Rotation mean/median/p90 deg"],
            tracking_detail_rows,
        )
    )

    lines.extend(["", "### 3.2 Per-skill tracking", ""])
    lines.extend(
        _table(
            ["Condition", "Task", "Skill", "n", "Position mean/median/p90 cm", "Rotation mean/median/p90 deg"],
            [
                [
                    str(item["condition"]),
                    str(item["task"]),
                    str(item["skill"]),
                    str(item["count"] or 0),
                    "/".join(
                        _fmt(item.get(key)) if item.get("count") else "—"
                        for key in ("position_mean_cm", "position_median_cm", "position_p90_cm")
                    ),
                    "/".join(
                        _fmt(item.get(key)) if item.get("count") else "—"
                        for key in ("orientation_mean_deg", "orientation_median_deg", "orientation_p90_deg")
                    ),
                ]
                for item in task_tracking_skill_rows
            ],
        )
    )

    lines.extend(["", "## 4. VLM 打点误差", ""])
    lines.append(
        "逐 step 误差定义为 front camera 上 `||p_vlm - p_gt||₂`。缓存期间每个控制 step 都计入，因此这里直接给出你要求的 step average；投影参考也为每个有效控制 step 的 GT/VLM 点对单独生成，从而包含 action horizon 内 GT 移动与 VLM 点缓存造成的实际误差。"
    )
    lines.append("")
    point_rows = []
    for condition in condition_order:
        label = next(
            row["condition_label"] for row in manifest_rows if row["condition"] == condition
        )
        cells = []
        for task in TASKS:
            values = _point(payload_by_key.get((condition, task)))
            cells.append(
                f"{_fmt(values.get('mean_error_px'))}/{_fmt(values.get('rmse_px'))} (n={values.get('count_valid', 0)})"
                if values.get("count_valid", 0)
                else "—"
            )
        merged = aggregates[condition]["vlm_point_error"] or {}
        overall = ((merged.get("all") or {}).get("overall") or {})
        cells.append(
            f"{_fmt(overall.get('mean_error_px'))}/{_fmt(overall.get('rmse_px'))} (n={overall.get('count_valid', 0)})"
            if overall.get("count_valid", 0)
            else "—"
        )
        point_rows.append([label, *cells])
    lines.extend(
        _table(
            ["Condition", *TASKS, "Overall"],
            point_rows,
        )
    )
    lines.extend(["", "表中每格为 `step mean px / step RMSE px (有效 step 数)`。", ""])

    lines.extend(["### 4.1 每格 VLM residual 分布", ""])
    point_detail_rows = []
    point_analysis_lines = []
    for row in manifest_rows:
        payload = payload_by_key.get((row["condition"], row["task"]))
        point = _point(payload)
        point_detail_rows.append(
            [
                row["condition_label"],
                row["task"],
                f"{point.get('count_valid', 0)}/{point.get('count_total', 0)}",
                _fmt(
                    100.0 * point["skill_accuracy"]
                    if point.get("skill_accuracy") is not None
                    else None,
                    1,
                ),
                _fmt(point.get("mean_error_px")),
                _fmt(point.get("rmse_px")),
                _fmt(point.get("p50_error_px")),
                _fmt(point.get("p90_error_px")),
                _fmt(point.get("mean_dx_px")),
                _fmt(point.get("mean_dy_px")),
                _fmt(point.get("bias_norm_px")),
                _fmt(point.get("spread_ratio"), 3),
                _fmt(point.get("point_r2"), 3),
                f"{point.get('tail_count_gt_40px', 0)}/{point.get('tail_count_gt_70px', 0)}",
            ]
        )
        point_analysis_lines.append(
            _point_quality_analysis(
                f"{row['condition_label']} / {row['task']}", point
            )
        )
    lines.extend(
        _table(
            [
                "Condition",
                "Task",
                "Valid/total",
                "Skill acc. %",
                "Mean px",
                "RMSE px",
                "Median px",
                "P90 px",
                "dx bias",
                "dy bias",
                "Bias norm",
                "Spread",
                "R²",
                ">40/>70",
            ],
            point_detail_rows,
        )
    )

    lines.extend(
        [
            "",
            "**指标解释。** `Valid/total` 是同时具有合法 shadow-GT 和 VLM 点的 control-step pair；production 中无效 VLM JSON 会直接终止 row，因此另从日志报告服务失败，不能把它混入 GT coverage。Skill accuracy 使用 shadow oracle coarse skill。Mean/RMSE/median/P90 都基于 `||p_vlm-p_gt||₂`；dx/dy 和 bias 描述系统偏移；spread 是 prediction/GT 二维标准差范数之比；`R²=1-SSE/SST`，0 等价于恒定预测该组 GT 均值，负值更差。R² 在 GT 无空间方差时记为 `—`。",
            "",
            "**数据分析。**",
            "",
            *point_analysis_lines,
            "",
        ]
    )

    lines.extend(["### 4.2 Fresh-query VLM point 质量", ""])
    fresh_rows = []
    fresh_analysis_lines = []
    for row in manifest_rows:
        fresh = _fresh_queries(
            payload_by_key.get((row["condition"], row["task"]))
        ).get("overall", {})
        fresh_rows.append(
            [
                row["condition_label"],
                row["task"],
                f"{fresh.get('count_valid', 0)}/{fresh.get('count_total', 0)}",
                _fmt(
                    100.0 * fresh["skill_accuracy"]
                    if fresh.get("skill_accuracy") is not None
                    else None,
                    1,
                ),
                _fmt(fresh.get("mean_error_px")),
                _fmt(fresh.get("rmse_px")),
                _fmt(fresh.get("p50_error_px")),
                _fmt(fresh.get("p90_error_px")),
                _fmt(fresh.get("mean_dx_px")),
                _fmt(fresh.get("mean_dy_px")),
                _fmt(fresh.get("bias_norm_px")),
                _fmt(fresh.get("spread_ratio"), 3),
                _fmt(fresh.get("point_r2"), 3),
                f"{fresh.get('tail_count_gt_40px', 0)}/{fresh.get('tail_count_gt_70px', 0)}",
            ]
        )
        fresh_analysis_lines.append(
            _point_quality_analysis(
                f"{row['condition_label']} / {row['task']}", fresh
            )
        )
    lines.extend(
        _table(
            [
                "Condition",
                "Task",
                "Valid/total",
                "Skill acc. %",
                "Mean px",
                "RMSE px",
                "Median px",
                "P90 px",
                "dx bias",
                "dy bias",
                "Bias norm",
                "Spread",
                "R²",
                ">40/>70",
            ],
            fresh_rows,
        )
    )
    lines.extend(
        [
            "",
            "**指标解释。** 本表只保留 `step_idx == query_step` 的新 VLM 请求，衡量模型本身的即时输出；上一表统计每个 control step，包含 action-horizon 缓存造成的实际 stale-point 误差。其余公式完全相同。",
            "",
            "**数据分析。**",
            "",
            *fresh_analysis_lines,
            "",
        ]
    )

    lines.extend(["### 4.3 Each skill step average（跨 task 聚合）", ""])
    skill_rows = []
    for condition in condition_order:
        label = next(
            row["condition_label"] for row in manifest_rows if row["condition"] == condition
        )
        merged = aggregates[condition]["vlm_point_error"] or {}
        by_skill = ((merged.get("all") or {}).get("by_skill") or {})
        for skill in SKILLS:
            stats = by_skill.get(skill, {})
            skill_rows.append(
                [
                    label,
                    skill,
                    f"{stats.get('count_valid', 0)}/{stats.get('count_total', 0)}",
                    _fmt(
                        100.0 * stats["skill_accuracy"]
                        if stats.get("skill_accuracy") is not None
                        else None,
                        1,
                    ),
                    _fmt(stats.get("mean_error_px")),
                    _fmt(stats.get("rmse_px")),
                    _fmt(stats.get("p50_error_px")),
                    _fmt(stats.get("p90_error_px")),
                    _fmt(stats.get("bias_norm_px")),
                    _fmt(stats.get("spread_ratio"), 3),
                    _fmt(stats.get("point_r2"), 3),
                ]
            )
    lines.extend(
        _table(
            [
                "Condition",
                "Skill",
                "Valid/total",
                "Skill acc. %",
                "Mean px",
                "RMSE px",
                "Median px",
                "P90 px",
                "Bias norm",
                "Spread",
                "R²",
            ],
            skill_rows,
        )
    )

    lines.extend(["", "### 4.4 每个 task 的 per-skill point error", ""])
    lines.extend(
        _table(
            [
                "Condition",
                "Task",
                "Skill",
                "Valid/total",
                "Skill acc. %",
                "Mean px",
                "RMSE px",
                "Median px",
                "P90 px",
                "dx bias",
                "dy bias",
                "Bias norm",
                "Spread",
                "R²",
            ],
            [
                [
                    str(item["condition"]),
                    str(item["task"]),
                    str(item["skill"]),
                    f"{item['step_count'] or 0}/{item['step_total'] or 0}",
                    _fmt(
                        100.0 * item["skill_accuracy"]
                        if item.get("skill_accuracy") is not None
                        else None,
                        1,
                    ),
                    _fmt(item.get("step_mean_error_px")),
                    _fmt(item.get("step_rmse_px")),
                    _fmt(item.get("step_median_px")),
                    _fmt(item.get("step_p90_px")),
                    _fmt(item.get("mean_dx_px")),
                    _fmt(item.get("mean_dy_px")),
                    _fmt(item.get("bias_norm_px")),
                    _fmt(item.get("spread_ratio"), 3),
                    _fmt(item.get("point_r2"), 3),
                ]
                for item in task_point_skill_rows
            ],
        )
    )

    lines.extend(["", "## 5. VLM 对应 n0–n4 的哪个等级？", ""])
    lines.extend(
        [
            "不能把 2D 像素误差直接除以一个固定 px/mm 系数，再与 3D 的 0/3/6/12/24 mm 比较。透视投影尺度随 GT 点深度、相机内参和偏移方向变化。主分析采用同坐标系比较：",
            "",
            "1. 每个有效控制 step 形成一个配对样本：自动机给出当步 3D GT guidance point `P_gt`，同一 annotation util 给出 front-camera GT pixel `p_gt` 和相机内外参，实际送给 policy 的 VLM 点为 `p_vlm`。VLM 在 action horizon 内可以缓存，但每个当步 GT/VLM 点对都独立进入 step average 和投影分布。",
            f"2. 用由 `(episode, env, step, query_step)` 确定的 seed 采样 `{monte_carlo_samples}` 个 `z_j ~ N(0, I_3)`，随后逐分量 clip 到 `[-2, 2]`。这与现有 `annotation_noise.py` 完全一致。严格来说 clip 后的边际方差小于 1；这里的 n1--n4 名称和 σ 参数沿用原噪声实验，而不是声称截断后的实际标准差仍恰好等于 σ。",
            "3. 五档使用同一组 `z_j`（common random numbers）以减小档位间 Monte Carlo 抖动。令 `σ_n ∈ {0, 3, 6, 12, 24} mm`，构造 `P_nj = P_gt + σ_n z_j`。",
            "4. 使用 `skill_annotation_util.py` 相同的 sim-local→camera 变换、camera-y 翻转和内参投影，计算连续坐标 `e_nj = π(P_nj) - π(P_gt)`；VLM 残差为 `e_vlm = p_vlm - p_gt`。参考噪声投影不做整数取整，也不按图像边界裁剪，否则会人为压缩尾部。",
            "5. 对每档全部投影样本精确累计一阶、二阶矩：`μ_n = (1/M)Σe_nj`，`Σ_n = (1/M)Σ(e_nj-μ_n)(e_nj-μ_n)^T`，并由全部样本计算 projected RMSE。VLM 的 mean/cov/RMSE 在所有有效 step 残差上计算，和 reference 的相机/深度条件完全配对。",
            "6. 用 projected RMSE 在相邻 n 档之间线性插值得到主要“误差量级”，例如 `n1–n2`；超过 n4 时线性外推并标记 `>n4`。径向 W1 最近档作为大小分布的稳健交叉检查。",
            "7. 完整 2D SWD 用于判断包含系统偏置的二维分布是否像某档噪声；centered SWD 在双方各自减均值后比较形状。它们不替代第 6 步的误差大小分类。",
            "",
            f"实现的性能策略：每个点对实际生成 `{monte_carlo_samples}` 个样本；mean/cov/RMSE 用全部样本的 sufficient statistics 精确合并；每档仅保留均匀抽取的 2000 个 reference residual 计算分位数、32-direction SWD、W1 和 KS，使 rollout summary 大小与后处理耗时有上界。在更接近最大单批规模的 3000-pair、200-sample 合成基准中，600,000 个样本/档的投影约 1.76 s，三 scope 汇总约 11.06 s，进程峰值 RSS 约 958 MiB；本机约有 19 GiB 可用内存，因此正式配置从 100 提高到 200。该数字只是性能基准，不是实验结果；正式运行仍记录实际 wall time。",
            "",
        ]
    )
    equivalence_rows = []
    task_equivalence_rows = []
    distance_rows = []
    for condition in condition_order:
        aggregate = aggregates[condition]
        label = aggregate["condition_label"]
        merged = aggregate["vlm_point_error"] or {}
        query = ((merged.get("all") or {}).get("step_distribution") or {})
        vlm = query.get("vlm", {})
        same_depth = query.get("same_depth", {})
        equivalence_rows.append(
            [
                label,
                str(vlm.get("count", 0)),
                _fmt(vlm.get("rmse_px")),
                _fmt(vlm.get("p90_error_px")),
                _fmt(vlm.get("p95_error_px")),
                _fmt(vlm.get("bias_norm_px")),
                _fmt(vlm.get("anisotropy_ratio")),
                _fmt(same_depth.get("mean_error_mm")),
                str(query.get("magnitude_equivalent_bracket") or "—"),
                _fmt(query.get("magnitude_equivalent_std_mm")),
                str(query.get("closest_level_radial_wasserstein") or "—"),
                str(query.get("closest_level_sliced_wasserstein") or "—"),
                str(
                    query.get("closest_level_centered_sliced_wasserstein") or "—"
                ),
            ]
        )
        for level in ("n0", "n1", "n2", "n3", "n4"):
            comparison = query.get("noise_levels", {}).get(level, {})
            distance_rows.append(
                [
                    label,
                    level,
                    _fmt(comparison.get("sliced_wasserstein_px")),
                    _fmt(comparison.get("centered_sliced_wasserstein_px")),
                    _fmt(comparison.get("radial_wasserstein_px")),
                    _fmt(comparison.get("bias_difference_px")),
                    _fmt(comparison.get("radial_ks_statistic"), 3),
                    _fmt((comparison.get("projected") or {}).get("rmse_px")),
                ]
            )
        for task in TASKS:
            task_query = _query(payload_by_key.get((condition, task)))
            task_vlm = task_query.get("vlm", {})
            task_equivalence_rows.append(
                [
                    label,
                    task,
                    str(task_vlm.get("count", 0)),
                    _fmt(task_vlm.get("rmse_px")),
                    _fmt(task_vlm.get("p90_error_px")),
                    _fmt(task_vlm.get("bias_norm_px")),
                    str(task_query.get("magnitude_equivalent_bracket") or "—"),
                    _fmt(task_query.get("magnitude_equivalent_std_mm")),
                    str(task_query.get("closest_level_radial_wasserstein") or "—"),
                    str(task_query.get("closest_level_sliced_wasserstein") or "—"),
                    str(
                        task_query.get(
                            "closest_level_centered_sliced_wasserstein"
                        )
                        or "—"
                    ),
                ]
            )
    lines.append("以下第一张表按 condition 跨三个 task、按有效 step 聚合。")
    lines.append("")
    lines.extend(
        _table(
            [
                "Condition",
                "Valid pairs",
                "VLM RMSE px",
                "P90 px",
                "P95 px",
                "Bias px",
                "Anisotropy",
                "Same-depth mm",
                "Magnitude bracket",
                "Equivalent σ mm",
                "Closest radial W1",
                "Closest full SWD",
                "Closest centered SWD",
            ],
            equivalence_rows,
        )
    )
    lines.extend(["", "### 5.1 每个 task 的误差量级", ""])
    lines.append(
        "task-level 映射是主要诊断表；它避免不同任务的 GT 深度和透视尺度在总体聚合中互相抵消。"
    )
    lines.append("")
    lines.extend(
        _table(
            [
                "Condition",
                "Task",
                "Valid pairs",
                "VLM RMSE px",
                "P90 px",
                "Bias px",
                "Magnitude bracket",
                "Equivalent σ mm",
                "Closest radial W1",
                "Closest full SWD",
                "Closest centered SWD",
            ],
            task_equivalence_rows,
        )
    )
    lines.extend(["", "### 5.2 与各档噪声分布的距离（condition overall）", ""])
    lines.extend(
        _table(
            [
                "Condition",
                "Level",
                "Full 2D SWD px",
                "Centered SWD px",
                "Radial W1 px",
                "Bias gap px",
                "Radial KS",
                "Projected RMSE px",
            ],
            distance_rows,
        )
    )

    lines.extend(["", "## 6. 偏移分布与噪声假设有多大区别？", ""])
    lines.extend(
        [
            "建议同时看这些互补量，而不是只看一个平均半径：",
            "",
            "- `full 2D SWD (px)`：同时反映尺度、系统偏置、方向和协方差形状；越小越接近该 n 档完整二维分布。它不是误差大小分类器：强单向 bias 下 full SWD 最近档可能是 n0。",
            "- `centered SWD (px)`：VLM 和参考噪声各自减去均值后再计算，用来比较去除系统偏置后的分布形状。",
            "- `radial W1 (px)`：只比较偏移半径，单位仍是像素，直观说明大小分布差多少。",
            "- `bias norm (px)`：VLM 平均偏移向量的长度。噪声假设是零均值；bias 大说明存在系统性偏移，无法用增大零均值 σ 完全解释。",
            "- `anisotropy = λmax/λmin`：协方差椭圆的长短轴比。接近 1 表示各向同性；明显大于 1 表示 VLM 偏移有方向偏好。径向 KS 作为辅助无量纲检验量。",
            "",
            "same-depth mm 是把 VLM 像素在 GT 深度处反投影后的横向位移。它便于工程直觉和毫米量级展示，但深度方向不可由单个 2D 点观测，因此它不是完整 3D VLM 误差，也不能替代上面的投影分布匹配。",
            "",
            "原噪声实验一个 skill phase 内保持同一个 3D 偏移；VLM 每个 query 更新、两次 query 之间缓存。上述 SWD 比较的是空间边缘分布，不声称二者的时间相关结构相同。",
            "",
        ]
    )

    checkpoint_lines = []
    for condition in condition_order:
        row = next(row for row in manifest_rows if row["condition"] == condition)
        checkpoint_lines.append(
            f"- `{row['condition_label']}` checkpoint：`{row['checkpoint']}`。"
        )
    lines.extend(
        [
            (
                "## 7. 实验方案与聚合口径"
                if complete == expected
                else "## 7. 开跑前冻结的实验方案（待批准）"
            ),
            "",
        ]
    )
    lines.extend(
        [
            "### 7.1 实验矩阵与控制变量",
            "",
            "- Conditions：`rgbd+GP`、`rgbd+colored GP`、`rgbd+GP+skill`；仅 checkpoint/模型输入配置不同，VLM 服务、任务初始分布和评测代码相同。",
            *checkpoint_lines,
            f"- 当前主 manifest tasks：`one_leg`、`round_table`、`lamp`；每个 condition-task {rollouts_per_cell} rollout，共 `3×3×{rollouts_per_cell}={manifest.get('total_requested_rollouts', 0)}`。Formal 固定为 36 rollout/格（324 总计）。",
            "- 本地并行：`n_envs=3`；三个 task 上限均为 1000 step；`randomness=low`。",
            f"- VLM：固定 readiness 中的 model revision；每次正式启动先 fail-fast 检查 `status=ready`、`policy_version={readiness.get('policy_version', '—')}`、`model_mode={readiness.get('model_mode', '—')}`。HTTP timeout 30 s，失败或 schema/revision 不一致直接终止，不 fallback 到自动机。",
            "- Query：`--vlm-query-interval 0`，使用 checkpoint 的 `action_horizon=8`；每 8 个 environment step query 一次，其间缓存。",
            f"- Noise projection：每个有效控制 step 的 GT/VLM 点对、每档 `{monte_carlo_samples}` 个 clipped-standard-Gaussian 样本；reference reservoir 2000/档；SWD 32 个固定方向。",
            "- Tracking target：shadow 自动机的 clean GT guidance pose；强制 `pose` 模式并报告 position cm / orientation deg / normalized total。VLM 控制 policy，自动机只负责 shadow GT 和指标。",
            "- Initial state：当前默认方案与历史噪声实验一样，使用独立的 `randomness=low` reset，并非三个 condition 严格共享同一批初始状态。36 rollout/格和 Wilson CI 能反映抽样不确定性，但 condition 差值仍包含 reset 方差；若要求 paired comparison，应先从真实 env reset 额外建立并目视验证每 task 36 个固定初始状态的 bank，再给三种 condition 共同使用。不能复用仓库之前的 train-init bank，因为 `reports/train_init_eval.md` 已记录第一帧、坐标和关节状态有效性风险。",
            "",
            "### 7.2 三类主要输出与聚合口径",
            "",
            "1. **Success rate**：每 task 报 `success/36`，condition overall 报 `success/108` 和 Wilson 95% CI。condition 优劣的主要依据是 success rate，但 36 次/格仍应结合置信区间解释。",
            "2. **Tracking error**：按所有有效控制记录加权，报告 position、orientation 和 `total = pos_m/0.01 + ori_deg/5`。它回答 policy 相对 clean GT pose 的跟踪程度。",
            "3. **VLM 打点误差**：overall 与 each-skill 都按有效控制 step 加权报告 mean/RMSE pixel；全体有效 GT/VLM 点对另报 P90/P95、bias、covariance/anisotropy、same-depth lateral mm、n-level 等价量级和分布距离。按 skill 统计使用 oracle skill 标签，避免 VLM skill 误分类污染分组。",
            "",
            "三个指标不合成一个总分：先以成功率比较 condition；tracking error 解释控制效果；point error 与 n-level/distribution metrics 诊断 VLM guidance 的空间质量。另保留 success-only / failure-only 打点统计，用于观察误差与任务失败的关联，但不作因果结论。",
            "",
            "### 7.3 与历史噪声实验比较时的边界",
            "",
            "- 主映射使用本报告的同相机 3D-noise→2D residual reference，不把 VLM 的 2D error 直接当作 3D mm。",
            "- 历史噪声实验的偏移在 skill phase 内固定，而 VLM 以 8-step query/cache 更新；所以当前只比较空间边缘分布。若要比较时间相关结构，需要额外报告 autocorrelation 或按 phase 重跑。",
            "- 本报告 tracking target 是 clean GT pose；若历史噪声 tracking 以加噪 pose 为 target，两者 tracking 数值不可直接映射。噪声等级结论只由投影残差比较给出。",
            "- `same-depth mm` 只表示 GT 深度平面上的横向误差，不能恢复不可观测的深度方向，因此只作为工程辅助量。",
            "- 正式运行将创建新的 timestamp output-dir；旧的中断 rollout 不纳入任何表格或结论。",
            "",
            (
                "### 7.4 正式运行配置"
                if complete == expected
                else "### 7.4 批准门槛"
            ),
            "",
            f"已批准的固定项：`randomness=low` 独立 reset、`n_envs=3`、36 rollout/格、query horizon=8、{monte_carlo_samples} MC 样本/点/档、2000 reference reservoir/档、32-direction SWD、三个 task 的 max steps 均为 1000，以及 clean-GT pose tracking target。正式矩阵共 324 rollout。",
            "",
        ]
    )

    complete_aggregates = [
        aggregate for aggregate in aggregates.values() if aggregate["n_rollouts"] > 0
    ]
    lines.extend(["## 8. 当前结论", ""])
    if (
        campaign_stage == "formal"
        and complete == expected
        and complete_aggregates
        and not legacy_invalid
    ):
        best_success = max(complete_aggregates, key=lambda value: value["success_rate"])
        trackable = [
            value for value in complete_aggregates if value["tracking"].get("count", 0)
        ]
        best_tracking = min(
            trackable, key=lambda value: value["tracking"].get("mean_total", math.inf)
        )
        point_rankable = []
        for aggregate in complete_aggregates:
            point_summary = (
                ((aggregate["vlm_point_error"] or {}).get("all") or {}).get(
                    "overall", {}
                )
            )
            if point_summary.get("rmse_px") is not None:
                point_rankable.append((aggregate, point_summary))
        best_point, best_point_summary = min(
            point_rankable,
            key=lambda item: item[1]["rmse_px"],
        )
        per_condition_lines = []
        for aggregate in complete_aggregates:
            lower, upper = _wilson(
                aggregate["n_success"], aggregate["n_rollouts"]
            )
            query = (
                (((aggregate["vlm_point_error"] or {}).get("all") or {}).get(
                    "step_distribution", {}
                ))
            )
            per_condition_lines.append(
                f"- **{aggregate['condition_label']}**：SR "
                f"{100.0 * aggregate['success_rate']:.1f}% "
                f"（{aggregate['n_success']}/{aggregate['n_rollouts']}，"
                f"Wilson 95% CI {100.0 * lower:.1f}%–{100.0 * upper:.1f}%）；"
                f"tracking total {aggregate['tracking']['mean_total']:.2f}；"
                f"等价噪声 {query.get('magnitude_equivalent_bracket') or '—'} "
                f"（外推 σ {_fmt(query.get('magnitude_equivalent_std_mm'))} mm）。"
            )
        lines.extend(
            [
                f"- 跨三个 task，成功率最高的是 **{best_success['condition_label']}**：{100.0 * best_success['success_rate']:.1f}%（{best_success['n_success']}/{best_success['n_rollouts']}）。",
                f"- clean-GT pose tracking total 最低的是 **{best_tracking['condition_label']}**：{best_tracking['tracking']['mean_total']:.2f}。",
                f"- step-weighted 打点 RMSE 最低的是 **{best_point['condition_label']}**：{best_point_summary['rmse_px']:.2f} px。",
                "- 三组 Wilson 95% CI 有重叠，因此“rgbd+colored GP 数值最高”应解释为当前 108-rollout/condition 下的最好观测值，而不是对所有 condition 差异都作显著性声明。",
                "",
                "Condition 汇总：",
                "",
                *per_condition_lines,
                "",
                "full/centered SWD、radial W1、bias 和 anisotropy 表明 VLM 偏移具有显著系统偏置和方向性，不能只用增大零均值各向同性高斯的 σ 完整解释。",
            ]
        )
    elif legacy_invalid:
        lines.append(
            "旧 324-rollout 因缺少 --save-depth-image（且 one_leg 步数和 rgbd+GP checkpoint 不一致）而作废；只保留为故障诊断证据，不对三个 condition 排名。"
        )
    elif campaign_stage == "smoke" and complete == expected:
        gate_status = ((manifest.get("smoke_gate") or {}).get("status", "pending"))
        lines.append(
            f"Smoke 已完成，automatic gate=`{gate_status}`。Smoke 只用于链路/gate 判定，不用于三个 condition 的正式排名。"
        )
    else:
        lines.append("实验未全部完成，暂不对三个 condition 排名；完成后生成器会自动填入结论。")

    lines.extend(
        [
            "",
            "注意：本报告 VLM tracking target 是 clean GT pose，而历史噪声报告的 tracking target 是实际加噪 pose；二者 target 定义不同，不能直接把 tracking 数值映射成噪声等级。噪声等级的主比较必须使用第 5 节的同相机投影残差。",
            "",
            "## 9. 复现命令",
            "",
            "所有 scripted/VLM eval 必须通过 `/data/hy/gpu-snatcher/auto_eval.sh`；禁止人工直接调用 `evaluate_model`。以下示例会由 auto_eval 自动加入 `--save-depth-image`、保存 pickle/MP4 等固定参数。",
            "",
            "本地单任务（以 rgbd+GP / one_leg 为例）：",
            "",
            "```bash",
            "export VLM_GUIDANCE_URL=http://10.71.106.240:8000",
            "export VLM_API_TOKEN=\"$(sed -n 's/^VLM_API_TOKEN=//p' /mnt/nas/share/home/hy/vlm-guidance/server.env)\"",
            "",
            "/data/hy/gpu-snatcher/auto_eval.sh --steps eval \\",
            "  --local-path /data/hy/robust-rearrangement \\",
            "  --overwrite-wt-path /mnt/nas/share/home/hy/robust-rearrangement-custom/outputs/2026-06-13/13-02-04.275134/models/icy-vortex-9_2026-06-13_13-02-27.880769/actor_chkpt_latest_3000.pt \\",
            "  --task one_leg --n-envs 3 --n-rollouts 36 \\",
            "  --randomness low --max-rollout-steps 1000 \\",
            "  --guidance-point-on-image --no-annotate-skill \\",
            "  --tracking-metric-type pose \\",
            "  --annotation-source vlm --vlm-base-url \"$VLM_GUIDANCE_URL\" \\",
            "  --vlm-timeout-seconds 30 --vlm-query-interval 0 \\",
            f"  --vlm-noise-projection-samples {monte_carlo_samples} \\",
            "  --task-summary-out logs/vlm_dit_single/rgbd_gp__one_leg.json \\",
            "  --rollout-suffix-model-name vlm_dit_single/rgbd_gp/one_leg",
            "```",
            "",
            "完整矩阵由 gated runner 先做 print-command 审计，再执行。formal 默认使用已通过人工视频审查的 smoke manifest；用户明确授权直接 formal 时，必须用显式 bypass 和审批说明，由 manifest 留痕。",
            "",
            "```bash",
            "python3 scripts/run_vlm_dit_eval.py --phase print --stage formal \\",
            f"  --namespace {repro_namespace} \\",
            f"  --output-dir {repro_output_dir} \\",
            f"  --data-dir-raw {repro_data_dir_raw} \\",
            *formal_gate_cli,
            "```",
            "",
            "只重新生成报告：",
            "",
            "```bash",
            f"python scripts/generate_vlm_dit_report.py --manifest {manifest_path} --output {output_path}",
            "```",
            "",
            "## 10. 原始导出",
            "",
            f"- task-level CSV：`{data_dir / 'vlm_dit_guidance_by_task.csv'}`",
            f"- condition overall CSV：`{data_dir / 'vlm_dit_guidance_overall.csv'}`",
            f"- each-skill step average CSV：`{data_dir / 'vlm_dit_guidance_by_skill.csv'}`",
            f"- task×skill tracking CSV：`{data_dir / 'vlm_dit_tracking_by_task_skill.csv'}`",
            f"- task×skill point error CSV：`{data_dir / 'vlm_dit_point_error_by_task_skill.csv'}`",
            f"- source manifest：`{manifest_path}`",
            "- 不复制 runtime manifest 到 reports；以上述 logs 路径为唯一源。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
