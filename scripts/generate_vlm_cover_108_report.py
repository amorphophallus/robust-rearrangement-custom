from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
import matplotlib.pyplot as plt

from src.eval.annotation_noise import build_annotation_noise_summary


TASKS = ("one_leg", "round_table", "lamp")
CONDITIONS = (
    ("gp", "rgbd+GP", "point"),
    ("colored_gp", "rgbd+colored GP", "point"),
    ("gp_skill", "rgbd+GP+skill", "point"),
    ("grasp_part", "rgbd+grasp-part", "grasp-part"),
    ("grasp_part_colored", "rgbd+grasp-part-colored", "grasp-part"),
)
NUMERIC_IDS = tuple(f"n{idx}" for idx in range(8))
LEGACY_IDS = {*NUMERIC_IDS[:5], "shuffle"}
POSITION_MM = dict(zip(NUMERIC_IDS, (0, 3, 6, 12, 24, 48, 96, 192)))
ORIENTATION_DEG = dict(
    zip(NUMERIC_IDS, (0.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 90.0))
)
VLM_SIGMA_TASK_STYLES = {
    "one_leg": "-",
    "round_table": "--",
    "lamp": ":",
}
VLM_SIGMA_POOLED_STYLE = "-"
SKILL_TYPES = ("push", "pick", "place", "insert", "screw")
VLM_FAMILY_LABELS = {
    "point": "Point VLM",
    "grasp": "Grasp VLM",
}
VLM_FAMILY_CONDITIONS = {
    "point": ("gp", "colored_gp", "gp_skill"),
    "grasp": ("grasp_part", "grasp_part_colored"),
}

# Task-level VLM point-error reference from the formal diagnostic summary.
# The VLM family is assigned below: the three point conditions share one VLM
# point output, while the two grasp conditions share a separate VLM output.
# Each condition/task cell contains 36 formal VLM trajectories.
VLM_SIGMA_REFERENCE = (
    ("gp", "one_leg", 15155, 36.34),
    ("gp", "round_table", 30884, 61.25),
    ("gp", "lamp", 24789, 81.93),
    ("colored_gp", "one_leg", 15107, 41.07),
    ("colored_gp", "round_table", 30295, 69.68),
    ("colored_gp", "lamp", 24137, 66.20),
    ("gp_skill", "one_leg", 15293, 31.09),
    ("gp_skill", "round_table", 32442, 86.87),
    ("gp_skill", "lamp", 24974, 56.91),
    ("grasp_part", "one_leg", 4694, 176.02),
    ("grasp_part", "round_table", 29291, 98.81),
    ("grasp_part", "lamp", 18886, 97.41),
    ("grasp_part_colored", "one_leg", 7474, 166.37),
    ("grasp_part_colored", "round_table", 29335, 101.74),
    ("grasp_part_colored", "lamp", 20606, 95.21),
)


def _save_figure_bundle(fig, path: Path) -> None:
    """Export a PNG preview plus editable vector companions."""
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    annotation_seed = row.get("annotation_seed", row.get("shuffle_seed", 0))
    return (
        str(row.get("condition_id")),
        str(row.get("noise_id")),
        int(row.get("replicate_id", 0)),
        int(row.get("simulator_seed", 0)),
        int(annotation_seed or 0),
    )


def _load_summary(row: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(row.get("summary_json", ""))
    path = Path(raw_path)
    if not path.is_file():
        base_repo_prefix = "/home/huyue/projects/robust-rearrangement-custom/"
        if raw_path.startswith(base_repo_prefix):
            path = Path(raw_path[len(base_repo_prefix):])
    if not path.is_file():
        raise ValueError(f"missing summary for {_key(row)}: {path}")
    return json.loads(path.read_text())


def _wilson(successes: int, count: int) -> tuple[float, float]:
    if count <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * count)) / count) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_TABLE_INT_FIELDS = {
    "annotation_seed",
    "completed",
    "condition_count",
    "diagnostics_rollout_n",
    "entered",
    "n_rollouts",
    "n_success",
    "phase_count",
    "replicate_id",
    "source_rollouts",
    "simulator_seed",
    "tracking_n",
    "tracking_rollout_count",
    "tracking_state_count",
    "tail_gt_40_count",
    "tail_gt_70_count",
    "tail_gt_100_count",
    "valid_pairs",
}
_TABLE_FLOAT_FIELDS = {
    "equivalent_p50_mm",
    "equivalent_p95_mm",
    "equivalent_rms_mm",
    "front_projection_visible_rate",
    "invalid_nonfinite_rate",
    "ordinal_intercept",
    "ordinal_slope_per_level",
    "pearson_r_on_ordinal_level",
    "pos_std_mm",
    "position_norm_max_mm",
    "position_norm_mean_mm",
    "position_norm_p50_mm",
    "position_norm_p90_mm",
    "position_norm_rms_mm",
    "position_sigma_mm_per_axis",
    "rotation_geodesic_max_deg",
    "rotation_geodesic_mean_deg",
    "rotation_geodesic_p50_deg",
    "rotation_geodesic_p90_deg",
    "rotation_geodesic_rms_deg",
    "orientation_sigma_deg",
    "orientation_equivalent_sigma_deg",
    "orientation_level",
    "lower_orientation_deg",
    "upper_orientation_deg",
    "position_equivalent_mm",
    "mean_error_px",
    "p50_error_px",
    "p90_error_px",
    "p95_error_px",
    "tracking_error_mean_deg",
    "success_rate",
    "equivalent_sigma_mm",
    "rmse_error_px",
    "tail_gt_40_fraction",
    "tail_gt_70_fraction",
    "tail_gt_100_fraction",
    "track_ori_deg",
    "track_pos_cm",
    "track_total",
    "wilson_high",
    "wilson_low",
    "workspace_valid_rate",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read a generated table while restoring the numeric columns used downstream."""
    if not path.is_file():
        raise ValueError(f"missing generated table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
            elif key == "replicate_id":
                row[key] = value if value == "pooled" else int(value)
            elif key in _TABLE_INT_FIELDS:
                row[key] = int(value)
            elif key in _TABLE_FLOAT_FIELDS:
                row[key] = float(value)
    return rows


def _validate_table_products(
    replicate_rows: list[dict[str, Any]],
    pooled_rows: list[dict[str, Any]],
    overall_replicate_rows: list[dict[str, Any]],
    vlm_sigma_rows: list[dict[str, Any]] | None = None,
    skill_type_rows: list[dict[str, Any]] | None = None,
    three_task_rows: list[dict[str, Any]] | None = None,
    noise_schedule_rows: list[dict[str, Any]] | None = None,
    vlm_sigma_3task_rows: list[dict[str, Any]] | None = None,
    orientation_equivalent_rows: list[dict[str, Any]] | None = None,
    vlm_skill_error_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Validate the persisted result tables before any figure reads them."""
    checks: list[dict[str, str]] = []
    errors: list[str] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({
            "table": "result tables",
            "check": name,
            "status": "ok" if passed else "error",
            "detail": detail,
        })
        if not passed:
            errors.append(f"{name}: {detail}")

    replicate_keys = [
        (row["condition_id"], row["noise_id"], row["task"], row["replicate_id"])
        for row in replicate_rows
    ]
    record(
        "replicate_unique_keys",
        len(replicate_keys) == len(set(replicate_keys)),
        f"rows={len(replicate_rows)}",
    )
    record(
        "replicate_rollout_count",
        all(row["n_rollouts"] == 36 for row in replicate_rows),
        "every task/replicate cell has n_rollouts=36",
    )
    record(
        "replicate_rate_arithmetic",
        all(math.isclose(
            row["success_rate"],
            row["n_success"] / row["n_rollouts"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) for row in replicate_rows),
        "success_rate equals n_success/n_rollouts",
    )

    replicate_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in replicate_rows:
        replicate_groups[
            (row["condition_id"], row["noise_id"], row["task"])
        ].append(row)
    pooled_index = {
        (row["condition_id"], row["noise_id"], row["task"]): row
        for row in pooled_rows
    }
    pooled_matches = True
    for key, rows in replicate_groups.items():
        pooled = pooled_index.get(key)
        if pooled is None or len(rows) != 3:
            pooled_matches = False
            continue
        successes = sum(row["n_success"] for row in rows)
        rollouts = sum(row["n_rollouts"] for row in rows)
        pooled_matches &= (
            pooled["n_success"] == successes
            and pooled["n_rollouts"] == rollouts == 108
            and math.isclose(
                pooled["success_rate"],
                successes / rollouts,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    record(
        "pooled_matches_replicates",
        pooled_matches,
        f"pooled_rows={len(pooled_rows)}",
    )
    record(
        "pooled_unique_keys",
        len(pooled_index) == len(pooled_rows),
        f"rows={len(pooled_rows)}",
    )
    expected_pooled_keys = {
        (condition_id, noise_id, task)
        for condition_id, _, family in CONDITIONS
        for noise_id in [*NUMERIC_IDS, "shuffle"] + (
            ["r180"] if family == "grasp-part" else []
        )
        for task in TASKS
    }
    record(
        "pooled_expected_coverage",
        set(pooled_index) == expected_pooled_keys,
        f"expected={len(expected_pooled_keys)} actual={len(pooled_index)}",
    )
    expected_positions = {noise_id: float(POSITION_MM[noise_id]) for noise_id in NUMERIC_IDS}
    position_values_ok = all(
        row["noise_id"] not in expected_positions
        or math.isclose(
            float(row["pos_std_mm"]),
            expected_positions[row["noise_id"]],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in pooled_rows
    )
    record(
        "pooled_position_scale",
        position_values_ok,
        "numeric noise IDs retain their table position σ/axis in mm",
    )
    record(
        "overall_replicate_rollout_count",
        all(row["n_rollouts"] == 108 for row in overall_replicate_rows),
        "every condition/noise/replicate cell has n_rollouts=108",
    )
    record(
        "tracking_denominator",
        all(row["tracking_n"] in (72, 108) for row in pooled_rows),
        "tracking_n is 72 for legacy-backed cells or 108 otherwise",
    )
    if vlm_sigma_rows is not None:
        sigma_keys = [
            (row["family"], row["task"])
            for row in vlm_sigma_rows
        ]
        expected_sigma_keys = {
            (family, task)
            for family in VLM_FAMILY_LABELS
            for task in TASKS
        }
        record(
            "vlm_sigma_unique_task_keys",
            len(sigma_keys) == len(set(sigma_keys)),
            f"rows={len(vlm_sigma_rows)}",
        )
        record(
            "vlm_sigma_expected_coverage",
            set(sigma_keys) == expected_sigma_keys,
            f"expected={len(expected_sigma_keys)} actual={len(set(sigma_keys))}",
        )
        record(
            "vlm_sigma_source_rollouts",
            all(
                row["source_rollouts"]
                == 36 * len(VLM_FAMILY_CONDITIONS[row["family"]])
                for row in vlm_sigma_rows
            ),
            "point uses 3×36 and grasp uses 2×36 source trajectories per task",
        )
        record(
            "vlm_sigma_valid_pairs",
            all(
                row["valid_pairs"] > 0 and row["equivalent_sigma_mm"] > 0.0
                for row in vlm_sigma_rows
            ),
            "each VLM sigma cell has positive valid control-step pairs and σ",
        )
    if vlm_sigma_3task_rows is not None:
        pooled_sigma_keys = [
            row["family"] for row in vlm_sigma_3task_rows
        ]
        record(
            "vlm_sigma_3task_expected_coverage",
            pooled_sigma_keys == ["point", "grasp"],
            "three-task pooled VLM σ has one Point and one Grasp row",
        )
        pooled_sigma_matches = True
        for row in vlm_sigma_3task_rows:
            family_rows = [
                item for item in (vlm_sigma_rows or [])
                if item["family"] == row["family"]
            ]
            total_pairs = sum(int(item["valid_pairs"]) for item in family_rows)
            expected_sigma = sum(
                int(item["valid_pairs"]) * float(item["equivalent_sigma_mm"])
                for item in family_rows
            ) / total_pairs
            pooled_sigma_matches &= math.isclose(
                float(row["equivalent_sigma_mm"]),
                expected_sigma,
                rel_tol=0.0,
                abs_tol=0.01,
            )
        record(
            "vlm_sigma_3task_weighted_arithmetic",
            pooled_sigma_matches,
            "three-task pooled VLM σ matches valid-pair-weighted task σ",
        )
    if orientation_equivalent_rows is not None:
        orientation_keys = [row["task"] for row in orientation_equivalent_rows]
        record(
            "orientation_equivalent_expected_coverage",
            orientation_keys == [*TASKS, "3task_pooled"],
            "orientation-equivalent rows cover three tasks plus pooled",
        )
        record(
            "orientation_equivalent_positive",
            all(
                float(row["tracking_error_mean_deg"]) >= 0.0
                and float(row["orientation_equivalent_sigma_deg"]) >= 0.0
                and float(row["position_equivalent_mm"]) >= 0.0
                for row in orientation_equivalent_rows
            ),
            "orientation-equivalent scales are non-negative",
        )
        record(
            "orientation_equivalent_source_is_n0",
            all(row["source_noise_id"] == "n0" for row in orientation_equivalent_rows),
            "behavioral orientation-equivalent scale uses injected n0 orientation noise",
        )
    if vlm_skill_error_rows is not None:
        skill_error_keys = [
            (row["family"], row["task"], row["skill"])
            for row in vlm_skill_error_rows
        ]
        expected_skill_error_keys = {
            ("grasp", task, skill)
            for task in TASKS
            for skill in SKILL_TYPES
        }
        record(
            "vlm_skill_error_unique_keys",
            len(skill_error_keys) == len(set(skill_error_keys)),
            f"rows={len(vlm_skill_error_rows)}",
        )
        record(
            "vlm_skill_error_expected_coverage",
            set(skill_error_keys) == expected_skill_error_keys,
            f"expected={len(expected_skill_error_keys)} actual={len(set(skill_error_keys))}",
        )
        record(
            "vlm_skill_error_positive_counts",
            all(row["valid_pairs"] > 0 for row in vlm_skill_error_rows),
            "every skill/task cell has positive valid VLM-GT pairs",
        )
        record(
            "vlm_skill_error_tail_order",
            all(
                row["tail_gt_40_fraction"]
                >= row["tail_gt_70_fraction"]
                >= row["tail_gt_100_fraction"]
                for row in vlm_skill_error_rows
            ),
            "large-error fractions decrease as the pixel threshold increases",
        )
        record(
            "vlm_skill_error_p95_vs_rms",
            all(
                row["equivalent_p95_mm"] >= row["equivalent_rms_mm"]
                for row in vlm_skill_error_rows
            ),
            "skill-specific p95-equivalent is not below RMS-equivalent",
        )
    if skill_type_rows is not None:
        skill_keys = [
            (
                row["condition_id"],
                row["noise_id"],
                row["task"],
                row["replicate_id"],
                row["skill_type"],
            )
            for row in skill_type_rows
        ]
        pooled_skill_rows = [
            row for row in skill_type_rows if row["replicate_id"] == "pooled"
        ]
        record(
            "skill_type_unique_keys",
            len(skill_keys) == len(set(skill_keys)),
            f"rows={len(skill_type_rows)}",
        )
        record(
            "skill_type_success_arithmetic",
            all(
                row["entered"] > 0
                and row["completed"] >= 0
                and row["completed"] <= row["entered"]
                and math.isclose(
                    row["success_rate"],
                    row["completed"] / row["entered"],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for row in skill_type_rows
                if row["entered"] > 0
            ),
            "skill-level success rate equals completed/entered",
        )
        record(
            "skill_type_pooled_coverage",
            len(pooled_skill_rows) > 0
            and all(row["replicate_id"] == "pooled" for row in pooled_skill_rows),
            f"pooled_rows={len(pooled_skill_rows)}",
        )
        record(
            "skill_type_tracking_counts",
            all(row["tracking_n"] >= 0 for row in skill_type_rows),
            "skill-level tracking counts are non-negative",
        )

    if three_task_rows is not None:
        three_task_keys = [
            (row["condition_id"], row["noise_id"])
            for row in three_task_rows
        ]
        record(
            "three_task_unique_keys",
            len(three_task_keys) == len(set(three_task_keys)),
            f"rows={len(three_task_rows)}",
        )
        expected_three_task_rows = sum(
            9 if family == "point" else 10
            for _, _, family in CONDITIONS
        )
        record(
            "three_task_expected_coverage",
            len(three_task_rows) == expected_three_task_rows,
            f"expected={expected_three_task_rows} actual={len(three_task_rows)}",
        )
        record(
            "three_task_success_arithmetic",
            all(math.isclose(
                row["success_rate"],
                row["n_success"] / row["n_rollouts"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ) for row in three_task_rows),
            "three-task success rate equals n_success/n_rollouts",
        )
        three_task_matches = True
        for row in three_task_rows:
            source_rows = [
                source
                for source in pooled_rows
                if source["condition_id"] == row["condition_id"]
                and source["noise_id"] == row["noise_id"]
            ]
            three_task_matches &= (
                row["n_success"] == sum(
                    source["n_success"] for source in source_rows
                )
                and row["n_rollouts"] == sum(
                    source["n_rollouts"] for source in source_rows
                )
            )
        record(
            "three_task_rollup_matches_pooled",
            three_task_matches,
            "three-task success rollup matches task-level pooled rows",
        )

    if noise_schedule_rows is not None:
        schedule_ids = [row["noise_id"] for row in noise_schedule_rows]
        record(
            "noise_schedule_unique_keys",
            len(schedule_ids) == len(set(schedule_ids)),
            f"rows={len(noise_schedule_rows)}",
        )
        record(
            "noise_schedule_expected_levels",
            schedule_ids == [*NUMERIC_IDS, "r180"],
            "n0–n7 plus r180 are present in order",
        )
        record(
            "noise_schedule_bound_values",
            all(
                math.isclose(
                    row["position_sigma_mm_per_axis"],
                    (
                        POSITION_MM[row["noise_id"]]
                        if row["noise_id"] in POSITION_MM
                        else 0.0
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    row["orientation_sigma_deg"],
                    (
                        ORIENTATION_DEG[row["noise_id"]]
                        if row["noise_id"] in ORIENTATION_DEG
                        else 180.0
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for row in noise_schedule_rows
            ),
            "position and orientation schedule values match the fixed noise design",
        )

    if errors:
        raise ValueError("table validation failed: " + "; ".join(errors))
    return checks


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _expected_rows(
    new_rows: list[dict[str, Any]], legacy_rows: list[dict[str, Any]]
) -> dict[tuple[str, str, int], dict[str, Any]]:
    duplicates: dict[tuple[str, str, int, int, int], int] = defaultdict(int)
    new_index = {}
    for row in new_rows:
        duplicates[_key(row)] += 1
        new_index[_key(row)] = row
    repeated = [key for key, count in duplicates.items() if count != 1]
    if repeated:
        raise ValueError(f"duplicate new manifest keys: {repeated}")
    legacy_index = {
        (str(row.get("condition_id")), str(row.get("noise_id"))): row
        for row in legacy_rows
        if row.get("status") == "ok"
    }
    selected = {}
    missing = []
    for condition_id, _, family in CONDITIONS:
        noise_ids = [*NUMERIC_IDS, "shuffle"]
        if family == "grasp-part":
            noise_ids.append("r180")
        for noise_id in noise_ids:
            for seed in (0, 1, 2):
                if seed == 0 and noise_id in LEGACY_IDS:
                    row = legacy_index.get((condition_id, noise_id))
                else:
                    row = new_index.get((condition_id, noise_id, seed, seed, seed))
                if row is None or row.get("status") != "ok":
                    missing.append((condition_id, noise_id, seed))
                else:
                    selected[(condition_id, noise_id, seed)] = row
    if missing:
        raise ValueError(f"incomplete replicate coverage: {missing[:12]} (total={len(missing)})")
    return selected


def _weighted_tracking(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    sums = defaultdict(float)
    metric_type = "position"
    for payload in payloads:
        tracking = payload.get("tracking_error") or {}
        if tracking.get("complete") is not True:
            raise ValueError("new tracking summary is incomplete")
        metric_type = str(tracking.get("metric_type", metric_type))
        overall = tracking.get("overall") or {}
        item_count = int(overall.get("count", 0))
        count += item_count
        for key in ("mean_pos_m", "mean_ori_deg", "mean_total"):
            if overall.get(key) is not None:
                sums[key] += float(overall[key]) * item_count
    return {
        "tracking_state_count": count,
        "tracking_metric_type": metric_type,
        "track_pos_cm": 100.0 * sums["mean_pos_m"] / count if count else None,
        "track_ori_deg": (
            sums["mean_ori_deg"] / count if count and metric_type == "pose" else None
        ),
        "track_total": (
            sums["mean_total"] / count if count and metric_type == "pose" else None
        ),
    }


def _skill_type_from_state(skill_state: str) -> str:
    """Map task-specific FSM states such as ``leg-top-screw`` to ``screw``."""
    return str(skill_state).rsplit("-", 1)[-1]


def _skill_type_summary(
    success_payloads: list[dict[str, Any]],
    tracking_payloads: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate cascading skill SR and tracking by the five skill types."""
    success_buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"entered": 0, "completed": 0}
    )
    for payload in success_payloads:
        state_counts = payload.get("skill_state_counts") or {}
        completion_counts = payload.get("skill_completion_counts") or {}
        for state in set(state_counts) | set(completion_counts):
            skill_type = _skill_type_from_state(state)
            if skill_type not in SKILL_TYPES:
                continue
            success_buckets[skill_type]["entered"] += int(state_counts.get(state, 0))
            success_buckets[skill_type]["completed"] += int(
                completion_counts.get(state, 0)
            )

    tracking_buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "pos_sum_m": 0.0,
            "ori_sum_deg": 0.0,
            "total_sum": 0.0,
            "pose_count": 0,
            "metric_type": "position",
        }
    )
    for payload in tracking_payloads:
        tracking = payload.get("tracking_error") or {}
        metric_type = str(tracking.get("metric_type", "position"))
        by_skill = tracking.get("by_skill") or {}
        for state, stats in by_skill.items():
            skill_type = _skill_type_from_state(state)
            if skill_type not in SKILL_TYPES:
                continue
            stats = stats or {}
            count = int(stats.get("count", 0) or 0)
            if count <= 0:
                continue
            bucket = tracking_buckets[skill_type]
            bucket["count"] += count
            bucket["pos_sum_m"] += float(stats.get("mean_pos_m", 0.0) or 0.0) * count
            if metric_type == "pose":
                bucket["pose_count"] += count
                bucket["ori_sum_deg"] += float(stats.get("mean_ori_deg", 0.0) or 0.0) * count
                bucket["total_sum"] += float(stats.get("mean_total", 0.0) or 0.0) * count
                bucket["metric_type"] = "pose"

    output: dict[str, dict[str, Any]] = {}
    for skill_type in SKILL_TYPES:
        success = success_buckets.get(skill_type, {"entered": 0, "completed": 0})
        tracking = tracking_buckets.get(skill_type)
        entered = int(success["entered"])
        completed = int(success["completed"])
        count = int(tracking["count"]) if tracking else 0
        pose_count = int(tracking["pose_count"]) if tracking else 0
        metric_type = tracking["metric_type"] if tracking else "position"
        if entered <= 0 and count <= 0:
            continue
        output[skill_type] = {
            "entered": entered,
            "completed": completed,
            "success_rate": completed / entered if entered else None,
            "tracking_n": count,
            "tracking_metric_type": metric_type,
            "track_pos_cm": (
                100.0 * float(tracking["pos_sum_m"]) / count if count else None
            ),
            "track_ori_deg": (
                float(tracking["ori_sum_deg"]) / pose_count
                if pose_count and metric_type == "pose"
                else None
            ),
            "track_total": (
                float(tracking["total_sum"]) / pose_count
                if pose_count and metric_type == "pose"
                else None
            ),
        }
    return output


def _build_products(selected: dict[tuple[str, str, int], dict[str, Any]]):
    replicate_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []
    skill_type_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    summaries = {key: _load_summary(row) for key, row in selected.items()}

    for condition_id, condition, family in CONDITIONS:
        noise_ids = [*NUMERIC_IDS, "shuffle"] + (["r180"] if family == "grasp-part" else [])
        for noise_id in noise_ids:
            for task in TASKS:
                task_payloads = []
                for seed in (0, 1, 2):
                    task_payload = (summaries[(condition_id, noise_id, seed)].get("per_task") or {}).get(task)
                    if not isinstance(task_payload, dict):
                        raise ValueError(f"missing per_task {condition_id}/{noise_id}/rep{seed}/{task}")
                    if int(task_payload.get("n_rollouts", -1)) != 36:
                        raise ValueError(f"wrong rollout count {condition_id}/{noise_id}/rep{seed}/{task}")
                    task_payloads.append(task_payload)
                    successes = int(task_payload.get("n_success", 0))
                    count = int(task_payload["n_rollouts"])
                    low, high = _wilson(successes, count)
                    replicate_rows.append({
                        "condition_id": condition_id,
                        "condition": condition,
                        "family": family,
                        "noise_id": noise_id,
                        "task": task,
                        "replicate_id": seed,
                        "simulator_seed": seed,
                        "annotation_seed": seed,
                        "n_success": successes,
                        "n_rollouts": count,
                        "success_rate": successes / count,
                        "wilson_low": low,
                        "wilson_high": high,
                    })

                successes = sum(int(item.get("n_success", 0)) for item in task_payloads)
                count = sum(int(item.get("n_rollouts", 0)) for item in task_payloads)
                low, high = _wilson(successes, count)
                tracking_payloads = task_payloads[1:] if noise_id in LEGACY_IDS else task_payloads
                tracking = _weighted_tracking(tracking_payloads)
                pooled_rows.append({
                    "condition_id": condition_id,
                    "condition": condition,
                    "family": family,
                    "noise_id": noise_id,
                    "pos_std_mm": POSITION_MM.get(noise_id, 0),
                    "task": task,
                    "n_success": successes,
                    "n_rollouts": count,
                    "success_rate": successes / count,
                    "wilson_low": low,
                    "wilson_high": high,
                    "tracking_n": 72 if noise_id in LEGACY_IDS else 108,
                    **tracking,
                })

                diagnostic_payloads = task_payloads[1:] if noise_id in LEGACY_IDS else task_payloads
                samples = []
                for payload in diagnostic_payloads:
                    samples.extend((payload.get("annotation_noise_stats") or {}).get("phase_samples", []))
                diagnostics = build_annotation_noise_summary(samples)
                position_distribution = diagnostics["position_norm_m"]
                rotation_distribution = diagnostics["rotation_geodesic_deg"]
                diagnostics_rows.append({
                    "condition_id": condition_id,
                    "condition": condition,
                    "noise_id": noise_id,
                    "task": task,
                    "diagnostics_rollout_n": 72 if noise_id in LEGACY_IDS else 108,
                    "phase_count": diagnostics["phase_count"],
                    "position_norm_rms_mm": (
                        1000.0 * position_distribution["rms"]
                        if position_distribution["rms"] is not None else None
                    ),
                    "position_norm_mean_mm": (
                        1000.0 * position_distribution["mean"]
                        if position_distribution["mean"] is not None else None
                    ),
                    "position_norm_p50_mm": (
                        1000.0 * position_distribution["p50"]
                        if position_distribution["p50"] is not None else None
                    ),
                    "position_norm_p90_mm": (
                        1000.0 * position_distribution["p90"]
                        if position_distribution["p90"] is not None else None
                    ),
                    "position_norm_max_mm": (
                        1000.0 * position_distribution["max"]
                        if position_distribution["max"] is not None else None
                    ),
                    "rotation_geodesic_mean_deg": rotation_distribution["mean"],
                    "rotation_geodesic_rms_deg": rotation_distribution["rms"],
                    "rotation_geodesic_p50_deg": rotation_distribution["p50"],
                    "rotation_geodesic_p90_deg": rotation_distribution["p90"],
                    "rotation_geodesic_max_deg": rotation_distribution["max"],
                    "workspace_valid_rate": diagnostics["workspace_valid_rate"],
                    "front_projection_visible_rate": diagnostics["front_projection_visible_rate"],
                    "invalid_nonfinite_rate": diagnostics["invalid_nonfinite_rate"],
                })

                for seed, payload in enumerate(task_payloads):
                    for state, entered in (payload.get("skill_state_counts") or {}).items():
                        completed = int((payload.get("skill_completion_counts") or {}).get(state, 0))
                        skill_rows.append({
                            "condition_id": condition_id,
                            "condition": condition,
                            "noise_id": noise_id,
                            "task": task,
                            "replicate_id": seed,
                            "skill_state": state,
                            "entered": int(entered),
                            "completed": completed,
                            "success_rate": completed / int(entered) if int(entered) else None,
                        })
                tracking_payloads = task_payloads[1:] if noise_id in LEGACY_IDS else task_payloads
                for seed, payload in enumerate(task_payloads):
                    skill_type_summary = _skill_type_summary(
                        [payload],
                        [] if noise_id in LEGACY_IDS and seed == 0 else [payload],
                    )
                    for skill_type, metrics in skill_type_summary.items():
                        skill_type_rows.append({
                            "condition_id": condition_id,
                            "condition": condition,
                            "family": family,
                            "noise_id": noise_id,
                            "pos_std_mm": POSITION_MM.get(noise_id, 0),
                            "task": task,
                            "replicate_id": seed,
                            "skill_type": skill_type,
                            **metrics,
                        })
                pooled_skill_type_summary = _skill_type_summary(
                    task_payloads,
                    tracking_payloads,
                )
                for skill_type, metrics in pooled_skill_type_summary.items():
                    skill_type_rows.append({
                        "condition_id": condition_id,
                        "condition": condition,
                        "family": family,
                        "noise_id": noise_id,
                        "pos_std_mm": POSITION_MM.get(noise_id, 0),
                        "task": task,
                        "replicate_id": "pooled",
                        "skill_type": skill_type,
                        **metrics,
                    })
                states = sorted({row["skill_state"] for row in skill_rows if row["condition_id"] == condition_id and row["noise_id"] == noise_id and row["task"] == task})
                for state in states:
                    relevant = [row for row in skill_rows if row["condition_id"] == condition_id and row["noise_id"] == noise_id and row["task"] == task and row["skill_state"] == state and row["replicate_id"] != "pooled"]
                    entered = sum(row["entered"] for row in relevant)
                    completed = sum(row["completed"] for row in relevant)
                    skill_rows.append({
                        "condition_id": condition_id,
                        "condition": condition,
                        "noise_id": noise_id,
                        "task": task,
                        "replicate_id": "pooled",
                        "skill_state": state,
                        "entered": entered,
                        "completed": completed,
                        "success_rate": completed / entered if entered else None,
                    })
    return replicate_rows, pooled_rows, skill_rows, skill_type_rows, diagnostics_rows


def _trend_rows(pooled_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for condition_id, condition, family in CONDITIONS:
        for task in TASKS:
            rows = sorted(
                [row for row in pooled_rows if row["condition_id"] == condition_id and row["task"] == task and row["noise_id"] in NUMERIC_IDS],
                key=lambda row: int(row["noise_id"][1:]),
            )
            xs = np.arange(8, dtype=np.float64)
            ys = np.asarray([row["success_rate"] for row in rows], dtype=np.float64)
            slope, intercept = np.polyfit(xs, ys, 1)
            output.append({
                "condition_id": condition_id,
                "condition": condition,
                "family": family,
                "task": task,
                "ordinal_slope_per_level": float(slope),
                "ordinal_intercept": float(intercept),
                "pearson_r_on_ordinal_level": float(np.corrcoef(xs, ys)[0, 1]) if np.std(ys) else 0.0,
            })
    return output


def _vlm_sigma_rows() -> list[dict[str, Any]]:
    condition_meta = {
        condition_id: (condition, family)
        for condition_id, condition, family in CONDITIONS
    }
    grouped: dict[tuple[str, str], list[tuple[str, int, float]]] = defaultdict(list)
    for condition_id, task, valid_pairs, equivalent_sigma_mm in VLM_SIGMA_REFERENCE:
        _, condition_family = condition_meta[condition_id]
        family = "grasp" if condition_family == "grasp-part" else "point"
        grouped[(family, task)].append(
            (condition_id, valid_pairs, equivalent_sigma_mm)
        )

    rows = []
    for (family, task), members in grouped.items():
        valid_pairs = sum(item[1] for item in members)
        equivalent_sigma_mm = sum(
            item[1] * item[2] for item in members
        ) / valid_pairs
        source_conditions = "; ".join(
            condition_meta[item[0]][0] for item in members
        )
        rows.append({
            "condition_id": family,
            "condition": VLM_FAMILY_LABELS[family],
            "family": family,
            "task": task,
            "source_conditions": source_conditions,
            "source_rollouts": 36 * len(members),
            "valid_pairs": valid_pairs,
            "equivalent_sigma_mm": round(equivalent_sigma_mm, 2),
            "sigma_kind": "position-equivalent",
            "aggregation": "valid-pair-weighted mean of condition/task Equivalent σ",
            "source_report": "formal VLM diagnostic task-level summary",
        })
    return rows


def _vlm_sigma_3task_rows(
    sigma_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool task-level VLM σ across the three tasks by valid pair count."""
    rows = []
    for family in ("point", "grasp"):
        family_rows = [row for row in sigma_rows if row["family"] == family]
        valid_pairs = sum(int(row["valid_pairs"]) for row in family_rows)
        if valid_pairs <= 0:
            raise ValueError(f"missing valid VLM pairs for {family}")
        equivalent_sigma_mm = sum(
            int(row["valid_pairs"]) * float(row["equivalent_sigma_mm"])
            for row in family_rows
        ) / valid_pairs
        rows.append({
            "condition_id": family,
            "condition": VLM_FAMILY_LABELS[family],
            "family": family,
            "task": "3task_pooled",
            "source_tasks": "; ".join(TASKS),
            "source_conditions": "; ".join(
                sorted({
                    condition
                    for row in family_rows
                    for condition in str(row["source_conditions"]).split("; ")
                })
            ),
            "source_rollouts": sum(
                int(row["source_rollouts"]) for row in family_rows
            ),
            "valid_pairs": valid_pairs,
            "equivalent_sigma_mm": round(equivalent_sigma_mm, 2),
            "sigma_kind": "position-equivalent",
            "aggregation": "valid-pair-weighted mean across task-level σ",
            "source_report": "formal VLM diagnostic task-level summary",
        })
    return rows


def _orientation_schedule_match(
    tracking_error_mean_deg: float,
) -> dict[str, Any]:
    """Map a behavioral orientation residual onto the fixed n0–n7 schedule."""
    orientation_values = np.asarray(
        [ORIENTATION_DEG[noise_id] for noise_id in NUMERIC_IDS],
        dtype=np.float64,
    )
    position_values = np.asarray(
        [POSITION_MM[noise_id] for noise_id in NUMERIC_IDS],
        dtype=np.float64,
    )
    if not np.all(np.diff(orientation_values) > 0.0):
        raise ValueError("orientation schedule must be strictly increasing")
    value = float(tracking_error_mean_deg)
    clipped = float(np.clip(value, orientation_values[0], orientation_values[-1]))
    upper = int(np.searchsorted(orientation_values, clipped, side="right"))
    if upper <= 0:
        lower = 0
        upper = 1
    elif upper >= len(orientation_values):
        lower = len(orientation_values) - 2
        upper = len(orientation_values) - 1
    else:
        lower = upper - 1
    lower_value = float(orientation_values[lower])
    upper_value = float(orientation_values[upper])
    fraction = (
        0.0
        if upper_value == lower_value
        else (clipped - lower_value) / (upper_value - lower_value)
    )
    level = lower + fraction
    if clipped <= orientation_values[0]:
        matched_level = "n0"
    elif clipped >= orientation_values[-1]:
        matched_level = "n7"
    else:
        matched_level = f"n{lower}–n{upper}"
    return {
        "orientation_equivalent_sigma_deg": value,
        "orientation_level": level,
        "matched_noise_level": matched_level,
        "position_equivalent_mm": float(
            np.interp(clipped, orientation_values, position_values)
        ),
        "lower_orientation_deg": lower_value,
        "upper_orientation_deg": upper_value,
    }


def _orientation_tracking_equivalent_rows(
    pooled_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive a behavioral orientation-equivalent marker from grasp n0 tracking.

    The n0 condition has no injected orientation noise. Its clean-GT tracking
    residual is therefore useful as a downstream behavioral scale, but it is
    not a raw upstream VLM orientation residual.
    """
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    for task in TASKS:
        grouped.append((
            task,
            [
                row
                for row in pooled_rows
                if row["family"] == "grasp-part"
                and row["task"] == task
                and row["noise_id"] == "n0"
                and row.get("track_ori_deg") is not None
                and int(row.get("tracking_state_count", 0)) > 0
            ],
        ))
    grouped.append((
        "3task_pooled",
        [
            row
            for row in pooled_rows
            if row["family"] == "grasp-part"
            and row["noise_id"] == "n0"
            and row.get("track_ori_deg") is not None
            and int(row.get("tracking_state_count", 0)) > 0
        ],
    ))

    rows = []
    for task, source_rows in grouped:
        tracking_state_count = sum(
            int(row["tracking_state_count"]) for row in source_rows
        )
        if tracking_state_count <= 0:
            raise ValueError(f"missing grasp n0 orientation tracking for {task}")
        tracking_error_mean_deg = sum(
            int(row["tracking_state_count"]) * float(row["track_ori_deg"])
            for row in source_rows
        ) / tracking_state_count
        match = _orientation_schedule_match(tracking_error_mean_deg)
        rows.append({
            "condition_id": "grasp",
            "condition": VLM_FAMILY_LABELS["grasp"],
            "family": "grasp",
            "task": task,
            "source_noise_id": "n0",
            "source_conditions": "; ".join(
                sorted({str(row["condition"]) for row in source_rows})
            ),
            "source_tasks": "; ".join(
                sorted({str(row["task"]) for row in source_rows})
            ),
            "tracking_state_count": tracking_state_count,
            "tracking_error_mean_deg": round(tracking_error_mean_deg, 2),
            **{
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in match.items()
            },
            "interpretation": (
                "behavioral orientation-equivalent scale from clean-GT "
                "tracking; not raw VLM orientation residual"
            ),
        })
    return rows


def _noise_schedule_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "noise_id": noise_id,
            "position_sigma_mm_per_axis": POSITION_MM[noise_id],
            "orientation_sigma_deg": ORIENTATION_DEG[noise_id],
            "schedule_role": "bound position+orientation perturbation",
        }
        for noise_id in NUMERIC_IDS
    ]
    rows.append({
        "noise_id": "r180",
        "position_sigma_mm_per_axis": 0.0,
        "orientation_sigma_deg": 180.0,
        "schedule_role": "orientation-only endpoint",
    })
    return rows


def _add_unique_figure_legend(fig, axes, **kwargs) -> None:
    """Build one figure-level legend without repeating labels from each subplot."""
    handles_by_label = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in handles_by_label:
                handles_by_label[label] = handle
    fig.legend(handles_by_label.values(), handles_by_label.keys(), **kwargs)


def _categorical_endpoint_offsets(
    rows: list[dict[str, Any]],
    metric_key: str,
    *,
    start_noise: str = "n7",
    endpoint_noise: str = "shuffle",
    spacing: float = 0.035,
) -> dict[str, float]:
    """Offset only complete endpoint pairs that are exactly coincident."""
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[str(row["condition_id"])][str(row["noise_id"])] = row

    endpoint_groups: dict[tuple[float, float], list[str]] = defaultdict(list)
    offsets = {condition_id: 0.0 for condition_id, _, _ in CONDITIONS}
    for condition_id, _, _ in CONDITIONS:
        start = by_condition.get(condition_id, {}).get(start_noise)
        endpoint = by_condition.get(condition_id, {}).get(endpoint_noise)
        if start is None or endpoint is None:
            continue
        start_value = start.get(metric_key)
        endpoint_value = endpoint.get(metric_key)
        if start_value is None or endpoint_value is None:
            continue
        endpoint_groups[
            (round(float(start_value), 12), round(float(endpoint_value), 12))
        ].append(condition_id)

    for condition_ids in endpoint_groups.values():
        if len(condition_ids) <= 1:
            continue
        center = 0.5 * (len(condition_ids) - 1)
        for idx, condition_id in enumerate(condition_ids):
            offsets[condition_id] = (idx - center) * spacing
    return offsets


def _plot_categorical_transition(
    axis,
    pooled_rows: list[dict[str, Any]],
    task: str,
    metric_key: str,
    colors: dict[str, str],
    styles: dict[str, dict[str, str]],
    *,
    scale: float = 1.0,
) -> None:
    """Draw the n7-to-Shuffle transition on a fresh36-style narrow axis."""
    task_rows = [row for row in pooled_rows if row["task"] == task]
    offsets = _categorical_endpoint_offsets(task_rows, metric_key)
    for condition_id, _, _ in CONDITIONS:
        start = next(
            (
                row
                for row in task_rows
                if row["condition_id"] == condition_id
                and row["noise_id"] == "n7"
            ),
            None,
        )
        endpoint = next(
            (
                row
                for row in task_rows
                if row["condition_id"] == condition_id
                and row["noise_id"] == "shuffle"
            ),
            None,
        )
        if start is None or endpoint is None:
            continue
        start_value = start.get(metric_key)
        endpoint_value = endpoint.get(metric_key)
        if start_value is None or endpoint_value is None:
            continue
        offset = offsets[condition_id]
        axis.plot(
            [offset, 1.0 + offset],
            [scale * float(start_value), scale * float(endpoint_value)],
            color=colors[condition_id],
            marker=styles[condition_id]["marker"],
            linestyle=styles[condition_id]["linestyle"],
            linewidth=0.75,
            markersize=3.0,
            markeredgewidth=0.45,
            alpha=0.92,
            zorder=3,
        )


def _plot_success(
    pooled_table_path: Path,
    sigma_table_path: Path,
    figures_dir: Path,
) -> Path:
    """Plot success and task-level VLM σ only from persisted tables."""
    pooled_rows = _read_csv(pooled_table_path)
    sigma_rows = _read_csv(sigma_table_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    numeric_path = figures_dir / "vlm_cover_108_success_numeric.png"
    colors = {item[0]: f"C{idx}" for idx, item in enumerate(CONDITIONS)}
    styles = {
        "gp": {"marker": "o", "linestyle": "-"},
        "colored_gp": {"marker": "s", "linestyle": "--"},
        "gp_skill": {"marker": "^", "linestyle": "-."},
        "grasp_part": {"marker": "D", "linestyle": ":"},
        "grasp_part_colored": {"marker": "P", "linestyle": "-."},
    }
    fig = plt.figure(figsize=(19, 5.0))
    grid = fig.add_gridspec(1, 6, width_ratios=[4, 1.4] * 3, wspace=0.08)
    shared_axis = None
    main_axes = []
    shuffle_axes = []
    for task_idx, task in enumerate(TASKS):
        axis = fig.add_subplot(grid[0, 2 * task_idx], sharey=shared_axis)
        if shared_axis is None:
            shared_axis = axis
        shuffle_axis = fig.add_subplot(
            grid[0, 2 * task_idx + 1], sharey=shared_axis
        )
        main_axes.append(axis)
        shuffle_axes.append(shuffle_axis)

        # The success figure uses only position-equivalent VLM references.
        # The Grasp orientation-equivalent scale remains a textual/appendix
        # diagnostic and is intentionally not drawn as a third vertical line.
        _plot_vlm_sigma_markers(
            axis,
            sigma_rows,
            task_filter=task,
            show_labels=task_idx == 0,
        )
        if task_idx == 0:
            axis.plot(
                [],
                [],
                color="C0",
                linestyle="--",
                linewidth=1.05,
                label="Point VLM position-equivalent σ",
            )
            axis.plot(
                [],
                [],
                color="C3",
                linestyle=":",
                linewidth=1.05,
                label="Grasp VLM position-equivalent σ",
            )
        axis.set_axisbelow(True)
        for condition_id, condition, family in CONDITIONS:
            rows = sorted(
                [
                    row
                    for row in pooled_rows
                    if row["condition_id"] == condition_id
                    and row["task"] == task
                    and row["noise_id"] in NUMERIC_IDS
                ],
                key=lambda row: float(row["pos_std_mm"]),
            )
            xs = [float(row["pos_std_mm"]) for row in rows]
            ys = [100.0 * row["success_rate"] for row in rows]
            axis.plot(
                xs,
                ys,
                label=condition,
                color=colors[condition_id],
                marker=styles[condition_id]["marker"],
                linestyle=styles[condition_id]["linestyle"],
                linewidth=0.75,
                markersize=3.0,
                markeredgewidth=0.45,
                alpha=0.92,
                zorder=3,
            )
        tick_rows = sorted(
            [
                row
                for row in pooled_rows
                if row["condition_id"] == CONDITIONS[0][0]
                and row["task"] == task
                and row["noise_id"] in NUMERIC_IDS
            ],
            key=lambda row: float(row["pos_std_mm"]),
        )
        position_values = [float(row["pos_std_mm"]) for row in tick_rows]
        noise_labels = [str(row["noise_id"]) for row in tick_rows]
        axis.set_title(task)
        axis.set_xticks(position_values)
        axis.set_xticklabels(
            noise_labels,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=7.5,
        )
        axis.set_xlim(-4.0, 205.0)
        axis.set_xlabel("noise position σ per axis (mm)")
        axis.grid(alpha=0.25)

        # Keep Shuffle categorical, as in the fresh36 figure, rather than
        # assigning it a fictitious position-noise coordinate.
        _plot_categorical_transition(
            shuffle_axis,
            pooled_rows,
            task,
            "success_rate",
            colors,
            styles,
            scale=100.0,
        )
        shuffle_axis.set_facecolor("#f4f4f4")
        shuffle_axis.set_xticks([0, 1])
        shuffle_axis.set_xticklabels(
            ["n7", "Shuffle"],
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=7.5,
        )
        shuffle_axis.set_xlim(-0.20, 1.20)
        shuffle_axis.grid(alpha=0.25)
        shuffle_axis.tick_params(labelleft=False)
        shuffle_axis.set_axisbelow(True)

    main_axes[0].set_ylabel("success rate (%)")
    _add_unique_figure_legend(
        fig,
        [*main_axes, *shuffle_axes],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        fontsize=8,
    )
    fig.suptitle(
        "Task-level success; actual noise scale with task-specific VLM σ markers",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.28, top=0.88, wspace=0.08)
    _save_figure_bundle(fig, numeric_path)
    plt.close(fig)
    return numeric_path


def _plot_tracking_error(pooled_table_path: Path, figures_dir: Path) -> Path:
    """Plot fresh36-style position/orientation/total tracking from the table."""
    pooled_rows = _read_csv(pooled_table_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "vlm_cover_108_tracking_error.png"
    colors = {item[0]: f"C{idx}" for idx, item in enumerate(CONDITIONS)}
    styles = {
        "gp": {"marker": "o", "linestyle": "-"},
        "colored_gp": {"marker": "s", "linestyle": "--"},
        "gp_skill": {"marker": "^", "linestyle": "-."},
        "grasp_part": {"marker": "D", "linestyle": ":"},
        "grasp_part_colored": {"marker": "P", "linestyle": "-."},
    }
    metrics = (
        ("track_pos_cm", "Position Error (cm)"),
        ("track_ori_deg", "Orientation Error (deg)"),
        ("track_total", "Total Error"),
    )
    fig = plt.figure(figsize=(19, 13))
    grid = fig.add_gridspec(
        len(metrics),
        2 * len(TASKS),
        width_ratios=[4, 1.4] * len(TASKS),
        hspace=0.22,
        wspace=0.08,
    )
    main_axes = []
    shuffle_axes = []
    for metric_idx, (metric_key, metric_label) in enumerate(metrics):
        shared_axis = None
        for task_idx, task in enumerate(TASKS):
            axis = fig.add_subplot(
                grid[metric_idx, 2 * task_idx],
                sharey=shared_axis,
            )
            if shared_axis is None:
                shared_axis = axis
            shuffle_axis = fig.add_subplot(
                grid[metric_idx, 2 * task_idx + 1],
                sharey=shared_axis,
            )
            main_axes.append(axis)
            shuffle_axes.append(shuffle_axis)

            for condition_id, condition, _ in CONDITIONS:
                if (
                    metric_key != "track_pos_cm"
                    and condition_id in {"gp", "colored_gp", "gp_skill"}
                ):
                    continue
                rows = sorted(
                    [
                        row
                        for row in pooled_rows
                        if row["condition_id"] == condition_id
                        and row["task"] == task
                        and row["noise_id"] in NUMERIC_IDS
                        and row.get(metric_key) is not None
                    ],
                    key=lambda row: float(row["pos_std_mm"]),
                )
                if not rows:
                    continue
                axis.plot(
                    [float(row["pos_std_mm"]) for row in rows],
                    [float(row[metric_key]) for row in rows],
                    label=condition,
                    color=colors[condition_id],
                    marker=styles[condition_id]["marker"],
                    linestyle=styles[condition_id]["linestyle"],
                    linewidth=0.75,
                    markersize=3.0,
                    markeredgewidth=0.45,
                    alpha=0.92,
                    zorder=3,
                )

            tick_rows = sorted(
                {
                    (float(row["pos_std_mm"]), str(row["noise_id"]))
                    for row in pooled_rows
                    if row["task"] == task and row["noise_id"] in NUMERIC_IDS
                }
            )
            if metric_idx == 0:
                axis.set_title(task)
                shuffle_axis.set_title("Categorical", fontsize=8)
            if metric_idx == len(metrics) - 1:
                axis.set_xlabel("noise position σ per axis (mm)")
            if task_idx == 0:
                axis.set_ylabel(metric_label)
            axis.set_xticks([value for value, _ in tick_rows])
            axis.set_xticklabels(
                [noise_id for _, noise_id in tick_rows],
                rotation=35,
                ha="right",
                rotation_mode="anchor",
                fontsize=7.5,
            )
            axis.set_xlim(-4.0, 205.0)
            axis.grid(alpha=0.25)
            axis.set_axisbelow(True)
            if task_idx > 0:
                axis.tick_params(labelleft=False)

            _plot_categorical_transition(
                shuffle_axis,
                pooled_rows,
                task,
                metric_key,
                colors,
                styles,
            )
            shuffle_axis.set_facecolor("#f4f4f4")
            shuffle_axis.set_xticks([0, 1])
            shuffle_axis.set_xticklabels(
                ["n7", "Shuffle"],
                rotation=35,
                ha="right",
                rotation_mode="anchor",
                fontsize=7.5,
            )
            shuffle_axis.set_xlim(-0.20, 1.20)
            shuffle_axis.grid(alpha=0.25)
            shuffle_axis.tick_params(labelleft=False)
            shuffle_axis.set_axisbelow(True)

    _add_unique_figure_legend(
        fig,
        [*main_axes, *shuffle_axes],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        fontsize=8,
    )
    fig.suptitle(
        "Pooled tracking error curves; position / orientation / total",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.05,
        right=0.99,
        bottom=0.12,
        top=0.94,
        wspace=0.08,
    )
    _save_figure_bundle(fig, figure_path)
    plt.close(fig)
    return figure_path


def _plot_vlm_sigma_markers(
    axis,
    sigma_rows: list[dict[str, Any]],
    *,
    task_filter: str | None = None,
    include_pooled: bool = False,
    show_labels: bool = False,
    show_pooled_labels: bool = False,
) -> None:
    """Mark position VLM scales; orientation markers are appendix-only diagnostics."""
    def keep_row(row: dict[str, Any]) -> bool:
        row_task = str(row["task"])
        if task_filter is not None:
            return row_task == task_filter
        return include_pooled or row_task != "3task_pooled"

    for family in ("point", "grasp"):
        family_rows = [
            row
            for row in sigma_rows
            if row["family"] == family and keep_row(row)
        ]
        color = "C0" if family == "point" else "C3"
        for row in family_rows:
            sigma = float(row["equivalent_sigma_mm"])
            row_task = str(row["task"])
            pooled = row_task == "3task_pooled"
            axis.axvline(
                sigma,
                color=color,
                linestyle=(
                    VLM_SIGMA_POOLED_STYLE
                    if pooled
                    else VLM_SIGMA_TASK_STYLES[row_task]
                ),
                linewidth=1.55 if pooled else 0.9,
                alpha=0.92 if pooled else 0.58,
                zorder=2 if pooled else 1,
            )
            if (
                (show_labels and not pooled)
                or (show_pooled_labels and pooled)
            ):
                label = (
                    f"{'P' if family == 'point' else 'G'}·"
                    f"{'3task' if pooled else row_task}"
                )
                axis.text(
                    sigma,
                    0.96 if pooled else 1.0,
                    label,
                    transform=axis.get_xaxis_transform(),
                    rotation=90,
                    rotation_mode="anchor",
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color=color,
                    alpha=0.9,
                    clip_on=False,
                )


def _three_task_pooled_rows(
    pooled_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pool the three task rows for the summary figures.

    Success is pooled by rollout count. Tracking is pooled by the number of
    valid final skill-state records contributing to each task-level mean.
    """
    rows: list[dict[str, Any]] = []
    noise_ids = [*NUMERIC_IDS, "shuffle", "r180"]
    for condition_id, condition, family in CONDITIONS:
        for noise_id in noise_ids:
            task_rows = [
                row
                for row in pooled_rows
                if row["condition_id"] == condition_id
                and row["noise_id"] == noise_id
            ]
            if not task_rows:
                continue
            n_success = sum(row["n_success"] for row in task_rows)
            n_rollouts = sum(row["n_rollouts"] for row in task_rows)
            track_rows = [
                row
                for row in task_rows
                if row.get("track_pos_cm") is not None
                and row.get("tracking_state_count", 0) > 0
            ]
            tracking_state_count = sum(
                row["tracking_state_count"] for row in track_rows
            )

            def weighted_metric(metric_key: str) -> float | None:
                valid_rows = [
                    row
                    for row in track_rows
                    if row.get(metric_key) is not None
                ]
                weight = sum(row["tracking_state_count"] for row in valid_rows)
                if weight <= 0:
                    return None
                return sum(
                    row["tracking_state_count"] * row[metric_key]
                    for row in valid_rows
                ) / weight

            pos_std_values = {
                row.get("pos_std_mm")
                for row in task_rows
                if row.get("pos_std_mm") is not None
            }
            rows.append({
                "condition": condition,
                "condition_id": condition_id,
                "family": family,
                "n_rollouts": n_rollouts,
                "n_success": n_success,
                "noise_id": noise_id,
                "pos_std_mm": (
                    next(iter(pos_std_values))
                    if len(pos_std_values) == 1
                    else None
                ),
                "success_rate": n_success / n_rollouts,
                "track_ori_deg": weighted_metric("track_ori_deg"),
                "track_pos_cm": weighted_metric("track_pos_cm"),
                "track_total": weighted_metric("track_total"),
                "tracking_metric_type": (
                    "pose" if family == "grasp-part" else "position"
                ),
                "tracking_n": sum(row["tracking_n"] for row in track_rows),
                "tracking_rollout_count": sum(
                    row["tracking_n"] for row in track_rows
                ),
                "tracking_state_count": tracking_state_count,
            })
    return rows


def _plot_aggregate_categorical_transition(
    axis,
    pooled_rows: list[dict[str, Any]],
    metric_key: str,
    colors: dict[str, str],
    styles: dict[str, dict[str, str]],
    *,
    scale: float = 1.0,
) -> None:
    """Draw pooled n7→Shuffle endpoints with overlap-only offsets."""
    offsets = _categorical_endpoint_offsets(pooled_rows, metric_key)
    for condition_id, _, _ in CONDITIONS:
        start = next(
            (
                row for row in pooled_rows
                if row["condition_id"] == condition_id
                and row["noise_id"] == "n7"
            ),
            None,
        )
        endpoint = next(
            (
                row for row in pooled_rows
                if row["condition_id"] == condition_id
                and row["noise_id"] == "shuffle"
            ),
            None,
        )
        if start is None or endpoint is None:
            continue
        start_value = start.get(metric_key)
        endpoint_value = endpoint.get(metric_key)
        if start_value is None or endpoint_value is None:
            continue
        offset = offsets[condition_id]
        axis.plot(
            [offset, 1.0 + offset],
            [scale * float(start_value), scale * float(endpoint_value)],
            color=colors[condition_id],
            marker=styles[condition_id]["marker"],
            linestyle=styles[condition_id]["linestyle"],
            linewidth=0.8,
            markersize=3.0,
            markeredgewidth=0.45,
            alpha=0.92,
            zorder=3,
            label=next(
                condition for cid, condition, _ in CONDITIONS
                if cid == condition_id
            ),
        )


def _plot_three_task_summary(
    three_task_table_path: Path,
    pooled_sigma_table_path: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    """Plot three-task pooled success and position tracking summaries."""
    rows = _read_csv(three_task_table_path)
    pooled_sigma_rows = _read_csv(pooled_sigma_table_path)
    numeric_rows = [
        row for row in rows if row["noise_id"] in NUMERIC_IDS
    ]
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {item[0]: f"C{idx}" for idx, item in enumerate(CONDITIONS)}
    styles = {
        "gp": {"marker": "o", "linestyle": "-"},
        "colored_gp": {"marker": "s", "linestyle": "--"},
        "gp_skill": {"marker": "^", "linestyle": "-."},
        "grasp_part": {"marker": "D", "linestyle": ":"},
        "grasp_part_colored": {"marker": "P", "linestyle": "-."},
    }
    generated: dict[str, Path] = {}

    success_path = figures_dir / "vlm_cover_108_success_3task_pooled.png"
    fig = plt.figure(figsize=(10.5, 4.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[5.8, 1.35], wspace=0.08)
    axis = fig.add_subplot(grid[0, 0])
    endpoint_axis = fig.add_subplot(grid[0, 1], sharey=axis)
    for condition_id, condition, _ in CONDITIONS:
        condition_rows = sorted(
            [
                row for row in numeric_rows
                if row["condition_id"] == condition_id
            ],
            key=lambda row: float(row["pos_std_mm"]),
        )
        axis.plot(
            [float(row["pos_std_mm"]) for row in condition_rows],
            [100.0 * row["success_rate"] for row in condition_rows],
            color=colors[condition_id],
            marker=styles[condition_id]["marker"],
            linestyle=styles[condition_id]["linestyle"],
            linewidth=0.85,
            markersize=3.2,
            markeredgewidth=0.45,
            alpha=0.92,
            label=condition,
        )
    # Keep only the two position-equivalent family references in the plot.
    # Orientation-equivalent matching is retained in the appendix table and
    # described in the report text, but is not shown as a vertical marker.
    _plot_vlm_sigma_markers(
        axis,
        pooled_sigma_rows,
        task_filter="3task_pooled",
        include_pooled=True,
        show_labels=False,
        show_pooled_labels=True,
    )
    axis.set_xlabel("noise position σ per axis (mm)")
    axis.set_ylabel("success rate (%)")
    axis.set_title("Three-task pooled success (n=324 per condition/noise)")
    axis.set_xticks(sorted({float(row["pos_std_mm"]) for row in numeric_rows}))
    axis.set_xticklabels(
        [f"n{i}" for i in range(8)],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=7.5,
    )
    axis.set_xlim(-4.0, 205.0)
    axis.set_ylim(0.0, 105.0)
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    _plot_aggregate_categorical_transition(
        endpoint_axis, rows, "success_rate", colors, styles, scale=100.0
    )
    endpoint_axis.set_facecolor("#f4f4f4")
    endpoint_axis.set_xticks([0, 1])
    endpoint_axis.set_xticklabels(
        ["n7", "Shuffle"],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=7.5,
    )
    endpoint_axis.set_xlim(-0.20, 1.20)
    endpoint_axis.grid(alpha=0.25)
    endpoint_axis.tick_params(labelleft=False)
    endpoint_axis.set_axisbelow(True)
    _add_unique_figure_legend(
        fig, [axis, endpoint_axis], loc="lower center",
        bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=8,
    )
    fig.suptitle(
        "Three-task pooled success; actual numeric noise scale with n7→Shuffle endpoint",
        fontsize=11.5,
        y=0.98,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.29, top=0.86)
    _save_figure_bundle(fig, success_path)
    plt.close(fig)
    generated["success_3task_pooled"] = success_path

    tracking_path = figures_dir / "vlm_cover_108_tracking_position_3task_pooled.png"
    fig = plt.figure(figsize=(10.5, 4.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[5.8, 1.35], wspace=0.08)
    axis = fig.add_subplot(grid[0, 0])
    endpoint_axis = fig.add_subplot(grid[0, 1], sharey=axis)
    for condition_id, condition, _ in CONDITIONS:
        condition_rows = sorted(
            [
                row for row in numeric_rows
                if row["condition_id"] == condition_id
                and row.get("track_pos_cm") is not None
            ],
            key=lambda row: float(row["pos_std_mm"]),
        )
        axis.plot(
            [float(row["pos_std_mm"]) for row in condition_rows],
            [float(row["track_pos_cm"]) for row in condition_rows],
            color=colors[condition_id],
            marker=styles[condition_id]["marker"],
            linestyle=styles[condition_id]["linestyle"],
            linewidth=0.85,
            markersize=3.2,
            markeredgewidth=0.45,
            alpha=0.92,
            label=condition,
        )
    _plot_vlm_sigma_markers(
        axis,
        pooled_sigma_rows,
        task_filter="3task_pooled",
        include_pooled=True,
        show_labels=False,
    )
    axis.set_xlabel("noise position σ per axis (mm)")
    axis.set_ylabel("position tracking error (cm)")
    axis.set_title("Three-task pooled position tracking (state-count weighted)")
    axis.set_xticks(sorted({float(row["pos_std_mm"]) for row in numeric_rows}))
    axis.set_xticklabels(
        [f"n{i}" for i in range(8)],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=7.5,
    )
    axis.set_xlim(-4.0, 205.0)
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    _plot_aggregate_categorical_transition(
        endpoint_axis, rows, "track_pos_cm", colors, styles
    )
    endpoint_axis.set_facecolor("#f4f4f4")
    endpoint_axis.set_xticks([0, 1])
    endpoint_axis.set_xticklabels(
        ["n7", "Shuffle"],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=7.5,
    )
    endpoint_axis.set_xlim(-0.20, 1.20)
    endpoint_axis.grid(alpha=0.25)
    endpoint_axis.tick_params(labelleft=False)
    endpoint_axis.set_axisbelow(True)
    _add_unique_figure_legend(
        fig, [axis, endpoint_axis], loc="lower center",
        bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=8,
    )
    fig.suptitle(
        "Three-task pooled position tracking; actual numeric noise scale with n7→Shuffle endpoint",
        fontsize=11.5,
        y=0.98,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.29, top=0.86)
    _save_figure_bundle(fig, tracking_path)
    plt.close(fig)
    generated["tracking_position_3task_pooled"] = tracking_path
    return generated


def _skill_vlm_error_row(
    rows: list[dict[str, Any]],
    *,
    task: str,
    skill_type: str,
) -> dict[str, Any] | None:
    return next(
        (
            row for row in rows
            if row["task"] == task
            and row["skill"] == skill_type
            and row["family"] == "grasp"
        ),
        None,
    )


def _plot_skill_vlm_reference_lines(
    axis,
    *,
    sigma_rows: list[dict[str, Any]],
    skill_error_rows: list[dict[str, Any]],
    task: str,
    skill_type: str,
    label_references: bool,
) -> None:
    """Add compact VLM error references to each skill-level panel."""
    for family, color, label in (
        ("point", "C0", "Point VLM RMS-equivalent σ"),
        ("grasp", "C3", "Grasp VLM RMS-equivalent σ"),
    ):
        row = next(
            (
                item for item in sigma_rows
                if item["family"] == family and item["task"] == task
            ),
            None,
        )
        if row is None:
            continue
        axis.axvline(
            float(row["equivalent_sigma_mm"]),
            color=color,
            linestyle="--",
            linewidth=0.75,
            alpha=0.42,
            zorder=1,
            label=label if label_references else None,
        )
    skill_row = _skill_vlm_error_row(
        skill_error_rows,
        task=task,
        skill_type=skill_type,
    )
    if skill_row is None:
        return
    p95_sigma = float(skill_row["equivalent_p95_mm"])
    clipped = min(p95_sigma, 204.0)
    axis.axvline(
        clipped,
        color="C3",
        linestyle=(0, (1.0, 1.4)),
        linewidth=0.95,
        alpha=0.72,
        zorder=1,
        label=(
            "Grasp VLM p95-equivalent"
            if label_references
            else None
        ),
    )
    if p95_sigma > 204.0:
        axis.text(
            204.0,
            0.88,
            "p95>n7",
            transform=axis.get_xaxis_transform(),
            rotation=90,
            rotation_mode="anchor",
            ha="right",
            va="top",
            fontsize=5.8,
            color="C3",
            alpha=0.82,
            clip_on=False,
        )


def _plot_skill_grids(
    skill_table_path: Path,
    sigma_table_path: Path,
    skill_error_table_path: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    """Plot fresh36-style five-skill × three-task grids from the skill table."""
    rows = [
        row
        for row in _read_csv(skill_table_path)
        if row["replicate_id"] == "pooled"
    ]
    sigma_rows = _read_csv(sigma_table_path)
    skill_error_rows = _read_csv(skill_error_table_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {item[0]: f"C{idx}" for idx, item in enumerate(CONDITIONS)}
    styles = {
        "gp": {"marker": "o", "linestyle": "-"},
        "colored_gp": {"marker": "s", "linestyle": "--"},
        "gp_skill": {"marker": "^", "linestyle": "-."},
        "grasp_part": {"marker": "D", "linestyle": ":"},
        "grasp_part_colored": {"marker": "P", "linestyle": "-."},
    }
    metric_specs = (
        ("success_rate", "Skill success rate (%)", False, 100.0, "skill_success_rate"),
        ("track_pos_cm", "Position error (cm)", False, 1.0, "tracking_position"),
        ("track_ori_deg", "Orientation error (deg)", True, 1.0, "tracking_orientation"),
        ("track_total", "Total error", True, 1.0, "tracking_total"),
    )
    generated: dict[str, Path] = {}

    for metric_key, metric_label, pose_only, scale, suffix in metric_specs:
        file_suffix = suffix[6:] if suffix.startswith("skill_") else suffix
        figure_path = figures_dir / f"vlm_cover_108_skill_{file_suffix}.png"
        fig = plt.figure(figsize=(19, 20))
        grid = fig.add_gridspec(
            len(SKILL_TYPES),
            2 * len(TASKS),
            width_ratios=[4, 1.4] * len(TASKS),
            hspace=0.30,
            wspace=0.08,
        )
        main_axes = []
        shuffle_axes = []
        for skill_idx, skill_type in enumerate(SKILL_TYPES):
            row_shared_axis = None
            for task_idx, task in enumerate(TASKS):
                axis = fig.add_subplot(
                    grid[skill_idx, 2 * task_idx],
                    sharey=row_shared_axis,
                )
                if row_shared_axis is None:
                    row_shared_axis = axis
                shuffle_axis = fig.add_subplot(
                    grid[skill_idx, 2 * task_idx + 1],
                    sharey=axis,
                )
                main_axes.append(axis)
                shuffle_axes.append(shuffle_axis)
                candidates = [
                    row
                    for row in rows
                    if row["task"] == task and row["skill_type"] == skill_type
                ]
                for condition_id, condition, _ in CONDITIONS:
                    if pose_only and condition_id in {"gp", "colored_gp", "gp_skill"}:
                        continue
                    condition_rows = [
                        row
                        for row in candidates
                        if row["condition_id"] == condition_id
                    ]
                    regular = sorted(
                        [
                            row
                            for row in condition_rows
                            if row["noise_id"] in NUMERIC_IDS
                            and row.get(metric_key) is not None
                        ],
                        key=lambda row: float(row["pos_std_mm"]),
                    )
                    if regular:
                        axis.plot(
                            [float(row["pos_std_mm"]) for row in regular],
                            [scale * float(row[metric_key]) for row in regular],
                            label=condition,
                            color=colors[condition_id],
                            marker=styles[condition_id]["marker"],
                            linestyle=styles[condition_id]["linestyle"],
                            linewidth=0.75,
                            markersize=2.8,
                            markeredgewidth=0.45,
                            alpha=0.92,
                            zorder=3,
                        )
                _plot_skill_vlm_reference_lines(
                    axis,
                    sigma_rows=sigma_rows,
                    skill_error_rows=skill_error_rows,
                    task=task,
                    skill_type=skill_type,
                    label_references=skill_idx == 0 and task_idx == 0,
                )
                tick_rows = sorted(
                    {
                        (float(row["pos_std_mm"]), str(row["noise_id"]))
                        for row in candidates
                        if row["noise_id"] in NUMERIC_IDS
                    }
                )
                if metric_key == "success_rate":
                    axis.set_ylim(0.0, 105.0)
                axis.set_xticks([value for value, _ in tick_rows])
                axis.set_xticklabels(
                    [noise_id for _, noise_id in tick_rows],
                    rotation=35,
                    ha="right",
                    rotation_mode="anchor",
                    fontsize=7.5,
                )
                axis.set_xlim(-4.0, 205.0)
                axis.grid(alpha=0.25)
                axis.set_axisbelow(True)
                if skill_idx == 0:
                    axis.set_title(task)
                    shuffle_axis.set_title("Categorical", fontsize=8)
                if skill_idx == len(SKILL_TYPES) - 1:
                    axis.set_xlabel("noise position σ per axis (mm)")
                if task_idx == 0:
                    axis.set_ylabel(f"{skill_type}\n{metric_label}")
                else:
                    axis.tick_params(labelleft=False)

                _plot_categorical_transition(
                    shuffle_axis,
                    candidates,
                    task,
                    metric_key,
                    colors,
                    styles,
                    scale=scale,
                )
                shuffle_axis.set_facecolor("#f4f4f4")
                shuffle_axis.set_xticks([0, 1])
                shuffle_axis.set_xticklabels(
                    ["n7", "Shuffle"],
                    rotation=35,
                    ha="right",
                    rotation_mode="anchor",
                    fontsize=7.5,
                )
                shuffle_axis.set_xlim(-0.20, 1.20)
                shuffle_axis.grid(alpha=0.25)
                shuffle_axis.tick_params(labelleft=False)
                shuffle_axis.set_axisbelow(True)

        _add_unique_figure_legend(
            fig,
            [*main_axes, *shuffle_axes],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=2 if pose_only else 5,
            fontsize=8,
        )
        title_label = (
            "success rate"
            if metric_key == "success_rate"
            else metric_label.lower()
        )
        fig.suptitle(
            f"Skill-level {title_label}; n0–n7 actual noise scale with VLM references",
            fontsize=12,
            y=0.995,
        )
        fig.subplots_adjust(
            left=0.06,
            right=0.99,
            bottom=0.075,
            top=0.97,
            wspace=0.08,
        )
        _save_figure_bundle(fig, figure_path)
        plt.close(fig)
        generated[suffix] = figure_path
    return generated


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(xs, ys)
    )
    denominator = math.sqrt(
        sum((x_value - x_mean) ** 2 for x_value in xs)
        * sum((y_value - y_mean) ** 2 for y_value in ys)
    )
    return numerator / denominator if denominator else float("nan")


def _tracking_endpoint_comparison(
    pooled_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the paired n7-versus-Shuffle tracking contrast.

    This is deliberately computed from the persisted pooled table.  It is a
    descriptive paired contrast over the 15 condition×task cells, not a
    replicate-level significance test.
    """
    pairs: list[dict[str, Any]] = []
    for condition_id, condition, family in CONDITIONS:
        for task in TASKS:
            n7 = next(
                row for row in pooled_rows
                if row["condition_id"] == condition_id
                and row["task"] == task
                and row["noise_id"] == "n7"
            )
            shuffle = next(
                row for row in pooled_rows
                if row["condition_id"] == condition_id
                and row["task"] == task
                and row["noise_id"] == "shuffle"
            )
            pair = {
                "condition_id": condition_id,
                "condition": condition,
                "family": family,
                "task": task,
                "pos_delta_cm": (
                    n7["track_pos_cm"] - shuffle["track_pos_cm"]
                ),
            }
            if (
                n7["track_ori_deg"] is not None
                and shuffle["track_ori_deg"] is not None
            ):
                pair["ori_delta_deg"] = (
                    n7["track_ori_deg"] - shuffle["track_ori_deg"]
                )
            if (
                n7["track_total"] is not None
                and shuffle["track_total"] is not None
            ):
                pair["total_delta"] = n7["track_total"] - shuffle["track_total"]
            pairs.append(pair)

    position_deltas = [pair["pos_delta_cm"] for pair in pairs]
    orientation_deltas = [
        pair["ori_delta_deg"] for pair in pairs
        if "ori_delta_deg" in pair
    ]
    total_deltas = [
        pair["total_delta"] for pair in pairs
        if "total_delta" in pair
    ]
    return {
        "pairs": pairs,
        "position_count": len(position_deltas),
        "position_positive": sum(delta > 0 for delta in position_deltas),
        "position_mean": sum(position_deltas) / len(position_deltas),
        "position_min": min(position_deltas),
        "position_max": max(position_deltas),
        "orientation_count": len(orientation_deltas),
        "orientation_positive": sum(delta > 0 for delta in orientation_deltas),
        "orientation_mean": (
            sum(orientation_deltas) / len(orientation_deltas)
            if orientation_deltas else None
        ),
        "total_count": len(total_deltas),
        "total_positive": sum(delta > 0 for delta in total_deltas),
        "total_mean": sum(total_deltas) / len(total_deltas) if total_deltas else None,
    }


def _position_tracking_correlations(
    pooled_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Correlate actual numeric noise with position tracking in the table."""
    correlations: dict[str, float] = {}
    for condition_id, _, _ in CONDITIONS:
        xs: list[float] = []
        ys: list[float] = []
        for row in pooled_rows:
            if (
                row["condition_id"] == condition_id
                and row["noise_id"] in NUMERIC_IDS
                and row["track_pos_cm"] is not None
            ):
                xs.append(float(row["pos_std_mm"]))
                ys.append(float(row["track_pos_cm"]))
        correlations[condition_id] = _pearson(xs, ys)
    return correlations


def _skill_rate(
    skill_type_rows: list[dict[str, Any]],
    condition_id: str,
    task: str,
    skill_type: str,
    noise_id: str,
) -> float:
    row = next(
        row for row in skill_type_rows
        if row["condition_id"] == condition_id
        and row["task"] == task
        and row["skill_type"] == skill_type
        and row["noise_id"] == noise_id
        and row["replicate_id"] == "pooled"
    )
    return float(row["success_rate"])


def _all_skill_rates(
    skill_type_rows: list[dict[str, Any]],
    skill_type: str,
    noise_id: str,
) -> float:
    rows = [
        row for row in skill_type_rows
        if row["skill_type"] == skill_type
        and row["noise_id"] == noise_id
        and row["replicate_id"] == "pooled"
    ]
    completed = sum(row["completed"] for row in rows)
    entered = sum(row["entered"] for row in rows)
    return completed / entered if entered else float("nan")


def generate_vlm_cover_108_report(
    *, manifest_path: Path, legacy_manifest_path: Path, report_path: Path,
    figures_dir: Path, data_dir: Path,
) -> None:
    selected = _expected_rows(_read_jsonl(manifest_path), _read_jsonl(legacy_manifest_path))
    (
        replicate_rows,
        pooled_rows,
        skill_rows,
        skill_type_rows,
        diagnostics_rows,
    ) = _build_products(selected)
    overall_replicate_rows = []
    for condition_id, condition, family in CONDITIONS:
        noise_ids = [*NUMERIC_IDS, "shuffle"] + (["r180"] if family == "grasp-part" else [])
        for noise_id in noise_ids:
            for seed in (0, 1, 2):
                rows = [row for row in replicate_rows if row["condition_id"] == condition_id and row["noise_id"] == noise_id and row["replicate_id"] == seed]
                successes = sum(row["n_success"] for row in rows)
                count = sum(row["n_rollouts"] for row in rows)
                low, high = _wilson(successes, count)
                overall_replicate_rows.append({
                    "condition_id": condition_id, "condition": condition,
                    "noise_id": noise_id, "replicate_id": seed,
                    "simulator_seed": seed, "annotation_seed": seed,
                    "n_success": successes, "n_rollouts": count,
                    "success_rate": successes / count,
                    "wilson_low": low, "wilson_high": high,
                })
    table_paths = {
        "success_by_replicate": data_dir / "success_by_replicate.csv",
        "success_overall_by_replicate": data_dir / "success_overall_by_replicate.csv",
        "success_tracking_pooled": data_dir / "success_tracking_pooled.csv",
        "three_task_pooled": data_dir / "three_task_pooled.csv",
        "vlm_sigma_by_task": data_dir / "vlm_sigma_by_task.csv",
        "vlm_sigma_3task_pooled": data_dir / "vlm_sigma_3task_pooled.csv",
        "vlm_skill_error_reference": data_dir / "vlm_skill_error_reference.csv",
        "vlm_skill_error_reference_pooled": (
            data_dir / "vlm_skill_error_reference_pooled.csv"
        ),
        "vlm_orientation_tracking_equivalent": (
            data_dir / "vlm_orientation_tracking_equivalent.csv"
        ),
        "noise_schedule": data_dir / "noise_schedule.csv",
        "skill_progression": data_dir / "skill_progression_replicate_and_pooled.csv",
        "skill_type_replicate_and_pooled": data_dir / "skill_type_replicate_and_pooled.csv",
        "diagnostics": data_dir / "annotation_noise_diagnostics.csv",
    }
    vlm_sigma_rows = _vlm_sigma_rows()
    vlm_sigma_3task_rows = _vlm_sigma_3task_rows(vlm_sigma_rows)
    noise_schedule_rows = _noise_schedule_rows()
    _write_csv(table_paths["success_by_replicate"], replicate_rows)
    _write_csv(table_paths["success_overall_by_replicate"], overall_replicate_rows)
    _write_csv(table_paths["success_tracking_pooled"], pooled_rows)
    three_task_rows = _three_task_pooled_rows(pooled_rows)
    _write_csv(table_paths["three_task_pooled"], three_task_rows)
    _write_csv(table_paths["vlm_sigma_by_task"], vlm_sigma_rows)
    _write_csv(table_paths["vlm_sigma_3task_pooled"], vlm_sigma_3task_rows)
    if not table_paths["vlm_skill_error_reference"].is_file():
        raise ValueError(
            "missing VLM skill error table: "
            f"{table_paths['vlm_skill_error_reference']}"
        )
    if not table_paths["vlm_skill_error_reference_pooled"].is_file():
        raise ValueError(
            "missing pooled VLM skill error table: "
            f"{table_paths['vlm_skill_error_reference_pooled']}"
        )
    orientation_equivalent_rows = _orientation_tracking_equivalent_rows(
        pooled_rows
    )
    _write_csv(
        table_paths["vlm_orientation_tracking_equivalent"],
        orientation_equivalent_rows,
    )
    _write_csv(table_paths["noise_schedule"], noise_schedule_rows)
    _write_csv(table_paths["skill_progression"], skill_rows)
    _write_csv(table_paths["skill_type_replicate_and_pooled"], skill_type_rows)
    _write_csv(table_paths["diagnostics"], diagnostics_rows)

    # The table layer is the only interface consumed by validation, report text,
    # and figures after the JSON-derived products have been persisted.
    replicate_rows = _read_csv(table_paths["success_by_replicate"])
    overall_replicate_rows = _read_csv(table_paths["success_overall_by_replicate"])
    pooled_rows = _read_csv(table_paths["success_tracking_pooled"])
    three_task_rows = _read_csv(table_paths["three_task_pooled"])
    vlm_sigma_rows = _read_csv(table_paths["vlm_sigma_by_task"])
    vlm_sigma_3task_rows = _read_csv(table_paths["vlm_sigma_3task_pooled"])
    vlm_skill_error_rows = _read_csv(table_paths["vlm_skill_error_reference"])
    vlm_skill_error_pooled_rows = _read_csv(
        table_paths["vlm_skill_error_reference_pooled"]
    )
    orientation_equivalent_rows = _read_csv(
        table_paths["vlm_orientation_tracking_equivalent"]
    )
    noise_schedule_rows = _read_csv(table_paths["noise_schedule"])
    skill_rows = _read_csv(table_paths["skill_progression"])
    skill_type_rows = _read_csv(table_paths["skill_type_replicate_and_pooled"])
    diagnostics_rows = _read_csv(table_paths["diagnostics"])
    table_validation_rows = _validate_table_products(
        replicate_rows,
        pooled_rows,
        overall_replicate_rows,
        vlm_sigma_rows,
        skill_type_rows,
        three_task_rows,
        noise_schedule_rows,
        vlm_sigma_3task_rows,
        orientation_equivalent_rows,
        vlm_skill_error_rows,
    )
    validation_path = data_dir / "table_validation.csv"
    _write_csv(validation_path, table_validation_rows)
    trend_rows = _trend_rows(pooled_rows)
    _write_csv(data_dir / "ordinal_trends.csv", trend_rows)

    numeric_figure = _plot_success(
        table_paths["success_tracking_pooled"],
        table_paths["vlm_sigma_by_task"],
        figures_dir,
    )
    tracking_figure = _plot_tracking_error(
        table_paths["success_tracking_pooled"],
        figures_dir,
    )
    three_task_figures = _plot_three_task_summary(
        table_paths["three_task_pooled"],
        table_paths["vlm_sigma_3task_pooled"],
        figures_dir,
    )
    skill_figures = _plot_skill_grids(
        table_paths["skill_type_replicate_and_pooled"],
        table_paths["vlm_sigma_by_task"],
        table_paths["vlm_skill_error_reference"],
        figures_dir,
    )
    extreme_error_figure = (
        report_path.parent.parent
        / "logs"
        / "vlm_extreme_error_20260910"
        / "generated"
        / "vlm_largest_errors_global.png"
    )
    data_index_path = data_dir / "data_index.json"
    data_index_path.parent.mkdir(parents=True, exist_ok=True)
    data_index_path.write_text(
        json.dumps(
            {
                "pipeline": "json -> tables -> figures",
                "raw_json_manifests": [
                    str(manifest_path),
                    str(legacy_manifest_path),
                ],
                "tables": {
                    name: str(path)
                    for name, path in table_paths.items()
                },
                "table_validation": str(validation_path),
                "figures_read_only_sources": [
                    "success_tracking_pooled.csv",
                    "three_task_pooled.csv",
                    "vlm_sigma_by_task.csv",
                    "vlm_sigma_3task_pooled.csv",
                    "vlm_skill_error_reference.csv",
                    "vlm_skill_error_reference_pooled.csv",
                    "vlm_orientation_tracking_equivalent.csv",
                    "skill_type_replicate_and_pooled.csv",
                ],
                "illustrative_assets": {
                    "vlm_largest_errors_global": str(
                        extreme_error_figure
                    )
                    if extreme_error_figure.is_file()
                    else None,
                },
                "figures": {
                    "numeric_png": str(numeric_figure),
                    "numeric_svg": str(numeric_figure.with_suffix(".svg")),
                    "numeric_pdf": str(numeric_figure.with_suffix(".pdf")),
                    "tracking_error_png": str(tracking_figure),
                    "tracking_error_svg": str(tracking_figure.with_suffix(".svg")),
                    "tracking_error_pdf": str(tracking_figure.with_suffix(".pdf")),
                    **{
                        f"{suffix}_{extension}": str(
                            path.with_suffix(f".{extension}")
                        )
                        for suffix, path in three_task_figures.items()
                        for extension in ("png", "svg", "pdf")
                    },
                    **{
                        f"{suffix}_{extension}": str(
                            path.with_suffix(f".{extension}")
                        )
                        for suffix, path in skill_figures.items()
                        for extension in ("png", "svg", "pdf")
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    overall_rows = []
    for condition_id, condition, _ in CONDITIONS:
        noise_ids = [*NUMERIC_IDS, "shuffle"] + (["r180"] if condition_id.startswith("grasp_part") else [])
        for noise_id in noise_ids:
            rows = [row for row in pooled_rows if row["condition_id"] == condition_id and row["noise_id"] == noise_id]
            successes = sum(row["n_success"] for row in rows)
            count = sum(row["n_rollouts"] for row in rows)
            low, high = _wilson(successes, count)
            overall_rows.append({
                "condition_id": condition_id,
                "condition": condition, "noise_id": noise_id,
                "n_success": successes,
                "n_rollouts": count,
                "success_rate": successes / count,
                "success": f"{successes}/{count} ({100*successes/count:.1f}%)",
                "wilson": f"[{100*low:.1f}, {100*high:.1f}]%",
            })

    overall_index = {
        (row["condition_id"], row["noise_id"]): row for row in overall_rows
    }
    pooled_index = {
        (row["condition_id"], row["noise_id"], row["task"]): row
        for row in pooled_rows
    }
    overall_replicate_index = {
        (row["condition_id"], row["noise_id"], row["replicate_id"]): row
        for row in overall_replicate_rows
    }
    headline_rows = []
    for condition_id, condition, _ in CONDITIONS:
        headline_rows.append({
            "condition": condition,
            "n0": overall_index[(condition_id, "n0")]["success"],
            "n5": overall_index[(condition_id, "n5")]["success"],
            "n6": overall_index[(condition_id, "n6")]["success"],
            "n7": overall_index[(condition_id, "n7")]["success"],
            "shuffle": overall_index[(condition_id, "shuffle")]["success"],
            "r180": (
                overall_index[(condition_id, "r180")]["success"]
                if (condition_id, "r180") in overall_index
                else "—"
            ),
        })

    numeric_changes = []
    for condition_id, condition, _ in CONDITIONS:
        clean = overall_index[(condition_id, "n0")]["success_rate"]
        n7 = overall_index[(condition_id, "n7")]["success_rate"]
        numeric_changes.append(
            f"{condition} `{100*clean:.1f}%→{100*n7:.1f}%` "
            f"(`{100*(n7-clean):+.1f} pp`)"
        )
    grasp_noise_summaries = []
    for condition_id in ("grasp_part", "grasp_part_colored"):
        clean = overall_index[(condition_id, "n0")]
        best = max(
            (overall_index[(condition_id, noise_id)] for noise_id in NUMERIC_IDS[1:]),
            key=lambda row: row["success_rate"],
        )
        replicate_changes = []
        for seed in (0, 1, 2):
            clean_replicate = overall_replicate_index[(condition_id, "n0", seed)]
            best_replicate = overall_replicate_index[
                (condition_id, best["noise_id"], seed)
            ]
            replicate_changes.append(
                100.0
                * (
                    best_replicate["success_rate"]
                    - clean_replicate["success_rate"]
                )
            )
        grasp_noise_summaries.append(
            f"{clean['condition']} 的最佳 noisy 点为 {best['noise_id']} "
            f"`{100*best['success_rate']:.1f}%`，相对 n0 "
            f"`{100*(best['success_rate']-clean['success_rate']):+.1f} pp` "
            f"（净变化 `{best['n_success']-clean['n_success']:+d}/324`；"
            f"三个 replicate 分别为 "
            f"`{'/'.join(f'{value:+.1f}' for value in replicate_changes)} pp`）"
        )
    matched_gap_summaries = []
    for grasp_id, point_id in (
        ("grasp_part", "gp"),
        ("grasp_part_colored", "colored_gp"),
    ):
        gap_n0 = (
            overall_index[(grasp_id, "n0")]["success_rate"]
            - overall_index[(point_id, "n0")]["success_rate"]
        )
        gap_n7 = (
            overall_index[(grasp_id, "n7")]["success_rate"]
            - overall_index[(point_id, "n7")]["success_rate"]
        )
        matched_gap_summaries.append(
            f"{overall_index[(grasp_id, 'n0')]['condition']}−"
            f"{overall_index[(point_id, 'n0')]['condition']} "
            f"`{100*gap_n0:+.1f}→{100*gap_n7:+.1f} pp`"
        )

    tracking_comparison = _tracking_endpoint_comparison(pooled_rows)
    tracking_correlations = _position_tracking_correlations(pooled_rows)
    low_noise_ranges: dict[str, tuple[float, float]] = {}
    full_noise_ranges: dict[str, tuple[float, float]] = {}
    for condition_id, _, _ in CONDITIONS:
        low_noise_rates = [
            overall_index[(condition_id, noise_id)]["success_rate"]
            for noise_id in NUMERIC_IDS[:5]
        ]
        full_noise_rates = [
            overall_index[(condition_id, noise_id)]["success_rate"]
            for noise_id in NUMERIC_IDS
        ]
        low_noise_ranges[condition_id] = (
            100.0 * (max(low_noise_rates) - min(low_noise_rates)),
            100.0 * float(np.std(low_noise_rates)),
        )
        full_noise_ranges[condition_id] = (
            100.0 * (max(full_noise_rates) - min(full_noise_rates)),
            100.0 * float(np.std(full_noise_rates)),
        )

    round_table_screw_n0 = _skill_rate(
        skill_type_rows, "gp", "round_table", "screw", "n0"
    )
    round_table_screw_n4 = _skill_rate(
        skill_type_rows, "gp", "round_table", "screw", "n4"
    )
    round_table_other_skill_changes = []
    for skill_type in ("push", "pick", "place", "insert"):
        start = _skill_rate(
            skill_type_rows, "gp", "round_table", skill_type, "n0"
        )
        end = _skill_rate(
            skill_type_rows, "gp", "round_table", skill_type, "n4"
        )
        round_table_other_skill_changes.append(
            f"{skill_type} `{100*start:.1f}%→{100*end:.1f}%`"
        )

    all_skill_n0_n7 = {
        skill_type: (
            _all_skill_rates(skill_type_rows, skill_type, "n0"),
            _all_skill_rates(skill_type_rows, skill_type, "n7"),
        )
        for skill_type in SKILL_TYPES
    }
    sigma_coverage = []
    for row in vlm_sigma_rows:
        sigma = float(row["equivalent_sigma_mm"])
        sigma_coverage.append(
            {
                "family": row["condition"],
                "task": row["task"],
                "sigma": sigma,
                "ratio": sigma / POSITION_MM["n7"],
            }
        )
    max_sigma = max(sigma_coverage, key=lambda row: row["sigma"])
    diagnostic_display = []
    for row in diagnostics_rows:
        diagnostic_display.append({
            "condition": row["condition"], "noise": row["noise_id"], "task": row["task"],
            "diag_n": row["diagnostics_rollout_n"],
            "pos_rms": "N/A" if row["position_norm_rms_mm"] is None else f"{row['position_norm_rms_mm']:.1f}",
            "pos_p90": "N/A" if row["position_norm_p90_mm"] is None else f"{row['position_norm_p90_mm']:.1f}",
            "rot_rms": "N/A" if row["rotation_geodesic_rms_deg"] is None else f"{row['rotation_geodesic_rms_deg']:.1f}",
            "rot_p90": "N/A" if row["rotation_geodesic_p90_deg"] is None else f"{row['rotation_geodesic_p90_deg']:.1f}",
            "workspace": f"{100*row['workspace_valid_rate']:.1f}%" if row["workspace_valid_rate"] is not None else "N/A",
            "visible": f"{100*row['front_projection_visible_rate']:.1f}%" if row["front_projection_visible_rate"] is not None else "N/A",
            "invalid": f"{100*row['invalid_nonfinite_rate']:.1f}%" if row["invalid_nonfinite_rate"] is not None else "N/A",
        })
    vlm_sigma_display = [
        {
            "condition": row["condition"],
            "task": row["task"],
            "source_conditions": row["source_conditions"],
            "source_rollouts": row["source_rollouts"],
            "valid_pairs": row["valid_pairs"],
            "sigma": f"{row['equivalent_sigma_mm']:.2f}",
        }
        for row in vlm_sigma_rows
    ]
    vlm_sigma_3task_display = [
        {
            "condition": row["condition"],
            "task": "3 task pooled",
            "source_conditions": row["source_conditions"],
            "source_rollouts": row["source_rollouts"],
            "valid_pairs": row["valid_pairs"],
            "sigma": f"{row['equivalent_sigma_mm']:.2f}",
        }
        for row in vlm_sigma_3task_rows
    ]
    orientation_equivalent_display = [
        {
            "condition": row["condition"],
            "task": row["task"],
            "tracking_error": f"{row['tracking_error_mean_deg']:.2f}",
            "matched_level": row["matched_noise_level"],
            "orientation_sigma": f"{row['orientation_equivalent_sigma_deg']:.2f}",
            "position": f"{row['position_equivalent_mm']:.2f}",
            "tracking_n": row["tracking_state_count"],
        }
        for row in orientation_equivalent_rows
    ]
    vlm_skill_error_display = []
    for skill in SKILL_TYPES:
        row = next(item for item in vlm_skill_error_pooled_rows if item["skill"] == skill)
        vlm_skill_error_display.append({
            "skill": row["skill"],
            "valid_pairs": f"{row['valid_pairs']:,}",
            "p50": f"{row['p50_error_px']:.1f} px / {row['equivalent_p50_mm']:.1f} mm",
            "rms": f"{row['rmse_error_px']:.1f} px / {row['equivalent_rms_mm']:.1f} mm",
            "p95": f"{row['p95_error_px']:.1f} px / {row['equivalent_p95_mm']:.1f} mm",
            "tail40": f"{100*row['tail_gt_40_fraction']:.1f}%",
            "tail70": f"{100*row['tail_gt_70_fraction']:.1f}%",
            "tail100": f"{100*row['tail_gt_100_fraction']:.1f}%",
        })
    max_skill_p95 = max(
        vlm_skill_error_rows,
        key=lambda row: float(row["equivalent_p95_mm"]),
    )
    noise_schedule_display = [
        {
            "noise_id": row["noise_id"],
            "position": f"{row['position_sigma_mm_per_axis']:.1f}",
            "orientation": f"{row['orientation_sigma_deg']:.1f}",
            "role": row["schedule_role"],
        }
        for row in noise_schedule_rows
    ]

    replicate_display = [{
        "condition": row["condition"], "noise": row["noise_id"], "task": row["task"],
        "rep": row["replicate_id"],
        "result": f"{row['n_success']}/{row['n_rollouts']} ({100*row['success_rate']:.1f}%)",
        "wilson": f"[{100*row['wilson_low']:.1f}, {100*row['wilson_high']:.1f}]%",
    } for row in replicate_rows]
    overall_replicate_display = [{
        "condition": row["condition"], "noise": row["noise_id"],
        "rep": row["replicate_id"],
        "result": f"{row['n_success']}/{row['n_rollouts']} ({100*row['success_rate']:.1f}%)",
        "wilson": f"[{100*row['wilson_low']:.1f}, {100*row['wilson_high']:.1f}]%",
    } for row in overall_replicate_rows]

    fresh36_comparison_rows = [
        {
            "claim": "colored GP 在数值噪声下最稳定",
            "verdict": "部分保留",
            "evidence": (
                f"低噪声 n0–n4 的 pooled range 为 "
                f"{low_noise_ranges['colored_gp'][0]:.1f} pp，略小于 GP "
                f"{low_noise_ranges['gp'][0]:.1f} pp；扩展到 n0–n7 后则为 "
                f"{full_noise_ranges['colored_gp'][0]:.1f} pp vs "
                f"{full_noise_ranges['gp'][0]:.1f} pp，优势不再保持。"
            ),
        },
        {
            "claim": "GP 与 colored GP 可能使用不同 tracking 机制",
            "verdict": "仍不支持机制区分",
            "evidence": (
                f"108 表中 position tracking 与数值噪声的 pooled-cell "
                f"Pearson r 为 GP {tracking_correlations['gp']:.3f}、"
                f"colored GP {tracking_correlations['colored_gp']:.3f}；"
                "两者都随噪声近似单调增大，但这只能说明共享的行为响应，"
                "不能识别内部机制差异。"
            ),
        },
        {
            "claim": "GP round_table 的提升集中在第一段 screw",
            "verdict": "复现并更清楚",
            "evidence": (
                f"round_table task success 从 n0 的 "
                f"{100*pooled_index[('gp', 'n0', 'round_table')]['success_rate']:.1f}% "
                f"升至 n7 的 {100*pooled_index[('gp', 'n7', 'round_table')]['success_rate']:.1f}%；"
                f"skill-level screw 在 n0→n4 为 "
                f"{100*round_table_screw_n0:.1f}%→"
                f"{100*round_table_screw_n4:.1f}%，而其余 skill 为 "
                f"{'；'.join(round_table_other_skill_changes)}。"
            ),
        },
        {
            "claim": "GP+skill 对连续数值噪声不稳定，并可能拒绝错误 guidance",
            "verdict": "成功率结论未复现，机制解释仍待验证",
            "evidence": (
                f"GP+skill 的 n0–n4 pooled range 仅 "
                f"{low_noise_ranges['gp_skill'][0]:.1f} pp，n0→n7 为 "
                f"{100*(overall_index[('gp_skill', 'n7')]['success_rate']-overall_index[('gp_skill', 'n0')]['success_rate']):+.1f} pp，"
                f"n7→Shuffle 为 "
                f"{100*(overall_index[('gp_skill', 'shuffle')]['success_rate']-overall_index[('gp_skill', 'n7')]['success_rate']):+.1f} pp；"
                "因此不能仅凭成功率宣称 gating。"
            ),
        },
        {
            "claim": "grasp 能容忍数值噪声，但 task variation 更大",
            "verdict": "支持",
            "evidence": (
                f"grasp-part 与 colored grasp-part 的 n0→n7 pooled success "
                f"分别为 {100*(overall_index[('grasp_part', 'n7')]['success_rate']-overall_index[('grasp_part', 'n0')]['success_rate']):+.1f} pp "
                f"和 {100*(overall_index[('grasp_part_colored', 'n7')]['success_rate']-overall_index[('grasp_part_colored', 'n0')]['success_rate']):+.1f} pp；"
                f"colored grasp-part 的 task-level n0–n7 range 最大为 "
                f"{max(100*max(float(row['success_rate']) for row in pooled_rows if row['condition_id']=='grasp_part_colored' and row['task']==task and row['noise_id'] in NUMERIC_IDS) - 100*min(float(row['success_rate']) for row in pooled_rows if row['condition_id']=='grasp_part_colored' and row['task']==task and row['noise_id'] in NUMERIC_IDS) for task in TASKS):.1f} pp。"
            ),
        },
        {
            "claim": "Shuffle 的成功率下降说明正确 semantic guidance 仍有用",
            "verdict": "成功率证据变弱，tracking 证据变强",
            "evidence": (
                "n7→Shuffle 的 overall success 变化为 "
                + "；".join(
                    f"{overall_index[(condition_id, 'shuffle')]['condition']} "
                    f"{100*(overall_index[(condition_id, 'shuffle')]['success_rate']-overall_index[(condition_id, 'n7')]['success_rate']):+.1f} pp"
                    for condition_id, _, _ in CONDITIONS
                )
                + f"；但 n7 的 position tracking 在 "
                f"{tracking_comparison['position_positive']}/"
                f"{tracking_comparison['position_count']} 个配对单元均高于 Shuffle，"
                f"平均差 {tracking_comparison['position_mean']:.1f} cm。"
            ),
        },
    ]

    rel_numeric = Path(os.path.relpath(numeric_figure, report_path.parent))
    rel_tracking = Path(
        os.path.relpath(tracking_figure, report_path.parent)
    )
    rel_three_task_success = Path(
        os.path.relpath(
            three_task_figures["success_3task_pooled"],
            report_path.parent,
        )
    )
    rel_three_task_tracking = Path(
        os.path.relpath(
            three_task_figures["tracking_position_3task_pooled"],
            report_path.parent,
        )
    )
    rel_extreme_error_figure = Path(
        os.path.relpath(extreme_error_figure, report_path.parent)
    )
    lines = [
        "# VLM 覆盖范围噪声补充实验（108 rollout/cell）", "",
        "> [!IMPORTANT]",
        "> **所有成功率都使用每 task 108 rollout。** 第一张关键结果表中的 Overall 是三个 task 合计，因此每格为 324 rollout；成功率表中不存在 72-rollout 结果。", "",
        "> [!NOTE]",
        "> `72` 只表示 tracking/diagnostics 的覆盖量：n0–n4 与 Shuffle 排除旧 seed-0 tracking 后，只汇总新 seed-1/2，即每 task `tracking_n=72`；n5–n7 与 r180 使用三个 replicate，为 `tracking_n=108`。旧 saved-8 tracking 未混入。", "",
        "> [!NOTE]",
        "> 数据索引固定为 `JSON → tables → figures`：JSON 只负责生成表，`table_validation.csv` 负责确认表内 eval 结果的分母、算术、pooled/replicate 一致性、skill-level 覆盖、三 task 合并和 VLM σ 覆盖；校验通过后，图只读取已校验的结果表。", "",
        "> grasp n0–n7 同时改变位置和旋转，只能解释为联合扰动。r180 是 orientation-only stress endpoint；VLM 角度指标是未处理 gripper/object symmetry 的 raw rotation error，r180 不等价于完整 VLM 错误分布。Shuffle 与 r180 均不进入连续趋势拟合。", "",
        "## 1. 结果图与主要结论", "",
        "### 1.1 task-level success（真实噪声尺度）", "",
        f"![Task-level success on the actual noise scale]({rel_numeric.as_posix()})", "",
        "横轴使用真实 position σ/axis（mm；n0–n7 分别为 0、3、6、12、24、48、96、192 mm），因此 n0–n4 会集中在低噪声段，不再等距排列。每个 task 右侧的灰色窄轴按 fresh36 的方式显示 `n7→Shuffle` categorical endpoint，Shuffle 不被当作连续 mm 噪声点。小 marker 与细连线只读取已校验的 `success_tracking_pooled.csv`，不直接 query JSON；本图不展示 replicate 离散度。每个 task 只画两条上游 position-equivalent VLM σ 线：蓝色 Point VLM、红色 Grasp VLM。Grasp 的 orientation-equivalent 对齐只在正文和附录中说明，不在图上增加第三条纵线。n7=192 mm/axis 覆盖最大的 task-level position σ；ordinal trend 导出见 `ordinal_trends.csv`。", "",
        "VLM σ 表示 VLM 点误差相当于多少 `mm/axis` 的 3D 位置噪声。统计单位是一个 control step 的有效 VLM–GT 点对；同一 task 内，三个 point condition 的 VLM 点合并计算一条 Point VLM position-equivalent σ 线，两个 grasp condition 的 VLM 点合并计算一条 Grasp VLM position-equivalent σ 线。Point 使用 `3×36=108` 条 source trajectory，Grasp 使用 `2×36=72` 条 source trajectory；每条线使用这些 trajectory 的全部有效 control-step pairs。Grasp 的 orientation-equivalent 对齐不在图中绘制，只在正文和附录报告；它不是 raw VLM orientation σ。", "",
        "对每个有效点对，先定义二维残差 `e_i = p_i^VLM − p_i^GT`，并计算 VLM 投影误差 `R_VLM = sqrt[(1/N) Σ_i ||e_i||₂²]`。参考噪声在同一帧的 GT 3D 点和相机标定上生成：`P_{i,n,j} = P_i^GT + σ_n z_{i,j}`，其中 `z_{i,j} ~ N(0, I₃)`，逐分量截断到 `[-2, 2]`，`σ_n ∈ {0, 3, 6, 12, 24} mm`；每个有效点对生成 `M=200` 个 Monte Carlo 样本，并在各噪声档之间复用同一批标准样本。", "",
        "将扰动点投影到前视相机，得到 `r_{i,n,j} = π(P_{i,n,j}) − π(P_i^GT)`，再计算参考投影误差 `R_n = sqrt[(1/(N·M)) Σ_i Σ_j ||r_{i,n,j}||₂²]`。将参考点按 `R_n` 排序；若相邻两点满足 `R_n ≤ R_VLM ≤ R_{n+1}`，则 `σ_eq = σ_n + [(R_VLM − R_n)/(R_{n+1} − R_n)]·(σ_{n+1} − σ_n)`；超过最高参考档时，使用最高两档线性外推。", "",
        "`pooled` 表示把同一 task、同一 VLM family 的 condition rows 合并，并按有效点对数加权：`σ_{f,t} = [Σ_c N_{c,t} σ_{c,t}] / [Σ_c N_{c,t}]`，其中 `N_{c,t}` 是有效 control-step pair 数。它不是不同 step 的 min–max，也不是把 Point 和 Grasp 混在一起；三个 task 分别计算，因此每个 task 的图上只有两条独立的 VLM σ 线。由于本轮 108-run 表没有保存原始 VLM residual vector，这里的 family-level σ 是对 formal diagnostic 的 condition×task Equivalent σ summary 做的 pair-weighted aggregation，而不是重新从 raw residual 逐点拟合。", "",
        "图中六条 task-specific position-equivalent 纵向线的统计量、有效 control-step pair 数和三 task pooled 汇总放在附录。Grasp 的 orientation-equivalent 对齐同样只作为文字与附录诊断，不在成功率或 pooled summary 图中展示。", "",
        "### 1.2 pooled tracking error（position / orientation / total）", "",
        f"![Pooled tracking error: position, orientation, and total]({rel_tracking.as_posix()})", "",
        "这张图按 fresh36 的指标生成：3 个 task × 3 个 tracking 指标（position error、orientation error、total error），其中 `total = pos_m / 0.01 + ori_deg / 5`，主轴显示 n0–n7 的真实 position σ/axis，右侧窄轴显示 `n7→Shuffle`。点和线均不展示 replicate 离散度；point 条件没有 pose-aware orientation/total tracking 定义，因此后两行只显示 grasp-part 条件。r180 保留在表格中，不单独画 endpoint 图。", "",
        "### 1.3 三 task 合并的 pooled summary", "",
        f"![Three-task pooled success]({rel_three_task_success.as_posix()})", "",
        f"![Three-task pooled position tracking]({rel_three_task_tracking.as_posix()})", "",
        "两张图把 `one_leg`、`round_table` 和 `lamp` 合并到同一条 condition 曲线。成功率按三个 task 的 rollout 数直接合并，因此每个 condition/noise cell 为 `3×108=324` 个 rollout；position tracking 按各 task 的有效 final skill-state 数加权，而不是简单平均三个 task 的均值。主轴显示真实 n0–n7 position σ/axis，右侧窄轴显示 `n7→Shuffle`；图不展示 replicate 离散度。pooled 图只保留两条三-task position-equivalent VLM 纵线：Point 与 Grasp。Grasp 的行为等效 orientation scale 只在正文和附录中说明，不在图上展示。三 task 合并表为 `three_task_pooled.csv`，VLM 标记读取 `vlm_sigma_3task_pooled.csv`。", "",
        "n0–n7 的 position 与 orientation 扰动是绑定的，而不是两个独立实验轴：", "",
        _markdown_table(noise_schedule_display, [("noise_id", "Level"), ("position", "Position σ/axis (mm)"), ("orientation", "Orientation σ (deg)"), ("role", "Schedule")]), "",
        "图下方的 orientation 对齐采用一个行为等效尺度。我们汇总 Grasp 在 n0（未注入 orientation noise）下、以 clean-GT 为参照的 orientation tracking residual，并按 `tracking_state_count` 加权。随后将该 residual 与固定的 `0/2.5/5/10/20/40/60/90°` orientation schedule 线性匹配，再用绑定的 position schedule 得到行为等效位置。该数值只在正文和附录报告，不在图上绘制；它表示下游策略在姿态误差下的行为等效覆盖位置，而不是 raw VLM orientation σ。", "",
        "### 1.4 主要结论", "",
        "- **n0→n7 没有出现共同的单调崩溃。** " + "；".join(numeric_changes) + "。各 condition 的 pooled Wilson 区间重叠，局部回升不能解释为噪声有益。",
        "- **当前数据不支持“噪声使 grasp 变得更好”。** " + "；".join(grasp_noise_summaries) + "。最佳点相对 clean 的差值很小，且没有跨 replicate 的一致增益。",
        "- **grasp 的相对数值优势随高噪声缩小，而非扩大。** 匹配对照的 n0→n7 gap 为 " + "；".join(matched_gap_summaries) + "。这支持“强噪声下没有崩溃”，不支持“grasp 比 point 更具相对噪声鲁棒性”。",
        "- **高噪声下成功率稳定不等于 annotation 仍然有效。** n7 已覆盖 VLM 最大等效位置误差，但 workspace-valid 与 front-visible 明显下降；成功 rollout 没有因 target 出界而删除。", "",
        "- **numeric noise 与 tracking error 呈现明显解耦。** 在所有 condition 和 task 中，position tracking 随 n0→n7 单调上升，而 pooled success 没有共同下降；这说明模型会承受目标点偏移，成功率本身不足以描述其控制行为。",
        "- **Shuffle 与 n7 的差异主要出现在 tracking，而不是 task success。** n7 的 position tracking 在 "
        + f"{tracking_comparison['position_positive']}/{tracking_comparison['position_count']} "
        + f"个配对单元均高于 Shuffle，平均高 "
        + f"{tracking_comparison['position_mean']:.1f} cm（范围 "
        + f"{tracking_comparison['position_min']:.1f}–{tracking_comparison['position_max']:.1f} cm）；"
        + f"grasp 的 orientation 和 total 平均分别高 "
        + f"{tracking_comparison['orientation_mean']:.1f}° 和 "
        + f"{tracking_comparison['total_mean']:.1f} 个 total 单位。"
        + "这一方向一致的差异支持模型会跟随 Shuffle 后的新 guidance，而不是简单忽略点。",
        "- **VLM σ 被 n7 的噪声范围覆盖。** n7 为 192 mm/axis；六条 task-level VLM σ 均不超过该范围，最大值为 "
        + f"{max_sigma['family']} {max_sigma['task']} 的 "
        + f"{max_sigma['sigma']:.2f} mm/axis（覆盖比 "
        + f"{max_sigma['ratio']:.3f}）。在此 benchmark 内，下游模型在覆盖上游 VLM 等效误差的压力下仍维持任务成功，支持双系统的 pipeline-level 可行性。",
        "",
        "### 1.5 fresh36 结论在 108 实验中的逐条复核", "",
        _markdown_table(
            fresh36_comparison_rows,
            [
                ("claim", "fresh36 结论"),
                ("verdict", "108 复核"),
                ("evidence", "108 证据"),
            ],
        ), "",
        "这里的“支持”表示 108 表中的方向和任务分解与 fresh36 一致；“部分保留”表示结论只在某一噪声区间成立；“机制解释仍待验证”表示现有汇总表能够显示行为差异，但没有 paired reset、donor-target 距离或 replicate-level tracking uncertainty 来完成因果区分。", "",
        "### 1.6 新的行为解释与边界", "",
        "**Shuffle 与大数值噪声检验的是两种不同的扰动。** 大数值噪声保留当前 semantic subtask，只把目标点沿同一指导语义推离；在 108 中，成功率在 n7 仍保持，而 tracking error 随噪声增大。Shuffle 则把点替换为另一 semantic state 的 guidance；此时 tracking error 反而较低，且 task success 没有系统性下降。一个与这些数据一致的解释是：模型确实使用点来组织动作。当点仍属于当前 subtask 但位置被大幅扰动时，模型尝试完成原 subtask，却必须承受错误目标；当点来自另一个 subtask 时，模型会跟随该点尝试另一个动作，因此终点更接近被置换后的 guidance。这个解释是行为层面的推断；要把它提升为机制证据，还需要记录 donor state、目标间距离，以及 paired reset 下的 episode-level trajectory。", "",
        "**上游 VLM 与下游控制的联动在本实验范围内是闭合的。** Point VLM 的 task-level 等效 σ 为 "
        + f"{min(row['sigma'] for row in sigma_coverage if row['family'] == 'Point VLM'):.2f}–"
        + f"{max(row['sigma'] for row in sigma_coverage if row['family'] == 'Point VLM'):.2f} mm/axis；"
        + "Grasp VLM 为 "
        + f"{min(row['sigma'] for row in sigma_coverage if row['family'] == 'Grasp VLM'):.2f}–"
        + f"{max(row['sigma'] for row in sigma_coverage if row['family'] == 'Grasp VLM'):.2f} mm/axis，"
        + "而 n7 使用 192 mm/axis。Point family 的 pooled success 从 n0 到 n7 为 "
        + f"{100*sum(int(row['n_success']) for row in pooled_rows if row['condition_id'] in VLM_FAMILY_CONDITIONS['point'] and row['noise_id'] == 'n0') / sum(int(row['n_rollouts']) for row in pooled_rows if row['condition_id'] in VLM_FAMILY_CONDITIONS['point'] and row['noise_id'] == 'n0'):.1f}%→"
        + f"{100*sum(int(row['n_success']) for row in pooled_rows if row['condition_id'] in VLM_FAMILY_CONDITIONS['point'] and row['noise_id'] == 'n7') / sum(int(row['n_rollouts']) for row in pooled_rows if row['condition_id'] in VLM_FAMILY_CONDITIONS['point'] and row['noise_id'] == 'n7'):.1f}%，"
        + "Grasp family 为 "
        + f"{100*sum(int(row['n_success']) for row in pooled_rows if row['condition_id'] in VLM_FAMILY_CONDITIONS['grasp'] and row['noise_id'] == 'n0') / sum(int(row['n_rollouts']) for row in pooled_rows if row['condition_id'] in VLM_FAMILY_CONDITIONS['grasp'] and row['noise_id'] == 'n0'):.1f}%→"
        + f"{100*sum(int(row['n_success']) for row in pooled_rows if row['condition_id'] in VLM_FAMILY_CONDITIONS['grasp'] and row['noise_id'] == 'n7') / sum(int(row['n_rollouts']) for row in pooled_rows if row['condition_id'] in VLM_FAMILY_CONDITIONS['grasp'] and row['noise_id'] == 'n7'):.1f}%。"
        + "因此数据支持“下游覆盖上游误差尺度”，但这仍是等效位置 σ 的 benchmark 结论，不等价于证明所有真实 VLM 误差分布、旋转对称性和时序相关性都已被覆盖。", "",
        "**skill-level 结果显示鲁棒性并不均匀。** 五类 skill 合并所有 task 与 condition 后，push 的 success rate 为 "
        + f"{100*all_skill_n0_n7['push'][0]:.1f}%→{100*all_skill_n0_n7['push'][1]:.1f}%，"
        + f"pick 为 {100*all_skill_n0_n7['pick'][0]:.1f}%→{100*all_skill_n0_n7['pick'][1]:.1f}%，"
        + f"place 为 {100*all_skill_n0_n7['place'][0]:.1f}%→{100*all_skill_n0_n7['place'][1]:.1f}%，"
        + f"insert 为 {100*all_skill_n0_n7['insert'][0]:.1f}%→{100*all_skill_n0_n7['insert'][1]:.1f}%，"
        + f"screw 为 {100*all_skill_n0_n7['screw'][0]:.1f}%→{100*all_skill_n0_n7['screw'][1]:.1f}%。"
        + "place 是整体较弱且在高噪声下略降的 skill；push、pick 和 insert 更稳定；screw 的 pooled success 可上升，但其变化受 task progression 与进入后续 skill 的选择效应影响，不能直接解释为噪声带来的能力提升。", "",
        "上述 tracking 差异是跨 task、跨 condition 的一致性描述，不是 replicate-level 显著性检验；当前图按要求不展示 replicate 离散度。因此文中使用“支持”“一致于”“提示”，不使用未经检验的“证明机制”。", "",
        "### 1.7 skill-level success rate 与 tracking error（5 skills × 3 tasks）", "",
        "Success rate", "",
        f"![Skill-level success rate]({Path(os.path.relpath(skill_figures['skill_success_rate'], report_path.parent)).as_posix()})", "",
        "Position error", "",
        f"![Skill-level position tracking error]({Path(os.path.relpath(skill_figures['tracking_position'], report_path.parent)).as_posix()})", "",
        "Orientation error", "",
        f"![Skill-level orientation tracking error]({Path(os.path.relpath(skill_figures['tracking_orientation'], report_path.parent)).as_posix()})", "",
        "Total error", "",
        f"![Skill-level total tracking error]({Path(os.path.relpath(skill_figures['tracking_total'], report_path.parent)).as_posix()})", "",
        "四张图沿用 fresh36 的 cascading skill 定义：每个子图是一种 skill type（`push/pick/place/insert/screw`）和一个 task，曲线表示不同 condition；success rate 为 `completed/entered`，不是 task success。tracking 统计每个 skill state 的最终有效段，并按同一 skill type 汇总；n0–n4 与 Shuffle 排除旧 seed-0 tracking 后使用 72 条 tracking rollout，n5–n7 使用 108 条。主轴使用真实 n0–n7 position σ/axis，右侧窄轴按 `n7→Shuffle` 显示 categorical endpoint；不展示 replicate 离散度。Point 条件只有 position tracking，Grasp 条件同时显示 position、orientation 和 `total = pos_m / 0.01 + ori_deg / 5`。每个主轴还叠加三条 VLM 参考线：task-level Point VLM RMS-equivalent σ、task-level Grasp VLM RMS-equivalent σ，以及该 skill 的 Grasp VLM p95-equivalent；若 p95 超出 n7，红色点线在右边界截断并标记 `p95>n7`。这些参考线只读取已校验的 `vlm_sigma_by_task.csv` 与 `vlm_skill_error_reference.csv`。", "",
        "### 1.8 关键 Overall 数据（每格 3 task × 108 = 324 rollout）", "",
        _markdown_table(headline_rows, [("condition", "Condition"), ("n0", "n0"), ("n5", "n5"), ("n6", "n6"), ("n7", "n7"), ("shuffle", "Shuffle"), ("r180", "r180")]), "",
        "## 2. 完整 pooled success 与 95% Wilson CI", "",
        _markdown_table(overall_rows, [("condition", "Condition"), ("noise_id", "Noise"), ("success", "Success"), ("wilson", "95% Wilson CI")]), "",
        "## 3. VLM skill mismatch 与 OOD 长尾诊断", "",
        "前面的 rollout 可视化揭示了一个需要单独处理的问题：某些 `round_table/place` 帧里的 VLM 点误差只接近 n1–n2，但 formal Grasp VLM 的整体误差仍然更大，而且不同 skill 之间并不在同一误差水平上。这不是矛盾，而是两个分布被混在了一起。第一，VLM 的点预测本质上是按 skill 变化的目标回归：`pick`、`push`、`place`、`insert` 和 `screw` 对可见部位、遮挡关系和目标几何的要求不同；把它们 pooled 成一条 σ 会隐藏这种 skill mismatch。第二，rollout failure 会把下游状态带到 VLM 训练分布之外，例如姿态偏离、部分装配、遮挡和相机视角变化。OOD 状态会制造长尾误差；但长尾不只出现在失败 rollout 中，也会出现在最终成功的 rollout 中，因此它描述的是上游观测风险，而不是简单的 failure label。", "",
        "### 3.1 极端误差样例（formal Grasp VLM）", "",
        f"![Largest VLM point errors in formal rollouts]({rel_extreme_error_figure.as_posix()})", "",
        "图 3.1 | Formal Grasp VLM rollout 中最大的 2-D 点误差。每个小图取一个 rollout×skill 的最大有效残差；绿色圆圈是 scripted target，红色叉号是保存下来的 VLM point，标题中的 `success/failure` 是该 rollout 的最终状态。图中最大的误差为 `one_leg/pick` 的 185.8 px，其次为 `one_leg/screw` 的 184.4 px、`lamp/screw` 的 179.8 px 和 `one_leg/place` 的 173.6 px。该 montage 用来展示长尾的形态和 OOD 候选状态，不作为新的成功率或 σ 估计，也不替代 JSON → tables → figures 的定量链路。", "",
        "这张图也解释了为什么单独查看一张低误差的 `round_table/place` 帧会低估 VLM 的总体风险：局部帧可以接近 n1–n2，但在不同 skill、不同 rollout state 和 failure/OOD 状态下，VLM 点会出现远离 scripted target 的长尾偏移。", "",
        "### 3.2 Skill-level 长尾统计与 p95 选择", "",
        "正式 Grasp VLM 诊断按有效 VLM–GT control-step pair 汇总。这里的 `p50` 表示典型误差，`RMS` 对较大误差更敏感，`p95` 表示最坏的 5% 尾部；`>40/>70/>100 px` 是直接报告大误差占比。对应的 `mm` 数值把同一个像素误差映射到共同的 projected-Gaussian RMS reference，便于与 n0–n7 的位置噪声轴比较。", "",
        "本轮正式 diagnostic summary 只保存了 Grasp VLM 的 task×skill residual，因此表格中的 skill-specific 长尾来自 Grasp VLM；Point VLM 仍只以 task-level RMS-equivalent σ 进入图中的蓝色参考线。不能把蓝色线解释成五类 skill 各自的 Point VLM 误差。", "",
        "设每个有效点对的二维像素残差幅度为 `e_i = ||p_i^VLM − p_i^GT||₂`。`p95` 定义为满足 `P(e_i ≤ q) ≥ 0.95` 的最小 `q`；在有限样本中，它是按 `e_i` 从小到大排序后位于 95% 分位的位置。阈值尾部占比定义为 `tail_τ = N(e_i > τ) / N`。因此 p95 给出尾部边界，tail fraction 给出尾部质量，两者应同时报告。", "",
        _markdown_table(
            vlm_skill_error_display,
            [
                ("skill", "Skill"),
                ("valid_pairs", "Valid pairs"),
                ("p50", "p50 (px / mm)"),
                ("rms", "RMS (px / mm)"),
                ("p95", "p95 (px / mm)"),
                ("tail40", ">40 px"),
                ("tail70", ">70 px"),
                ("tail100", ">100 px"),
            ],
        ), "",
        "这张表说明大误差并不是均匀分布的：`push` 的 >40 px、>70 px 和 >100 px 占比分别为 "
        + f"{100*next(row['tail_gt_40_fraction'] for row in vlm_skill_error_pooled_rows if row['skill'] == 'push'):.1f}%、"
        + f"{100*next(row['tail_gt_70_fraction'] for row in vlm_skill_error_pooled_rows if row['skill'] == 'push'):.1f}% 和 "
        + f"{100*next(row['tail_gt_100_fraction'] for row in vlm_skill_error_pooled_rows if row['skill'] == 'push'):.1f}%；"
        + "`pick` 的对应比例为 "
        + f"{100*next(row['tail_gt_40_fraction'] for row in vlm_skill_error_pooled_rows if row['skill'] == 'pick'):.1f}%、"
        + f"{100*next(row['tail_gt_70_fraction'] for row in vlm_skill_error_pooled_rows if row['skill'] == 'pick'):.1f}% 和 "
        + f"{100*next(row['tail_gt_100_fraction'] for row in vlm_skill_error_pooled_rows if row['skill'] == 'pick'):.1f}%。"
        + "因此“某一张图看起来像 n1–n2”不能代表所有 skill 的 VLM 误差等级。", "",
        "p95 比均值更能表达 OOD 风险，因为它不会被大量小误差稀释；但 p95 不能单独作为噪声 σ。它只保留尾部位置，忽略了尾部以下的误差质量，而且部分 skill 的 p95-equivalent 已超过 n7=192 mm/axis，例如 "
        + f"{max_skill_p95['skill']} 的 {max_skill_p95['task']} p95-equivalent 为 {max_skill_p95['equivalent_p95_mm']:.1f} mm/axis。"
        + "因此本报告采用 `p50 + RMS + p95 + tail fraction` 的四件套：RMS-equivalent 作为主要 VLM σ 线，p95-equivalent 作为 skill-specific stress line，大误差占比用来说明尾部质量。", "",
        "在所有 skill-level 图中，主轴现在标出三条纵向参考线：蓝色虚线是 task-level Point VLM RMS-equivalent σ，红色虚线是 task-level Grasp VLM RMS-equivalent σ，红色点线是该 skill 的 Grasp VLM p95-equivalent。p95 超过 n7 时，点线在图的右边界截断，并标注 `p95>n7`；这表示尾部已经超出当前噪声设计，而不是把超出的值伪装成 n7。", "",
        "该诊断也限定了结论边界：108 实验可以证明下游策略在已测 n0–n7 范围内能够承受上游 VLM 的典型误差，并显示哪些 skill 的 VLM 尾部更危险；它不能证明 rollout failure/OOD 状态下的 VLM 误差已经被完整覆盖。下一步应优先按 skill 和 rollout state 分层报告 failure/OOD 比例，并保留每个 control step 的 VLM–GT residual，而不是只保留一个 pooled σ。", "",
        "## 4. 三个 replicate 的独立结果", "",
        "### 4.1 Overall（每 replicate 合并三个 task，n=108）", "",
        _markdown_table(overall_replicate_display, [("condition", "Condition"), ("noise", "Noise"), ("rep", "Replicate"), ("result", "Success"), ("wilson", "95% Wilson CI")]), "",
        "### 4.2 By task（每格 n=36）", "",
        _markdown_table(replicate_display, [("condition", "Condition"), ("noise", "Noise"), ("task", "Task"), ("rep", "Replicate"), ("result", "Success"), ("wilson", "95% Wilson CI")]), "",
        "## 5. 实际扰动、tracking 覆盖与可见性", "",
        "下表的 `diagnostics_n` 不是成功率分母。n0–n4/Shuffle 的 `72` 来自新 seed-1/2；n5–n7/r180 的 `108` 来自三个 replicate。所有 success cell 仍为 108 rollout，且不因 target 越界而删除。可见率下降时，应将数值偏移与 annotation 消失/出界分别解释。", "",
        _markdown_table(diagnostic_display, [("condition", "Condition"), ("noise", "Noise"), ("task", "Task"), ("diag_n", "diagnostics_n"), ("pos_rms", "Position norm RMS (mm)"), ("pos_p90", "Position norm P90 (mm)"), ("rot_rms", "Rotation geodesic RMS (deg)"), ("rot_p90", "Rotation geodesic P90 (deg)"), ("workspace", "Workspace valid"), ("visible", "Front visible"), ("invalid", "Invalid/non-finite")]), "",
        "## 附录 A：VLM position-equivalent σ 与 orientation 对齐", "",
        "下表是图中 position-equivalent VLM σ 的完整 anchor table。它们只用于把真实 VLM 误差与 n0–n7 的数值噪声尺度对齐，不是额外的成功率观测。", "",
        _markdown_table(vlm_sigma_display, [("condition", "VLM family"), ("task", "Task"), ("source_conditions", "Combined conditions"), ("source_rollouts", "Source trajectories"), ("valid_pairs", "Valid control-step pairs"), ("sigma", "Equivalent position σ (mm/axis)")]), "",
        "三 task pooled 的 position-equivalent σ 按有效 control-step pair 数加权，供 pooled summary 图中的两条纵线使用。", "",
        _markdown_table(vlm_sigma_3task_display, [("condition", "VLM family"), ("task", "Task"), ("source_conditions", "Combined conditions"), ("source_rollouts", "Source trajectories"), ("valid_pairs", "Valid control-step pairs"), ("sigma", "Equivalent position σ (mm/axis)")]), "",
        "Grasp 的 orientation-equivalent scale 不在图中展示。它由 n0（未注入 orientation noise）的 clean-GT orientation tracking residual 与固定 orientation schedule 匹配，再映射回绑定的 position schedule；该量表示下游策略的行为等效覆盖位置，不是 raw VLM orientation σ。", "",
        _markdown_table(orientation_equivalent_display, [("condition", "Source"), ("task", "Task"), ("tracking_error", "n0 tracking error (deg)"), ("matched_level", "Matched orientation level"), ("orientation_sigma", "Behavioral orientation-equivalent σ (deg)"), ("position", "Position-axis coordinate (mm)"), ("tracking_n", "Tracking states")]), "",
        "## 6. 数据产品", "",
        "- `success_by_replicate.csv`: 三个 replicate 的 task success 与 Wilson CI。",
        "- `success_overall_by_replicate.csv`: 三个 replicate 的 overall success 与 Wilson CI。",
        "- `success_tracking_pooled.csv`: pooled success 与正式 tracking（含 72/108 样本量）。",
        "- `three_task_pooled.csv`: 将三个 task 合并后的 success 与 position tracking summary；success 按 rollout 合并，tracking 按有效 skill-state 数加权。",
        "- `vlm_sigma_by_task.csv`: 每个 VLM family×task 的合并条件、source trajectory 数、有效 control-step pair 数和 task-level Equivalent σ。",
        "- `vlm_sigma_3task_pooled.csv`: Point/Grasp family 跨三个 task 的 position-equivalent σ；按有效点对数加权，供 pooled summary 图的粗实线使用。",
        "- `vlm_orientation_tracking_equivalent.csv`: Grasp n0 clean-GT orientation tracking residual 映射到绑定 n0–n7 orientation schedule 的行为等效尺度；不冒充 raw VLM orientation residual。",
        "- `noise_schedule.csv`: 固定的 n0–n7 position/orientation 绑定幅度，以及 r180 orientation-only endpoint；用于解释 position 横轴与 orientation 扰动的对应关系。",
        "- `skill_progression_replicate_and_pooled.csv`: replicate 与 pooled skill-state success progression。",
        "- `vlm_skill_error_reference.csv`: formal Grasp VLM 按 task×skill 汇总的 p50、RMS、p95 像素误差、p95-equivalent 以及 >40/>70/>100 px 尾部占比；skill-level 图读取此表。",
        "- `vlm_skill_error_reference_pooled.csv`: 按 skill 跨 task/condition 的 valid-pair-weighted 汇总，供本章表格和文字使用。",
        "- `skill_type_replicate_and_pooled.csv`: 5 类 skill 的 pooled success rate、tracking position/orientation/total 及对应样本量；skill-level 图读取此表。",
        "- `annotation_noise_diagnostics.csv`: 实际扰动、workspace、projection 和 invalid 统计。",
        "- `ordinal_trends.csv`: 仅 n0–n7 的预先固定 ordinal 描述性趋势。", "",
        "- `table_validation.csv`: 对结果表的行唯一性、分母、成功率算术和 pooled/replicate 一致性校验；校验失败时不会生成图。",
        "- `data_index.json`: 固定 `json -> tables -> figures` 数据索引；task-level 图读取对应的已校验 tables，三 task 合并图读取 `three_task_pooled.csv`。", "",
        f"- New manifest: `{manifest_path}`", f"- Read-only legacy manifest: `{legacy_manifest_path}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
