"""Step-level VLM point error metrics against a shadow automaton."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


SKILLS = ("push", "pick", "place", "insert", "screw")


def make_point_error_record(
    *,
    step_idx: int,
    oracle_skill: str | None,
    oracle_point: Any,
    vlm_point: Any,
    query_step: int,
) -> dict[str, Any]:
    record = {
        "step_idx": int(step_idx),
        "oracle_skill": oracle_skill,
        "query_step": int(query_step),
        "cache_age_steps": int(step_idx - query_step),
        "oracle_point": None,
        "vlm_point": None,
        "error_px": None,
        "valid": False,
    }
    predicted = np.asarray(vlm_point, dtype=np.float64)
    if predicted.shape != (2,) or not np.isfinite(predicted).all():
        raise ValueError("invalid VLM point in metric record")
    record["vlm_point"] = predicted.tolist()
    if oracle_point is None:
        return record
    target = np.asarray(oracle_point, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all():
        return record
    record["oracle_point"] = target.tolist()
    record["error_px"] = float(np.linalg.norm(predicted - target))
    record["valid"] = True
    return record


def _summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    errors = [float(row["error_px"]) for row in rows if row.get("valid")]
    count_total = len(rows)
    count_valid = len(errors)
    sum_error = float(sum(errors))
    sum_squared = float(sum(error * error for error in errors))
    return {
        "mean_error_px": sum_error / count_valid if count_valid else None,
        "rmse_px": math.sqrt(sum_squared / count_valid) if count_valid else None,
        "count_valid": count_valid,
        "count_invalid_gt": count_total - count_valid,
        "count_total": count_total,
        "coverage": count_valid / count_total if count_total else None,
        "sum_error_px": sum_error,
        "sum_squared_error_px": sum_squared,
    }


def summarize_point_error_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _summarize(records),
        "by_skill": {
            skill: _summarize(
                row for row in records if row.get("oracle_skill") == skill
            )
            for skill in SKILLS
        },
    }


def build_vlm_point_error_summary(
    records_per_env: Sequence[Sequence[dict[str, Any]]],
    success_flags: Sequence[bool],
) -> dict[str, Any]:
    if len(records_per_env) != len(success_flags):
        raise ValueError("point-error env records/success flags mismatch")
    all_records = [row for episode in records_per_env for row in episode]
    success_records = [
        row
        for episode, success in zip(records_per_env, success_flags)
        if success
        for row in episode
    ]
    failure_records = [
        row
        for episode, success in zip(records_per_env, success_flags)
        if not success
        for row in episode
    ]
    return {
        "all": summarize_point_error_records(all_records),
        "success_only": summarize_point_error_records(success_records),
        "failure_only": summarize_point_error_records(failure_records),
    }


def merge_vlm_point_error_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def merge_stats(stats_list):
        count_valid = sum(int(stats.get("count_valid", 0)) for stats in stats_list)
        count_total = sum(int(stats.get("count_total", 0)) for stats in stats_list)
        sum_error = sum(float(stats.get("sum_error_px", 0.0)) for stats in stats_list)
        sum_squared = sum(
            float(stats.get("sum_squared_error_px", 0.0)) for stats in stats_list
        )
        return {
            "mean_error_px": sum_error / count_valid if count_valid else None,
            "rmse_px": math.sqrt(sum_squared / count_valid) if count_valid else None,
            "count_valid": count_valid,
            "count_invalid_gt": count_total - count_valid,
            "count_total": count_total,
            "coverage": count_valid / count_total if count_total else None,
            "sum_error_px": sum_error,
            "sum_squared_error_px": sum_squared,
        }

    output = {}
    for scope in ("all", "success_only", "failure_only"):
        scoped = [summary.get(scope, {}) for summary in summaries]
        output[scope] = {
            "overall": merge_stats([item.get("overall", {}) for item in scoped]),
            "by_skill": {
                skill: merge_stats(
                    [item.get("by_skill", {}).get(skill, {}) for item in scoped]
                )
                for skill in SKILLS
            },
        }
    return output
