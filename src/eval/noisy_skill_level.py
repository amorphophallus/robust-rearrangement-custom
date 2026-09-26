"""Paired metrics for noisy fixed-state skill-level evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def paired_endpoint_metrics(
    clean: Mapping[str, Any], noisy: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute endpoint metrics for one exact state/repeat pair.

    ``x0`` and ``x_delta`` are final end-effector positions under clean and
    noisy guidance. ``p0`` and ``p_delta`` are the clean and policy-input
    guidance targets recorded by the noisy rollout at its terminal frame.
    """

    x0 = np.asarray(clean["end_ee_pos_robot_base_m"], dtype=np.float64)
    x_delta = np.asarray(noisy["end_ee_pos_robot_base_m"], dtype=np.float64)
    p0 = np.asarray(noisy["p0_robot_base_m"], dtype=np.float64)
    p_delta = np.asarray(noisy["p_delta_robot_base_m"], dtype=np.float64)
    delta = p_delta - p0
    displacement = x_delta - x0
    delta_sq = float(np.dot(delta, delta))
    is_clean = delta_sq <= 1e-18
    return {
        "e_gt_m": float(np.linalg.norm(x_delta - p0)),
        "e_input_m": float(np.linalg.norm(x_delta - p_delta)),
        "delta_x_m": None if is_clean else float(np.linalg.norm(displacement)),
        "guidance_following_displacement_m": (
            None
            if is_clean
            else float(np.dot(displacement, delta) / np.sqrt(delta_sq))
        ),
        "guidance_gain": (
            None if is_clean else float(np.dot(displacement, delta) / delta_sq)
        ),
        "x0_robot_base_m": x0.tolist(),
        "x_delta_robot_base_m": x_delta.tolist(),
        "p0_robot_base_m": p0.tolist(),
        "p_delta_robot_base_m": p_delta.tolist(),
        "delta_robot_base_m": delta.tolist(),
        "delta_norm_m": float(np.sqrt(delta_sq)),
    }


def summarize_paired_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["eval_noise_level"])].append(row)
    return {
        level: {
            "attempted": len(items),
            "completed": sum(bool(item["completed_current_stage"]) for item in items),
            "success_rate": (
                sum(bool(item["completed_current_stage"]) for item in items) / len(items)
            ),
            "e_gt_m": summarize_paired_records_metric(items, "e_gt_m"),
            "e_input_m": summarize_paired_records_metric(items, "e_input_m"),
            "delta_x_m": summarize_paired_records_metric(items, "delta_x_m"),
            "guidance_following_displacement_m": summarize_paired_records_metric(
                items, "guidance_following_displacement_m"
            ),
            "guidance_following_positive_rate": summarize_positive_rate(
                items, "guidance_following_displacement_m"
            ),
            "guidance_gain": summarize_paired_records_metric(items, "guidance_gain"),
        }
        for level, items in sorted(grouped.items())
    }


def summarize_paired_records_metric(
    rows: Iterable[Mapping[str, Any]], name: str
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[name]) for row in rows if row.get(name) is not None],
        dtype=np.float64,
    )
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def summarize_positive_rate(
    rows: Iterable[Mapping[str, Any]], name: str
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[name]) for row in rows if row.get(name) is not None],
        dtype=np.float64,
    )
    if not len(values):
        return {"count": 0, "positive": 0, "rate": None}
    positive = int(np.count_nonzero(values > 0.0))
    return {
        "count": int(len(values)),
        "positive": positive,
        "rate": float(positive / len(values)),
    }
