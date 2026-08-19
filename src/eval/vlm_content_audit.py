"""Post-rollout checks for stale or collapsed VLM guidance content.

The control loop intentionally caches one VLM result for an action horizon, so
this audit only compares observations whose ``cache_age_steps`` is zero.  It is
kept separate from the step-weighted point-error metric: the latter measures
policy exposure, while this module checks whether fresh model outputs actually
respond to changing inputs and targets.
"""

from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import lzma
import math
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np


POINT_SCALE_1000_TO_PX = np.asarray((319.0 / 1000.0, 239.0 / 1000.0))
DEFAULT_EXPECTED_QUERY_INTERVAL = 8
DEFAULT_MIN_FRESH_QUERIES = 20
DEFAULT_MIN_GT_SPREAD_PX = 10.0
DEFAULT_MAX_SPREAD_RATIO = 0.35
DEFAULT_LARGE_GT_DISPLACEMENT_PX = 10.0
DEFAULT_MIN_LARGE_TRANSITIONS = 3
DEFAULT_MAX_TRANSITION_RESPONSE_RATIO = 0.35


def _load_pickle(path: Path) -> Any:
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as stream:
            return pickle.load(stream)
    with path.open("rb") as stream:
        return pickle.load(stream)


def _point(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return point


def _point_from_observation(observation: dict[str, Any], key: str) -> np.ndarray | None:
    mapping = observation.get(key) or {}
    if not isinstance(mapping, dict):
        return None
    return _point(mapping.get("color_image2"))


def _vector_stats(values: Iterable[np.ndarray]) -> dict[str, Any] | None:
    rows = list(values)
    if not rows:
        return None
    array = np.stack(rows)
    return {
        "count": int(len(array)),
        "mean_xy_px": np.mean(array, axis=0).tolist(),
        "std_xy_px": np.std(array, axis=0).tolist(),
        "range_xy_px": (np.max(array, axis=0) - np.min(array, axis=0)).tolist(),
        "spread_norm_px": float(np.linalg.norm(np.std(array, axis=0))),
    }


def _scalar_stats(values: Iterable[float]) -> dict[str, Any] | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return None
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _group_point_stats(records: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        label = record.get(label_key)
        if label is not None:
            grouped[str(label)].append(record["vlm_point"])
    return {
        label: _vector_stats(points)
        for label, points in sorted(grouped.items())
    }


def _fresh_records(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frame_idx, observation in enumerate(observations):
        metadata = observation.get("vlm_annotation") or {}
        if metadata.get("cache_age_steps") != 0:
            continue
        vlm_point = _point_from_observation(observation, "guidance_point_2d")
        point_1000 = _point(metadata.get("point_1000"))
        if vlm_point is None or point_1000 is None:
            continue
        oracle_point = _point_from_observation(
            observation, "oracle_guidance_point_2d"
        )
        image = observation.get("color_image2")
        image_hash = None
        if image is not None:
            image_array = np.asarray(image)
            image_hash = hashlib.sha256(image_array.tobytes()).hexdigest()
        records.append(
            {
                "frame_idx": int(frame_idx),
                "query_step": int(metadata.get("query_step", -1)),
                "request_id": str(metadata.get("request_id", "")),
                "model_revision": str(metadata.get("model_revision", "")),
                "vlm_skill": observation.get("skill"),
                "oracle_skill": observation.get("oracle_skill"),
                "vlm_point": vlm_point,
                "oracle_point": oracle_point,
                "point_1000": point_1000,
                "image_hash": image_hash,
            }
        )
    return records


def audit_rollout_pickle(
    path: str | Path,
    *,
    expected_query_interval: int = DEFAULT_EXPECTED_QUERY_INTERVAL,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit one saved episode and return its compact report plus fresh records."""

    pickle_path = Path(path)
    payload = _load_pickle(pickle_path)
    observations = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(observations, list):
        raise ValueError(f"rollout has no observation list: {pickle_path}")
    records = _fresh_records(observations)
    del payload, observations
    gc.collect()

    query_steps = [record["query_step"] for record in records]
    query_deltas = np.diff(query_steps) if len(query_steps) > 1 else np.asarray([])
    request_ids = [record["request_id"] for record in records]
    scale_errors = [
        float(
            np.linalg.norm(
                record["vlm_point"] - record["point_1000"] * POINT_SCALE_1000_TO_PX
            )
        )
        for record in records
    ]
    image_hashes = [record["image_hash"] for record in records if record["image_hash"]]
    revisions = sorted({record["model_revision"] for record in records})

    failures: list[str] = []
    if not records:
        failures.append("no fresh VLM queries")
    if len(request_ids) != len(set(request_ids)):
        failures.append("request IDs repeat within one rollout")
    if len(query_deltas) and not np.all(query_deltas == expected_query_interval):
        failures.append(
            f"fresh query interval is not consistently {expected_query_interval} steps"
        )
    if scale_errors and max(scale_errors) > 1e-3:
        failures.append("point_1000 to point_px conversion mismatch")
    if len(revisions) != 1:
        failures.append(f"model revision is not constant: {revisions}")

    report = {
        "path": str(pickle_path),
        "compressed": pickle_path.name.endswith(".pkl.xz"),
        "fresh_query_count": len(records),
        "request_ids_unique": len(request_ids) == len(set(request_ids)),
        "query_steps": {
            "first": query_steps[0] if query_steps else None,
            "last": query_steps[-1] if query_steps else None,
            "expected_interval": expected_query_interval,
            "interval_valid": bool(
                not len(query_deltas)
                or np.all(query_deltas == expected_query_interval)
            ),
        },
        "model_revisions": revisions,
        "point_scale_max_abs_error_px": max(scale_errors) if scale_errors else None,
        "saved_front_image_hashes": {
            "count": len(image_hashes),
            "unique": len(set(image_hashes)),
            "unique_fraction": (
                len(set(image_hashes)) / len(image_hashes) if image_hashes else None
            ),
            "note": (
                "Hashes use saved front frames (which may contain the display marker), "
                "not the original HTTP PNG bytes."
            ),
        },
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }
    return report, records


def _episode_transitions(
    records: list[dict[str, Any]],
    *,
    large_gt_displacement_px: float,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {
        "adjacent_vlm": [],
        "vlm_skill_change_vlm": [],
        "oracle_skill_change_vlm": [],
        "oracle_skill_change_gt": [],
        "large_gt_vlm": [],
        "large_gt_gt": [],
    }
    for previous, current in zip(records, records[1:]):
        vlm_displacement = float(
            np.linalg.norm(current["vlm_point"] - previous["vlm_point"])
        )
        result["adjacent_vlm"].append(vlm_displacement)
        if current.get("vlm_skill") != previous.get("vlm_skill"):
            result["vlm_skill_change_vlm"].append(vlm_displacement)
        previous_gt = previous.get("oracle_point")
        current_gt = current.get("oracle_point")
        if previous_gt is None or current_gt is None:
            continue
        gt_displacement = float(np.linalg.norm(current_gt - previous_gt))
        if current.get("oracle_skill") != previous.get("oracle_skill"):
            result["oracle_skill_change_vlm"].append(vlm_displacement)
            result["oracle_skill_change_gt"].append(gt_displacement)
        if gt_displacement >= large_gt_displacement_px:
            result["large_gt_vlm"].append(vlm_displacement)
            result["large_gt_gt"].append(gt_displacement)
    return result


def _discover_row_pickles(manifest: dict[str, Any], row: dict[str, Any]) -> list[Path]:
    data_dir_raw = Path(manifest["data_dir_raw"])
    base = (
        data_dir_raw
        / "raw"
        / "diffik"
        / "sim"
        / row["task"]
        / "rollout"
        / manifest.get("randomness", "low")
    )
    candidates = []
    if base.is_dir():
        for mode_dir in base.iterdir():
            candidate = mode_dir / Path(row["rollout_suffix"])
            if candidate.is_dir():
                candidates.append(candidate)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one rollout directory for {row['condition']}/{row['task']}, "
            f"found {len(candidates)} under {base}: {candidates}"
        )
    return sorted(
        path
        for path in candidates[0].rglob("*.pkl*")
        if path.name.endswith((".pkl", ".pkl.xz"))
    )


def audit_manifest_rollouts(
    manifest: dict[str, Any],
    *,
    expected_query_interval: int = DEFAULT_EXPECTED_QUERY_INTERVAL,
    min_fresh_queries: int = DEFAULT_MIN_FRESH_QUERIES,
    min_gt_spread_px: float = DEFAULT_MIN_GT_SPREAD_PX,
    max_spread_ratio: float = DEFAULT_MAX_SPREAD_RATIO,
    large_gt_displacement_px: float = DEFAULT_LARGE_GT_DISPLACEMENT_PX,
    min_large_transitions: int = DEFAULT_MIN_LARGE_TRANSITIONS,
    max_transition_response_ratio: float = DEFAULT_MAX_TRANSITION_RESPONSE_RATIO,
) -> dict[str, Any]:
    """Audit every manifest row and flag point-head regression-to-mean."""

    row_reports: list[dict[str, Any]] = []
    for row in manifest.get("runs", []):
        failures: list[str] = []
        try:
            pickle_paths = _discover_row_pickles(manifest, row)
        except (KeyError, FileNotFoundError) as exc:
            row_reports.append(
                {
                    "condition": row.get("condition"),
                    "task": row.get("task"),
                    "status": "failed",
                    "failures": [str(exc)],
                    "episodes": [],
                }
            )
            continue

        expected_rollouts = int(manifest.get("n_rollouts_per_task", 0))
        if len(pickle_paths) != expected_rollouts:
            failures.append(
                f"found {len(pickle_paths)} pickle artifacts, expected {expected_rollouts}"
            )
        compressed = [str(path) for path in pickle_paths if path.name.endswith(".pkl.xz")]
        if compressed:
            failures.append(f"compressed pickle artifacts are forbidden: {len(compressed)}")

        episode_reports: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []
        transition_values: dict[str, list[float]] = defaultdict(list)
        for pickle_path in pickle_paths:
            try:
                episode_report, records = audit_rollout_pickle(
                    pickle_path,
                    expected_query_interval=expected_query_interval,
                )
            except Exception as exc:
                episode_reports.append(
                    {
                        "path": str(pickle_path),
                        "status": "failed",
                        "failures": [f"unreadable rollout: {exc}"],
                    }
                )
                failures.append(f"unreadable rollout: {pickle_path}: {exc}")
                continue
            episode_reports.append(episode_report)
            if episode_report["status"] != "passed":
                failures.extend(
                    f"{pickle_path.name}: {reason}"
                    for reason in episode_report["failures"]
                )
            all_records.extend(records)
            transitions = _episode_transitions(
                records,
                large_gt_displacement_px=large_gt_displacement_px,
            )
            for key, values in transitions.items():
                transition_values[key].extend(values)

        valid_gt_records = [
            record for record in all_records if record.get("oracle_point") is not None
        ]
        vlm_stats = _vector_stats(record["vlm_point"] for record in valid_gt_records)
        gt_stats = _vector_stats(record["oracle_point"] for record in valid_gt_records)
        spread_ratio = None
        if vlm_stats and gt_stats and gt_stats["spread_norm_px"] > 0.0:
            spread_ratio = vlm_stats["spread_norm_px"] / gt_stats["spread_norm_px"]

        large_vlm = transition_values["large_gt_vlm"]
        large_gt = transition_values["large_gt_gt"]
        transition_response_ratio = None
        if large_gt and float(np.mean(large_gt)) > 0.0:
            transition_response_ratio = float(np.mean(large_vlm) / np.mean(large_gt))

        collapse_failures: list[str] = []
        if (
            len(valid_gt_records) >= min_fresh_queries
            and gt_stats is not None
            and gt_stats["spread_norm_px"] >= min_gt_spread_px
            and spread_ratio is not None
            and spread_ratio < max_spread_ratio
        ):
            collapse_failures.append(
                "fresh VLM point spread is too small relative to GT "
                f"({spread_ratio:.3f} < {max_spread_ratio:.3f})"
            )
        if (
            len(large_gt) >= min_large_transitions
            and transition_response_ratio is not None
            and transition_response_ratio < max_transition_response_ratio
        ):
            collapse_failures.append(
                "fresh VLM points under-react to large GT movements "
                f"({transition_response_ratio:.3f} < "
                f"{max_transition_response_ratio:.3f})"
            )
        failures.extend(collapse_failures)

        image_count = sum(
            int((episode.get("saved_front_image_hashes") or {}).get("count", 0))
            for episode in episode_reports
        )
        image_unique = sum(
            int((episode.get("saved_front_image_hashes") or {}).get("unique", 0))
            for episode in episode_reports
        )
        row_reports.append(
            {
                "condition": row["condition"],
                "task": row["task"],
                "status": "passed" if not failures else "failed",
                "failures": failures,
                "artifact_count": len(pickle_paths),
                "fresh_query_count": len(all_records),
                "valid_gt_fresh_query_count": len(valid_gt_records),
                "saved_front_image_hashes": {
                    "count": image_count,
                    "episode_unique_sum": image_unique,
                    "note": "Uniqueness is summed per episode; IDs may repeat across rounds.",
                },
                "vlm_point": vlm_stats,
                "oracle_point": gt_stats,
                "vlm_to_gt_spread_ratio": spread_ratio,
                "transitions": {
                    "adjacent_vlm_displacement_px": _scalar_stats(
                        transition_values["adjacent_vlm"]
                    ),
                    "vlm_skill_change_vlm_displacement_px": _scalar_stats(
                        transition_values["vlm_skill_change_vlm"]
                    ),
                    "oracle_skill_change_vlm_displacement_px": _scalar_stats(
                        transition_values["oracle_skill_change_vlm"]
                    ),
                    "oracle_skill_change_gt_displacement_px": _scalar_stats(
                        transition_values["oracle_skill_change_gt"]
                    ),
                    "large_gt_displacement_threshold_px": large_gt_displacement_px,
                    "large_gt_transition_count": len(large_gt),
                    "large_gt_vlm_displacement_px": _scalar_stats(large_vlm),
                    "large_gt_oracle_displacement_px": _scalar_stats(large_gt),
                    "vlm_to_gt_response_ratio": transition_response_ratio,
                },
                "vlm_point_by_vlm_skill": _group_point_stats(
                    all_records, "vlm_skill"
                ),
                "vlm_point_by_oracle_skill": _group_point_stats(
                    all_records, "oracle_skill"
                ),
                "episodes": episode_reports,
            }
        )
        del all_records
        gc.collect()

    failed_rows = [
        f"{row['condition']}/{row['task']}" for row in row_reports if row["status"] != "passed"
    ]
    return {
        "version": 1,
        "status": "passed" if not failed_rows else "failed",
        "failed_rows": failed_rows,
        "thresholds": {
            "expected_query_interval": expected_query_interval,
            "min_fresh_queries": min_fresh_queries,
            "min_gt_spread_px": min_gt_spread_px,
            "max_vlm_to_gt_spread_ratio": max_spread_ratio,
            "large_gt_displacement_px": large_gt_displacement_px,
            "min_large_transitions": min_large_transitions,
            "max_vlm_to_gt_transition_response_ratio": (
                max_transition_response_ratio
            ),
        },
        "interpretation": (
            "Only cache_age_steps=0 observations are analyzed. A failed dynamics "
            "check indicates point-head collapse/regression-to-mean, not the intended "
            "eight-step action-horizon cache."
        ),
        "rows": row_reports,
    }
