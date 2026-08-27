"""Step-level VLM point metrics against a shadow automaton.

The policy-facing VLM output is a 2-D front-camera pixel.  The annotation-noise
experiments perturb a 3-D guidance point, so their millimetre standard
deviations are not directly comparable to a pixel error.  This module builds a
camera- and depth-conditioned reference by projecting the same 3-D clipped
Gaussian noise through the front camera for every valid control-step pair.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SKILLS = ("push", "pick", "place", "insert", "screw")
NOISE_LEVEL_STD_M = {
    "n0": 0.0,
    "n1": 0.003,
    "n2": 0.006,
    "n3": 0.012,
    "n4": 0.024,
}
DEFAULT_MONTE_CARLO_SAMPLES_PER_PAIR = 200
REFERENCE_RESERVOIR_SIZE = 2_000


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _point_to_camera(camera_info: Mapping[str, Any]) -> np.ndarray:
    """Select canonical extrinsics, falling back for legacy saved rollouts."""

    value = camera_info.get("robot_base_to_camera")
    if value is None:
        value = camera_info.get("sim_local_to_camera")
    return _to_numpy(value)


def _project_continuous(
    point_3d: Any,
    camera_info: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return continuous ``uv`` and camera-frame xyz without image clipping."""

    point = _to_numpy(point_3d)
    intrinsics = _to_numpy(camera_info.get("intrinsics"))
    point_to_camera = _point_to_camera(camera_info)
    if point.shape != (3,) or intrinsics.shape != (3, 3):
        return None
    if point_to_camera.shape != (4, 4):
        return None
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point
    point_camera = point_to_camera @ homogeneous
    if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-8:
        return None
    point_cv = point_camera[:3].copy()
    point_cv[1] = -point_cv[1]
    point_image = intrinsics @ point_cv
    uv = point_image[:2] / point_image[2]
    if not np.isfinite(uv).all():
        return None
    return uv.astype(np.float64), point_camera[:3].astype(np.float64)


def _project_points_continuous(
    points_3d: Any,
    camera_info: Mapping[str, Any],
) -> np.ndarray | None:
    """Vectorized continuous projection matching annotation-util conventions."""

    points = _to_numpy(points_3d)
    intrinsics = _to_numpy(camera_info.get("intrinsics"))
    point_to_camera = _point_to_camera(camera_info)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    if intrinsics.shape != (3, 3) or point_to_camera.shape != (4, 4):
        return None
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    points_camera = (point_to_camera @ homogeneous.T).T
    if not np.isfinite(points_camera).all() or np.any(points_camera[:, 2] <= 1e-8):
        return None
    points_cv = points_camera[:, :3].copy()
    points_cv[:, 1] *= -1.0
    points_image = (intrinsics @ points_cv.T).T
    uv = points_image[:, :2] / points_image[:, 2:3]
    return uv.astype(np.float64) if np.isfinite(uv).all() else None


def _same_depth_error_mm(
    predicted_uv: np.ndarray,
    oracle_camera_xyz: np.ndarray,
    camera_info: Mapping[str, Any],
) -> float | None:
    """Unproject a pixel at oracle depth; this is lateral, not full 3-D error."""

    intrinsics = _to_numpy(camera_info.get("intrinsics"))
    if intrinsics.shape != (3, 3):
        return None
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    depth = float(oracle_camera_xyz[2])
    if fx <= 0.0 or fy <= 0.0 or depth <= 0.0:
        return None
    predicted_x = (float(predicted_uv[0]) - cx) * depth / fx
    predicted_y = -(float(predicted_uv[1]) - cy) * depth / fy
    lateral_m = math.hypot(
        predicted_x - float(oracle_camera_xyz[0]),
        predicted_y - float(oracle_camera_xyz[1]),
    )
    return float(1000.0 * lateral_m)


def _project_noise_residuals(
    oracle_point_3d: Any,
    camera_info: Mapping[str, Any],
    *,
    seed: int,
    n_samples: int = DEFAULT_MONTE_CARLO_SAMPLES_PER_PAIR,
) -> tuple[dict[str, list[list[float]]], float | None] | None:
    clean = _project_continuous(oracle_point_3d, camera_info)
    if clean is None:
        return None
    clean_uv, camera_xyz = clean
    if int(n_samples) <= 0:
        raise ValueError("projected-noise Monte Carlo sample count must be positive")
    rng = np.random.default_rng(int(seed))
    # This is the same component-wise N(0, sigma^2), clipped at +/-2 sigma,
    # used by annotation_noise.py. Common standard samples are reused across
    # n0--n4 to reduce Monte Carlo variance between levels.
    standard_noise = np.clip(
        rng.normal(0.0, 1.0, size=(int(n_samples), 3)), -2.0, 2.0
    )
    point = _to_numpy(oracle_point_3d)
    residuals: dict[str, list[list[float]]] = {}
    for level, std_m in NOISE_LEVEL_STD_M.items():
        projected_uv = _project_points_continuous(
            point[None, :] + standard_noise * std_m,
            camera_info,
        )
        if projected_uv is None:
            return None
        residuals[level] = (projected_uv - clean_uv[None, :]).astype(float).tolist()
    return residuals, float(camera_xyz[2])


def make_point_error_record(
    *,
    step_idx: int,
    oracle_skill: str | None,
    vlm_skill: str | None = None,
    oracle_point: Any,
    vlm_point: Any,
    query_step: int,
    oracle_point_3d: Any = None,
    camera_info: Mapping[str, Any] | None = None,
    noise_seed: int = 0,
    noise_projection_samples: int = DEFAULT_MONTE_CARLO_SAMPLES_PER_PAIR,
) -> dict[str, Any]:
    is_fresh_query = int(step_idx) == int(query_step)
    record = {
        "step_idx": int(step_idx),
        "oracle_skill": oracle_skill,
        "vlm_skill": vlm_skill,
        "query_step": int(query_step),
        "cache_age_steps": int(step_idx - query_step),
        "is_fresh_query": is_fresh_query,
        "oracle_point": None,
        "vlm_point": None,
        "residual_px": None,
        "error_px": None,
        "same_depth_error_mm": None,
        "projected_noise_residuals_px": None,
        "valid": False,
    }
    predicted = _to_numpy(vlm_point)
    if predicted.shape != (2,) or not np.isfinite(predicted).all():
        raise ValueError("invalid VLM point in metric record")
    record["vlm_point"] = predicted.tolist()
    if oracle_point is None:
        return record
    target = _to_numpy(oracle_point)
    if target.shape != (2,) or not np.isfinite(target).all():
        return record
    residual = predicted - target
    record["oracle_point"] = target.tolist()
    record["residual_px"] = residual.tolist()
    record["error_px"] = float(np.linalg.norm(residual))
    record["valid"] = True

    if oracle_point_3d is None or camera_info is None:
        return record
    clean = _project_continuous(oracle_point_3d, camera_info)
    if clean is not None:
        record["same_depth_error_mm"] = _same_depth_error_mm(
            predicted, clean[1], camera_info
        )
    # Build the camera/depth-conditioned noise reference for every valid
    # policy-facing GT/VLM pair.  This deliberately includes cached VLM steps:
    # the oracle point can move during the action horizon, and that staleness is
    # part of the guidance error actually seen by the policy.
    projected = _project_noise_residuals(
        oracle_point_3d,
        camera_info,
        seed=noise_seed,
        n_samples=noise_projection_samples,
    )
    if projected is not None:
        record["projected_noise_residuals_px"] = projected[0]
    return record


def _safe_centered_sum(sum_values: float, sum_squared: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return max(float(sum_squared) - float(sum_values) ** 2 / count, 0.0)


def _finish_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """Derive merge-safe point, spread, correlation, R2, and skill metrics."""

    count_valid = int(stats.get("count_valid", 0))
    count_total = int(stats.get("count_total", 0))
    sum_error = float(stats.get("sum_error_px", 0.0))
    sum_squared_error = float(stats.get("sum_squared_error_px", 0.0))
    sum_dx = float(stats.get("sum_dx_px", 0.0))
    sum_dy = float(stats.get("sum_dy_px", 0.0))
    same_depth_count = int(stats.get("same_depth_count", 0))
    same_depth_sum = float(stats.get("sum_same_depth_error_mm", 0.0))
    same_depth_squared = float(
        stats.get("sum_squared_same_depth_error_mm", 0.0)
    )
    errors = np.asarray(stats.get("error_samples_px", []), dtype=np.float64)
    errors = errors[np.isfinite(errors)]

    target_ss_x = _safe_centered_sum(
        stats.get("sum_target_x_px", 0.0),
        stats.get("sum_target_x_squared_px", 0.0),
        count_valid,
    )
    target_ss_y = _safe_centered_sum(
        stats.get("sum_target_y_px", 0.0),
        stats.get("sum_target_y_squared_px", 0.0),
        count_valid,
    )
    prediction_ss_x = _safe_centered_sum(
        stats.get("sum_prediction_x_px", 0.0),
        stats.get("sum_prediction_x_squared_px", 0.0),
        count_valid,
    )
    prediction_ss_y = _safe_centered_sum(
        stats.get("sum_prediction_y_px", 0.0),
        stats.get("sum_prediction_y_squared_px", 0.0),
        count_valid,
    )
    target_centered_ss = target_ss_x + target_ss_y
    prediction_centered_ss = prediction_ss_x + prediction_ss_y

    def correlation(axis: str, target_ss: float, prediction_ss: float):
        if count_valid <= 0 or target_ss <= 1e-12 or prediction_ss <= 1e-12:
            return None
        covariance_sum = float(
            stats.get(f"sum_target_prediction_{axis}_px2", 0.0)
        ) - (
            float(stats.get(f"sum_target_{axis}_px", 0.0))
            * float(stats.get(f"sum_prediction_{axis}_px", 0.0))
            / count_valid
        )
        return float(covariance_sum / math.sqrt(target_ss * prediction_ss))

    skill_count = int(stats.get("skill_count", 0))
    skill_correct_count = int(stats.get("skill_correct_count", 0))
    stats.update(
        {
            "mean_error_px": sum_error / count_valid if count_valid else None,
            "rmse_px": (
                math.sqrt(sum_squared_error / count_valid) if count_valid else None
            ),
            "p50_error_px": (
                float(np.quantile(errors, 0.50)) if len(errors) else None
            ),
            "p90_error_px": (
                float(np.quantile(errors, 0.90)) if len(errors) else None
            ),
            "p95_error_px": (
                float(np.quantile(errors, 0.95)) if len(errors) else None
            ),
            "count_invalid_gt": count_total - count_valid,
            "coverage": count_valid / count_total if count_total else None,
            "mean_dx_px": sum_dx / count_valid if count_valid else None,
            "mean_dy_px": sum_dy / count_valid if count_valid else None,
            "bias_norm_px": (
                math.hypot(sum_dx / count_valid, sum_dy / count_valid)
                if count_valid
                else None
            ),
            "target_spread_px": (
                math.sqrt(target_centered_ss / count_valid)
                if count_valid
                else None
            ),
            "prediction_spread_px": (
                math.sqrt(prediction_centered_ss / count_valid)
                if count_valid
                else None
            ),
            "spread_ratio": (
                math.sqrt(prediction_centered_ss / target_centered_ss)
                if target_centered_ss > 1e-12
                else None
            ),
            "corr_u": correlation("x", target_ss_x, prediction_ss_x),
            "corr_v": correlation("y", target_ss_y, prediction_ss_y),
            "point_r2": (
                1.0 - sum_squared_error / target_centered_ss
                if target_centered_ss > 1e-12
                else None
            ),
            "tail_count_gt_40px": int(stats.get("tail_count_gt_40px", 0)),
            "tail_count_gt_70px": int(stats.get("tail_count_gt_70px", 0)),
            "skill_accuracy": (
                skill_correct_count / skill_count if skill_count else None
            ),
            "mean_same_depth_error_mm": (
                same_depth_sum / same_depth_count if same_depth_count else None
            ),
            "rmse_same_depth_error_mm": (
                math.sqrt(same_depth_squared / same_depth_count)
                if same_depth_count
                else None
            ),
        }
    )
    return stats


def _summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    valid_rows = [row for row in rows if row.get("valid")]
    errors = [float(row["error_px"]) for row in valid_rows]
    residuals = [row.get("residual_px") for row in valid_rows]
    residuals = [
        np.asarray(value, dtype=np.float64)
        for value in residuals
        if value is not None and np.asarray(value).shape == (2,)
    ]
    same_depth = [
        float(row["same_depth_error_mm"])
        for row in valid_rows
        if row.get("same_depth_error_mm") is not None
        and np.isfinite(row["same_depth_error_mm"])
    ]
    count_total = len(rows)
    count_valid = len(errors)
    sum_error = float(sum(errors))
    sum_squared = float(sum(error * error for error in errors))
    residual_array = (
        np.stack(residuals, axis=0)
        if residuals
        else np.empty((0, 2), dtype=np.float64)
    )
    sum_dx = float(residual_array[:, 0].sum())
    sum_dy = float(residual_array[:, 1].sum())
    sum_dx_squared = float(np.square(residual_array[:, 0]).sum())
    sum_dy_squared = float(np.square(residual_array[:, 1]).sum())
    sum_dxdy = float((residual_array[:, 0] * residual_array[:, 1]).sum())
    same_depth_sum = float(sum(same_depth))
    same_depth_sum_squared = float(sum(value * value for value in same_depth))
    valid_targets = np.asarray(
        [row["oracle_point"] for row in valid_rows], dtype=np.float64
    ).reshape(-1, 2)
    valid_predictions = np.asarray(
        [row["vlm_point"] for row in valid_rows], dtype=np.float64
    ).reshape(-1, 2)
    skill_pairs = [
        (str(row["oracle_skill"]), str(row["vlm_skill"]))
        for row in rows
        if row.get("oracle_skill") in SKILLS and row.get("vlm_skill") in SKILLS
    ]
    skill_confusion = {
        oracle: {
            predicted: sum(
                pair == (oracle, predicted) for pair in skill_pairs
            )
            for predicted in SKILLS
        }
        for oracle in SKILLS
    }
    stats = {
        "count_valid": count_valid,
        "count_total": count_total,
        "same_depth_count": len(same_depth),
        "error_samples_px": errors,
        "tail_count_gt_40px": sum(error > 40.0 for error in errors),
        "tail_count_gt_70px": sum(error > 70.0 for error in errors),
        "skill_count": len(skill_pairs),
        "skill_correct_count": sum(oracle == predicted for oracle, predicted in skill_pairs),
        "skill_confusion": skill_confusion,
        # Sufficient statistics make multi-batch merges exact.
        "sum_error_px": sum_error,
        "sum_squared_error_px": sum_squared,
        "sum_dx_px": sum_dx,
        "sum_dy_px": sum_dy,
        "sum_dx_squared_px": sum_dx_squared,
        "sum_dy_squared_px": sum_dy_squared,
        "sum_dxdy_px": sum_dxdy,
        "sum_target_x_px": float(valid_targets[:, 0].sum()),
        "sum_target_y_px": float(valid_targets[:, 1].sum()),
        "sum_target_x_squared_px": float(np.square(valid_targets[:, 0]).sum()),
        "sum_target_y_squared_px": float(np.square(valid_targets[:, 1]).sum()),
        "sum_prediction_x_px": float(valid_predictions[:, 0].sum()),
        "sum_prediction_y_px": float(valid_predictions[:, 1].sum()),
        "sum_prediction_x_squared_px": float(
            np.square(valid_predictions[:, 0]).sum()
        ),
        "sum_prediction_y_squared_px": float(
            np.square(valid_predictions[:, 1]).sum()
        ),
        "sum_target_prediction_x_px2": float(
            (valid_targets[:, 0] * valid_predictions[:, 0]).sum()
        ),
        "sum_target_prediction_y_px2": float(
            (valid_targets[:, 1] * valid_predictions[:, 1]).sum()
        ),
        "sum_same_depth_error_mm": same_depth_sum,
        "sum_squared_same_depth_error_mm": same_depth_sum_squared,
    }
    return _finish_summary(stats)


def _empty_moments() -> dict[str, float | int]:
    return {
        "count": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
        "sum_radial": 0.0,
        "sum_radial2": 0.0,
    }


def _update_moments(target: dict[str, Any], values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    if not len(values):
        return
    radial = np.linalg.norm(values, axis=1)
    target["count"] += int(len(values))
    target["sum_x"] += float(values[:, 0].sum())
    target["sum_y"] += float(values[:, 1].sum())
    target["sum_x2"] += float(np.square(values[:, 0]).sum())
    target["sum_y2"] += float(np.square(values[:, 1]).sum())
    target["sum_xy"] += float((values[:, 0] * values[:, 1]).sum())
    target["sum_radial"] += float(radial.sum())
    target["sum_radial2"] += float(np.square(radial).sum())


def _merge_moments(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    merged = _empty_moments()
    for item in items:
        for key in merged:
            merged[key] += item.get(key, 0)
    merged["count"] = int(merged["count"])
    return merged


def _reference_reservoir(values: Sequence[Any]) -> list[list[float]]:
    """Keep a bounded, evenly spread sample for distribution-distance metrics."""

    if len(values) <= REFERENCE_RESERVOIR_SIZE:
        return list(values)
    indices = np.linspace(
        0,
        len(values) - 1,
        num=REFERENCE_RESERVOIR_SIZE,
        dtype=np.int64,
    )
    return [values[int(index)] for index in indices]


def _distribution_samples(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    vlm: list[list[float]] = []
    noise = {level: [] for level in NOISE_LEVEL_STD_M}
    noise_moments = {level: _empty_moments() for level in NOISE_LEVEL_STD_M}
    same_depth: list[float] = []
    reference_pair_count = 0
    samples_per_pair: set[int] = set()
    for row in records:
        if not row.get("valid"):
            continue
        residual = np.asarray(row.get("residual_px"), dtype=np.float64)
        projected = row.get("projected_noise_residuals_px")
        if residual.shape != (2,) or not np.isfinite(residual).all():
            continue
        if not isinstance(projected, Mapping):
            continue
        noise_values = {
            level: np.asarray(projected.get(level), dtype=np.float64)
            for level in NOISE_LEVEL_STD_M
        }
        if any(
            value.ndim != 2
            or value.shape[1] != 2
            or not len(value)
            or not np.isfinite(value).all()
            for value in noise_values.values()
        ):
            continue
        level_counts = {len(value) for value in noise_values.values()}
        if len(level_counts) != 1:
            continue
        vlm.append(residual.tolist())
        for level, value in noise_values.items():
            noise[level].extend(value.tolist())
            _update_moments(noise_moments[level], value)
        reference_pair_count += 1
        samples_per_pair.add(next(iter(level_counts)))
        depth_error = row.get("same_depth_error_mm")
        if depth_error is not None and np.isfinite(depth_error):
            same_depth.append(float(depth_error))
    return {
        "vlm_residuals_px": vlm,
        "projected_noise_residuals_px": {
            level: _reference_reservoir(values) for level, values in noise.items()
        },
        "projected_noise_moments": noise_moments,
        "same_depth_errors_mm": same_depth,
        "reference_pair_count": reference_pair_count,
        "monte_carlo_samples_per_pair": (
            next(iter(samples_per_pair)) if len(samples_per_pair) == 1 else None
        ),
        "reference_reservoir_size_per_level": REFERENCE_RESERVOIR_SIZE,
    }


def _wasserstein_1d(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if not len(first) or not len(second):
        return None
    if len(first) == len(second):
        return float(np.mean(np.abs(np.sort(first) - np.sort(second))))
    # A bounded common quantile grid avoids O(reference_size^2)-like cost when
    # the VLM distribution has one sample/query and the Monte Carlo reference
    # has many samples per control-step pair.
    n_quantiles = min(max(len(first), len(second)), 2_048)
    quantiles = (np.arange(n_quantiles, dtype=np.float64) + 0.5) / n_quantiles
    return float(
        np.mean(
            np.abs(
                np.quantile(first, quantiles, method="linear")
                - np.quantile(second, quantiles, method="linear")
            )
        )
    )


def _sliced_wasserstein_2d(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_directions: int = 32,
) -> float | None:
    if len(first) == 0 or len(second) == 0:
        return None
    angles = np.arange(n_directions, dtype=np.float64) * math.pi / n_directions
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    distances = [
        _wasserstein_1d(first @ direction, second @ direction)
        for direction in directions
    ]
    valid = [value for value in distances if value is not None]
    return float(np.mean(valid)) if valid else None


def _ks_statistic(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.sort(np.asarray(first, dtype=np.float64).reshape(-1))
    second = np.sort(np.asarray(second, dtype=np.float64).reshape(-1))
    if not len(first) or not len(second):
        return None
    support = np.unique(np.concatenate([first, second]))
    first_cdf = np.searchsorted(first, support, side="right") / len(first)
    second_cdf = np.searchsorted(second, support, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def _residual_stats(
    values: np.ndarray,
    *,
    moments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    count = int((moments or {}).get("count", len(values)))
    if count == 0:
        return {
            "count": 0,
            "mean_error_px": None,
            "rmse_px": None,
            "p50_error_px": None,
            "p90_error_px": None,
            "p95_error_px": None,
            "mean_dx_px": None,
            "mean_dy_px": None,
            "bias_norm_px": None,
            "covariance_px2": None,
            "anisotropy_ratio": None,
            "principal_direction_deg": None,
        }
    radial = np.linalg.norm(values, axis=1) if len(values) else np.empty(0)
    if moments is None:
        mean = values.mean(axis=0)
        covariance = (
            np.cov(values, rowvar=False, ddof=0)
            if len(values) > 1
            else np.zeros((2, 2), dtype=np.float64)
        )
        mean_radial = float(radial.mean())
        rmse_radial = float(math.sqrt(np.square(radial).mean()))
    else:
        mean = np.array(
            [float(moments.get("sum_x", 0.0)) / count,
             float(moments.get("sum_y", 0.0)) / count],
            dtype=np.float64,
        )
        covariance = np.array(
            [
                [
                    float(moments.get("sum_x2", 0.0)) / count - mean[0] ** 2,
                    float(moments.get("sum_xy", 0.0)) / count - mean[0] * mean[1],
                ],
                [
                    float(moments.get("sum_xy", 0.0)) / count - mean[0] * mean[1],
                    float(moments.get("sum_y2", 0.0)) / count - mean[1] ** 2,
                ],
            ],
            dtype=np.float64,
        )
        covariance[np.abs(covariance) < 1e-15] = 0.0
        mean_radial = float(moments.get("sum_radial", 0.0)) / count
        rmse_radial = math.sqrt(
            max(float(moments.get("sum_radial2", 0.0)) / count, 0.0)
        )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(max(eigenvalues[-1], 0.0))
    smallest = float(max(eigenvalues[0], 0.0))
    anisotropy = (
        largest / smallest
        if smallest > 1e-12
        else (float("inf") if largest > 1e-12 else None)
    )
    principal = eigenvectors[:, -1]
    principal_deg = (
        float(math.degrees(math.atan2(principal[1], principal[0])))
        if largest > 1e-12
        else None
    )
    return {
        "count": count,
        "mean_error_px": mean_radial,
        "rmse_px": rmse_radial,
        "p50_error_px": float(np.quantile(radial, 0.50)) if len(radial) else None,
        "p90_error_px": float(np.quantile(radial, 0.90)) if len(radial) else None,
        "p95_error_px": float(np.quantile(radial, 0.95)) if len(radial) else None,
        "mean_dx_px": float(mean[0]),
        "mean_dy_px": float(mean[1]),
        "bias_norm_px": float(np.linalg.norm(mean)),
        "covariance_px2": covariance.astype(float).tolist(),
        "anisotropy_ratio": anisotropy,
        "principal_direction_deg": principal_deg,
    }


def _build_distribution_comparison(
    samples: Mapping[str, Any],
    *,
    basis: str = "valid_control_step_gt_vlm_pairs",
) -> dict[str, Any]:
    vlm = np.asarray(samples.get("vlm_residuals_px", []), dtype=np.float64).reshape(-1, 2)
    same_depth = np.asarray(samples.get("same_depth_errors_mm", []), dtype=np.float64)
    vlm_radial = np.linalg.norm(vlm, axis=1) if len(vlm) else np.empty(0)
    by_level: dict[str, Any] = {}
    for level in NOISE_LEVEL_STD_M:
        values = np.asarray(
            samples.get("projected_noise_residuals_px", {}).get(level, []),
            dtype=np.float64,
        ).reshape(-1, 2)
        moments = samples.get("projected_noise_moments", {}).get(level)
        projected_stats = _residual_stats(values, moments=moments)
        radial = np.linalg.norm(values, axis=1) if len(values) else np.empty(0)
        centered_vlm = vlm - vlm.mean(axis=0) if len(vlm) else vlm
        projected_mean = (
            np.asarray(
                [projected_stats["mean_dx_px"], projected_stats["mean_dy_px"]],
                dtype=np.float64,
            )
            if projected_stats["mean_dx_px"] is not None
            else None
        )
        centered_values = (
            values - projected_mean[None, :]
            if len(values) and projected_mean is not None
            else values
        )
        vlm_mean = vlm.mean(axis=0) if len(vlm) else None
        by_level[level] = {
            "noise_std_mm": 1000.0 * NOISE_LEVEL_STD_M[level],
            "projected": projected_stats,
            "sliced_wasserstein_px": _sliced_wasserstein_2d(vlm, values),
            "centered_sliced_wasserstein_px": _sliced_wasserstein_2d(
                centered_vlm, centered_values
            ),
            "radial_wasserstein_px": _wasserstein_1d(vlm_radial, radial),
            "radial_ks_statistic": _ks_statistic(vlm_radial, radial),
            "bias_difference_px": (
                float(np.linalg.norm(vlm_mean - projected_mean))
                if vlm_mean is not None and projected_mean is not None
                else None
            ),
        }

    def closest(metric: str) -> str | None:
        candidates = [
            (level, values.get(metric)) for level, values in by_level.items()
        ]
        candidates = [
            (level, float(value))
            for level, value in candidates
            if value is not None and np.isfinite(value)
        ]
        return min(candidates, key=lambda item: item[1])[0] if candidates else None

    vlm_stats = _residual_stats(vlm)
    magnitude_candidates = []
    if vlm_stats["rmse_px"] is not None:
        for level, values in by_level.items():
            projected_rmse = values["projected"]["rmse_px"]
            if projected_rmse is not None:
                magnitude_candidates.append(
                    (level, abs(vlm_stats["rmse_px"] - projected_rmse))
                )

    def equivalent_std_mm() -> tuple[float | None, str | None]:
        target = vlm_stats.get("rmse_px")
        points = []
        for level, std_m in NOISE_LEVEL_STD_M.items():
            projected_rmse = by_level[level]["projected"].get("rmse_px")
            if projected_rmse is not None:
                points.append((1000.0 * std_m, float(projected_rmse), level))
        if target is None or len(points) < 2:
            return None, None
        points.sort(key=lambda value: value[1])
        if float(target) > points[-1][1]:
            low, high = points[-2], points[-1]
            bracket = f">{high[2]}"
        else:
            low, high = points[0], points[1]
            bracket = high[2]
            for left, right in zip(points, points[1:]):
                if left[1] <= float(target) <= right[1]:
                    low, high = left, right
                    bracket = (
                        left[2]
                        if math.isclose(float(target), left[1], abs_tol=1e-12)
                        else f"{left[2]}–{right[2]}"
                    )
                    break
        denominator = high[1] - low[1]
        if abs(denominator) <= 1e-12:
            return low[0], bracket
        fraction = (float(target) - low[1]) / denominator
        return float(low[0] + fraction * (high[0] - low[0])), bracket

    equivalent_mm, magnitude_bracket = equivalent_std_mm()
    return {
        "basis": basis,
        "projection_reference": {
            "reference_pair_count": int(samples.get("reference_pair_count", 0)),
            "monte_carlo_samples_per_pair": samples.get(
                "monte_carlo_samples_per_pair"
            ),
            "reference_reservoir_size_per_level": int(
                samples.get("reference_reservoir_size_per_level", 0)
            ),
            "sampling_distribution": "component-wise N(0, 1), clipped to [-2, 2]",
            "common_random_numbers_across_levels": True,
        },
        "vlm": vlm_stats,
        "same_depth": {
            "count": int(len(same_depth)),
            "mean_error_mm": float(same_depth.mean()) if len(same_depth) else None,
            "rmse_mm": (
                float(math.sqrt(np.square(same_depth).mean()))
                if len(same_depth)
                else None
            ),
            "p50_error_mm": float(np.quantile(same_depth, 0.50)) if len(same_depth) else None,
            "p90_error_mm": float(np.quantile(same_depth, 0.90)) if len(same_depth) else None,
            "p95_error_mm": float(np.quantile(same_depth, 0.95)) if len(same_depth) else None,
            "interpretation": "same-depth lateral error; not full 3-D error",
        },
        "noise_levels": by_level,
        "closest_level_sliced_wasserstein": closest("sliced_wasserstein_px"),
        "closest_level_centered_sliced_wasserstein": closest(
            "centered_sliced_wasserstein_px"
        ),
        "closest_level_radial_wasserstein": closest("radial_wasserstein_px"),
        "closest_level_rmse": (
            min(magnitude_candidates, key=lambda item: item[1])[0]
            if magnitude_candidates
            else None
        ),
        "magnitude_equivalent_std_mm": equivalent_mm,
        "magnitude_equivalent_bracket": magnitude_bracket,
    }


def summarize_point_error_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    fresh_records = [row for row in records if row.get("is_fresh_query")]
    samples = _distribution_samples(records)
    fresh_samples = _distribution_samples(fresh_records)
    return {
        "overall": _summarize(records),
        "by_skill": {
            skill: _summarize(
                row for row in records if row.get("oracle_skill") == skill
            )
            for skill in SKILLS
        },
        "step_distribution": _build_distribution_comparison(samples),
        "distribution_samples": samples,
        "fresh_queries": {
            "overall": _summarize(fresh_records),
            "by_skill": {
                skill: _summarize(
                    row
                    for row in fresh_records
                    if row.get("oracle_skill") == skill
                )
                for skill in SKILLS
            },
            "distribution": _build_distribution_comparison(
                fresh_samples,
                basis="valid_fresh_query_gt_vlm_pairs",
            ),
            "distribution_samples": fresh_samples,
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


def _merge_samples(sample_sets: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sample_sets = list(sample_sets)
    merged: dict[str, Any] = {
        "vlm_residuals_px": [],
        "projected_noise_residuals_px": {
            level: [] for level in NOISE_LEVEL_STD_M
        },
        "projected_noise_moments": {
            level: _empty_moments() for level in NOISE_LEVEL_STD_M
        },
        "same_depth_errors_mm": [],
        "reference_pair_count": 0,
        "monte_carlo_samples_per_pair": None,
        "reference_reservoir_size_per_level": REFERENCE_RESERVOIR_SIZE,
    }
    sample_counts = {
        int(samples["monte_carlo_samples_per_pair"])
        for samples in sample_sets
        if samples.get("monte_carlo_samples_per_pair") is not None
    }
    merged["monte_carlo_samples_per_pair"] = (
        next(iter(sample_counts)) if len(sample_counts) == 1 else None
    )
    for samples in sample_sets:
        merged["vlm_residuals_px"].extend(samples.get("vlm_residuals_px", []))
        merged["same_depth_errors_mm"].extend(
            samples.get("same_depth_errors_mm", [])
        )
        projected = samples.get("projected_noise_residuals_px", {})
        projected_moments = samples.get("projected_noise_moments", {})
        merged["reference_pair_count"] += int(
            samples.get("reference_pair_count", 0)
        )
        for level in NOISE_LEVEL_STD_M:
            merged["projected_noise_residuals_px"][level].extend(
                projected.get(level, [])
            )
            merged["projected_noise_moments"][level] = _merge_moments(
                [
                    merged["projected_noise_moments"][level],
                    projected_moments.get(level, {}),
                ]
            )
    for level in NOISE_LEVEL_STD_M:
        merged["projected_noise_residuals_px"][level] = _reference_reservoir(
            merged["projected_noise_residuals_px"][level]
        )
    return merged


def merge_vlm_point_error_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def merge_stats(stats_list):
        count_valid = sum(int(stats.get("count_valid", 0)) for stats in stats_list)
        count_total = sum(int(stats.get("count_total", 0)) for stats in stats_list)
        same_depth_count = sum(int(stats.get("same_depth_count", 0)) for stats in stats_list)

        def total(key: str) -> float:
            return sum(float(stats.get(key, 0.0)) for stats in stats_list)

        additive_fields = (
            "sum_error_px",
            "sum_squared_error_px",
            "sum_dx_px",
            "sum_dy_px",
            "sum_dx_squared_px",
            "sum_dy_squared_px",
            "sum_dxdy_px",
            "sum_target_x_px",
            "sum_target_y_px",
            "sum_target_x_squared_px",
            "sum_target_y_squared_px",
            "sum_prediction_x_px",
            "sum_prediction_y_px",
            "sum_prediction_x_squared_px",
            "sum_prediction_y_squared_px",
            "sum_target_prediction_x_px2",
            "sum_target_prediction_y_px2",
            "sum_same_depth_error_mm",
            "sum_squared_same_depth_error_mm",
        )
        confusion = {
            oracle: {
                predicted: sum(
                    int(
                        stats.get("skill_confusion", {})
                        .get(oracle, {})
                        .get(predicted, 0)
                    )
                    for stats in stats_list
                )
                for predicted in SKILLS
            }
            for oracle in SKILLS
        }
        merged = {
            "count_valid": count_valid,
            "count_total": count_total,
            "same_depth_count": same_depth_count,
            "error_samples_px": [
                float(value)
                for stats in stats_list
                for value in stats.get("error_samples_px", [])
            ],
            "tail_count_gt_40px": sum(
                int(stats.get("tail_count_gt_40px", 0)) for stats in stats_list
            ),
            "tail_count_gt_70px": sum(
                int(stats.get("tail_count_gt_70px", 0)) for stats in stats_list
            ),
            "skill_count": sum(int(stats.get("skill_count", 0)) for stats in stats_list),
            "skill_correct_count": sum(
                int(stats.get("skill_correct_count", 0)) for stats in stats_list
            ),
            "skill_confusion": confusion,
        }
        merged.update({field: total(field) for field in additive_fields})
        return _finish_summary(merged)

    output = {}
    for scope in ("all", "success_only", "failure_only"):
        scoped = [summary.get(scope, {}) for summary in summaries]
        samples = _merge_samples(
            item.get("distribution_samples", {}) for item in scoped
        )
        output[scope] = {
            "overall": merge_stats([item.get("overall", {}) for item in scoped]),
            "by_skill": {
                skill: merge_stats(
                    [item.get("by_skill", {}).get(skill, {}) for item in scoped]
                )
                for skill in SKILLS
            },
            "step_distribution": _build_distribution_comparison(samples),
            "distribution_samples": samples,
        }
        fresh_scoped = [item.get("fresh_queries", {}) for item in scoped]
        fresh_samples = _merge_samples(
            item.get("distribution_samples", {}) for item in fresh_scoped
        )
        output[scope]["fresh_queries"] = {
            "overall": merge_stats(
                [item.get("overall", {}) for item in fresh_scoped]
            ),
            "by_skill": {
                skill: merge_stats(
                    [item.get("by_skill", {}).get(skill, {}) for item in fresh_scoped]
                )
                for skill in SKILLS
            },
            "distribution": _build_distribution_comparison(
                fresh_samples,
                basis="valid_fresh_query_gt_vlm_pairs",
            ),
            "distribution_samples": fresh_samples,
        }
    return output
