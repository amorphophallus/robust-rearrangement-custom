"""Streaming statistics used to normalize RGB-D depth observations.

Depth values intentionally stay in the representation stored by the dataset.  In
particular, this module does not change sign or units.  Zero and non-finite values
are treated as invalid so that legacy all-zero depth placeholders do not affect
the normalizer.
"""

from __future__ import annotations

from typing import Dict, Mapping, MutableMapping

import numpy as np


DEPTH_NORMALIZER_STATS_ATTR = "depth_normalizer_stats"
DEPTH_CAMERA_KEYS = {
    "wrist": "depth_image1",
    "front": "depth_image2",
}
# Checkpoints created before dataset-backed depth normalization used these
# constants in ResnetEncoder.forward and therefore did not persist any buffers.
LEGACY_DEPTH_NORMALIZER_STATS = {
    "wrist": {"count": 1, "mean": 0.107, "std": 0.05, "M2": 0.0025},
    "front": {"count": 1, "mean": 1.03, "std": 0.493, "M2": 0.243049},
}
_DEPTH_STATS_CHUNK_SIZE = 1_000_000


def empty_depth_moments() -> Dict[str, Dict[str, float | int]]:
    return {
        camera_name: {"count": 0, "mean": 0.0, "M2": 0.0}
        for camera_name in DEPTH_CAMERA_KEYS
    }


def _merge_one(
    destination: MutableMapping[str, float | int],
    source: Mapping[str, float | int],
) -> None:
    source_count = int(source["count"])
    if source_count == 0:
        return

    destination_count = int(destination["count"])
    source_mean = float(source["mean"])
    source_m2 = float(source["M2"])

    if destination_count == 0:
        destination["count"] = source_count
        destination["mean"] = source_mean
        destination["M2"] = source_m2
        return

    destination_mean = float(destination["mean"])
    destination_m2 = float(destination["M2"])
    combined_count = destination_count + source_count
    delta = source_mean - destination_mean

    destination["count"] = combined_count
    destination["mean"] = (
        destination_mean + delta * source_count / combined_count
    )
    destination["M2"] = (
        destination_m2
        + source_m2
        + delta * delta * destination_count * source_count / combined_count
    )


def update_depth_moments(
    moments: MutableMapping[str, MutableMapping[str, float | int]],
    camera_name: str,
    values: np.ndarray,
) -> None:
    if camera_name not in DEPTH_CAMERA_KEYS:
        raise KeyError(
            f"Unknown depth camera {camera_name!r}; expected one of "
            f"{tuple(DEPTH_CAMERA_KEYS)}."
        )

    flat_values = np.asarray(values).reshape(-1)
    for start in range(0, flat_values.size, _DEPTH_STATS_CHUNK_SIZE):
        chunk = flat_values[start : start + _DEPTH_STATS_CHUNK_SIZE]
        valid_mask = np.isfinite(chunk) & (chunk != 0)
        if not np.any(valid_mask):
            continue

        valid_values = np.asarray(chunk[valid_mask], dtype=np.float64)
        batch_count = int(valid_values.size)
        batch_mean = float(np.mean(valid_values, dtype=np.float64))
        centered = valid_values - batch_mean
        batch_m2 = float(np.dot(centered, centered))

        _merge_one(
            moments[camera_name],
            {"count": batch_count, "mean": batch_mean, "M2": batch_m2},
        )


def merge_depth_moments(
    destination: MutableMapping[str, MutableMapping[str, float | int]],
    source: Mapping[str, Mapping[str, float | int]],
) -> None:
    for camera_name in DEPTH_CAMERA_KEYS:
        if camera_name not in source:
            continue
        _merge_one(destination[camera_name], source[camera_name])


def finalize_depth_moments(
    moments: Mapping[str, Mapping[str, float | int]],
) -> Dict[str, Dict[str, float | int]]:
    finalized: Dict[str, Dict[str, float | int]] = {}
    for camera_name in DEPTH_CAMERA_KEYS:
        camera_stats = moments[camera_name]
        count = int(camera_stats["count"])
        mean = float(camera_stats["mean"])
        m2 = max(float(camera_stats["M2"]), 0.0)
        variance = m2 / count if count > 0 else 0.0
        finalized[camera_name] = {
            "count": count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "M2": m2,
        }
    return finalized


def deserialize_depth_moments(
    raw_stats: Mapping[str, Mapping[str, float | int]] | None,
) -> Dict[str, Dict[str, float | int]]:
    if raw_stats is None:
        raise ValueError("Depth normalizer statistics are missing.")

    moments = empty_depth_moments()
    missing_cameras = set(DEPTH_CAMERA_KEYS) - set(raw_stats)
    if missing_cameras:
        raise ValueError(
            "Depth normalizer statistics are missing cameras: "
            f"{sorted(missing_cameras)}."
        )

    for camera_name in DEPTH_CAMERA_KEYS:
        camera_stats = raw_stats[camera_name]
        required_keys = {"count", "mean", "std", "M2"}
        missing_keys = required_keys - set(camera_stats)
        if missing_keys:
            raise ValueError(
                f"Depth statistics for {camera_name!r} are missing fields "
                f"{sorted(missing_keys)}."
            )
        count = int(camera_stats["count"])
        mean = float(camera_stats["mean"])
        std = float(camera_stats["std"])
        m2 = float(camera_stats["M2"])
        if count < 0 or not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(
                f"Invalid depth statistics for {camera_name!r}: "
                f"count={count}, mean={mean}, std={std}."
            )
        if std < 0 or not np.isfinite(m2) or m2 < 0:
            raise ValueError(
                f"Invalid depth moments for {camera_name!r}: std={std}, M2={m2}."
            )
        if count == 0 and (mean != 0.0 or std != 0.0 or m2 != 0.0):
            raise ValueError(
                f"Empty depth statistics for {camera_name!r} must be all zeros."
            )
        expected_std = float(np.sqrt(m2 / count)) if count > 0 else 0.0
        if not np.isclose(std, expected_std, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"Inconsistent depth statistics for {camera_name!r}: "
                f"std={std}, sqrt(M2/count)={expected_std}."
            )
        moments[camera_name] = {"count": count, "mean": mean, "M2": m2}
    return moments


def validate_usable_depth_stats(
    stats: Mapping[str, Mapping[str, float | int]],
) -> None:
    errors = []
    for camera_name in DEPTH_CAMERA_KEYS:
        if camera_name not in stats:
            errors.append(f"missing {camera_name}")
            continue
        camera_stats = stats[camera_name]
        count = int(camera_stats.get("count", 0))
        mean = float(camera_stats.get("mean", float("nan")))
        std = float(camera_stats.get("std", float("nan")))
        if count <= 0:
            errors.append(f"{camera_name} has no finite non-zero pixels")
        if not np.isfinite(mean):
            errors.append(f"{camera_name} mean is not finite")
        if not np.isfinite(std) or std <= 0:
            errors.append(f"{camera_name} std must be finite and positive, got {std}")

    if errors:
        raise ValueError("Unusable depth normalizer statistics: " + "; ".join(errors))
