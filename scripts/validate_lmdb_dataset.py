import argparse
from pathlib import Path

import numpy as np

from src.dataset.lmdb import (
    LMDB_FORMAT_VERSION,
    NORMALIZER_STATS_ATTR,
    episode_data_key,
    frame_key,
    json_loads_bytes,
    open_lmdb_env,
    read_lmdb_episode_index,
    read_lmdb_meta,
    unpack_frame,
    unpack_named_arrays,
)
from src.dataset.depth_stats import (
    DEPTH_CAMERA_KEYS,
    DEPTH_NORMALIZER_STATS_ATTR,
    empty_depth_moments,
    finalize_depth_moments,
    update_depth_moments,
)


NORMALIZER_STATS_KEYS = (
    "robot_state",
    "action/delta",
    "action/pos",
    "skill",
    "parts_poses",
)


def as_stats(raw_stats):
    return {
        key: {
            "min": np.asarray(value["min"], dtype=np.float32),
            "max": np.asarray(value["max"], dtype=np.float32),
        }
        for key, value in (raw_stats or {}).items()
    }


def update_stats(stats, key, array):
    if key not in NORMALIZER_STATS_KEYS or array.size == 0 or array.shape[0] == 0:
        return

    local_min = np.min(array, axis=0).astype(np.float32, copy=False)
    local_max = np.max(array, axis=0).astype(np.float32, copy=False)
    if key not in stats:
        stats[key] = {"min": local_min.copy(), "max": local_max.copy()}
        return

    stats[key]["min"] = np.minimum(stats[key]["min"], local_min)
    stats[key]["max"] = np.maximum(stats[key]["max"], local_max)


def check_array_specs(path, episode_idx, frame_count, arrays, lowdim_specs, stats):
    errors = []

    for key, spec in lowdim_specs.items():
        if key not in arrays:
            errors.append(f"episode {episode_idx}: missing lowdim key {key}")
            continue

        array = arrays[key]
        expected_dtype = np.dtype(spec["dtype"])
        expected_trailing_shape = tuple(spec["shape"][1:])

        if array.dtype != expected_dtype:
            errors.append(
                f"episode {episode_idx} key {key}: dtype {array.dtype} != {expected_dtype}"
            )
        if array.shape[1:] != expected_trailing_shape:
            errors.append(
                f"episode {episode_idx} key {key}: trailing shape {array.shape[1:]} "
                f"!= {expected_trailing_shape}"
            )

        if array.shape[0] not in (0, frame_count):
            errors.append(
                f"episode {episode_idx} key {key}: leading dim {array.shape[0]} "
                f"!= frame_count {frame_count}"
            )

        update_stats(stats, key, array)

    if errors:
        joined = "\n  ".join(errors)
        raise ValueError(f"{path} lowdim validation failed:\n  {joined}")


def check_frame_specs(path, txn, frame_indices, frame_specs):
    specs = frame_specs["specs"]
    for frame_idx in frame_indices:
        raw_frame = txn.get(frame_key(frame_idx))
        if raw_frame is None:
            raise KeyError(f"{path}: missing frame payload {frame_idx}")

        decoded = unpack_frame(raw_frame, frame_specs)
        for key, array in decoded.items():
            spec = specs[key]
            expected_dtype = np.dtype(spec["dtype"])
            expected_shape = tuple(spec["shape"])
            if array.dtype != expected_dtype:
                raise ValueError(
                    f"{path} frame {frame_idx} key {key}: dtype {array.dtype} "
                    f"!= {expected_dtype}"
                )
            if array.shape != expected_shape:
                raise ValueError(
                    f"{path} frame {frame_idx} key {key}: shape {array.shape} "
                    f"!= {expected_shape}"
                )


def compare_full_stats(path, computed_stats, stored_stats, atol):
    for key in NORMALIZER_STATS_KEYS:
        if key not in computed_stats and key not in stored_stats:
            continue
        if key not in computed_stats:
            raise ValueError(f"{path}: stored stats contain {key}, but data scan does not")
        if key not in stored_stats:
            raise ValueError(f"{path}: data contains {key}, but stored stats are missing it")

        np.testing.assert_allclose(
            computed_stats[key]["min"],
            stored_stats[key]["min"],
            atol=atol,
            rtol=0,
            err_msg=f"{path}: normalizer min mismatch for {key}",
        )
        np.testing.assert_allclose(
            computed_stats[key]["max"],
            stored_stats[key]["max"],
            atol=atol,
            rtol=0,
            err_msg=f"{path}: normalizer max mismatch for {key}",
        )


def update_episode_depth_stats(
    path,
    txn,
    frame_start,
    frame_end,
    frame_specs,
    depth_moments,
):
    depth_keys = list(DEPTH_CAMERA_KEYS.values())
    per_camera_values = {camera_name: [] for camera_name in DEPTH_CAMERA_KEYS}
    for frame_idx in range(frame_start, frame_end):
        raw_frame = txn.get(frame_key(frame_idx))
        if raw_frame is None:
            raise KeyError(f"{path}: missing frame payload {frame_idx}")
        decoded = unpack_frame(raw_frame, frame_specs, keys=depth_keys)
        for camera_name, depth_key in DEPTH_CAMERA_KEYS.items():
            per_camera_values[camera_name].append(decoded[depth_key])

    for camera_name, values in per_camera_values.items():
        if values:
            update_depth_moments(
                depth_moments,
                camera_name,
                np.stack(values, axis=0),
            )


def compare_depth_stats(path, computed_stats, stored_stats, atol):
    if stored_stats is None:
        raise ValueError(
            f"{path}: metadata is missing {DEPTH_NORMALIZER_STATS_ATTR!r}"
        )

    for camera_name in DEPTH_CAMERA_KEYS:
        if camera_name not in stored_stats:
            raise ValueError(f"{path}: stored depth stats are missing {camera_name}")
        expected = computed_stats[camera_name]
        actual = stored_stats[camera_name]
        if int(actual["count"]) != int(expected["count"]):
            raise ValueError(
                f"{path}: {camera_name} depth count mismatch: "
                f"stored={actual['count']} computed={expected['count']}"
            )
        for stat_name in ("mean", "std", "M2"):
            np.testing.assert_allclose(
                float(actual[stat_name]),
                float(expected[stat_name]),
                atol=atol,
                rtol=1e-10,
                err_msg=(
                    f"{path}: {camera_name} depth {stat_name} mismatch"
                ),
            )


def validate_path(path: Path, sample_episodes: int, full_stats: bool, atol: float):
    meta = read_lmdb_meta(path)
    episode_index = read_lmdb_episode_index(path)

    if meta.get("format") != "robust_rearrangement_lmdb":
        raise ValueError(f"{path}: unexpected format {meta.get('format')}")
    if int(meta.get("format_version", -1)) != LMDB_FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version {meta.get('format_version')} != {LMDB_FORMAT_VERSION}"
        )

    attrs = meta["attrs"]
    if int(attrs["n_episodes"]) != len(episode_index):
        raise ValueError(
            f"{path}: attrs n_episodes={attrs['n_episodes']} but index has "
            f"{len(episode_index)} entries"
        )

    lowdim_specs = meta["lowdim_specs"]
    frame_specs = meta["frame_specs"]
    stored_stats = as_stats(attrs.get(NORMALIZER_STATS_ATTR))
    stored_depth_stats = attrs.get(DEPTH_NORMALIZER_STATS_ATTR)

    if full_stats:
        selected_indices = range(len(episode_index))
    else:
        selected_indices = range(min(sample_episodes, len(episode_index)))
    selected_count = len(selected_indices)

    computed_stats = {}
    computed_depth_moments = empty_depth_moments()
    checked_frames = 0
    env = open_lmdb_env(path, readonly=True)
    try:
        with env.begin(write=False) as txn:
            for episode_idx in selected_indices:
                episode_meta = episode_index[episode_idx]
                frame_start = int(episode_meta["frame_start"])
                frame_end = int(episode_meta["frame_end"])
                frame_count = frame_end - frame_start

                raw_episode = txn.get(episode_data_key(episode_idx))
                if raw_episode is None:
                    raise KeyError(f"{path}: missing episode payload {episode_idx}")
                arrays = unpack_named_arrays(raw_episode)
                check_array_specs(
                    path,
                    episode_idx,
                    frame_count,
                    arrays,
                    lowdim_specs,
                    computed_stats,
                )

                frame_candidates = {
                    frame_start,
                    frame_start + frame_count // 2,
                    frame_end - 1,
                }
                frame_candidates = sorted(
                    idx for idx in frame_candidates if frame_start <= idx < frame_end
                )
                check_frame_specs(path, txn, frame_candidates, frame_specs)
                checked_frames += len(frame_candidates)
                if full_stats:
                    update_episode_depth_stats(
                        path,
                        txn,
                        frame_start,
                        frame_end,
                        frame_specs,
                        computed_depth_moments,
                    )

            raw_meta = txn.get(b"__meta__")
            if json_loads_bytes(raw_meta) != meta:
                raise ValueError(f"{path}: metadata roundtrip check failed")
    finally:
        env.close()

    if full_stats:
        compare_full_stats(path, computed_stats, stored_stats, atol)
        compare_depth_stats(
            path,
            finalize_depth_moments(computed_depth_moments),
            stored_depth_stats,
            atol,
        )

    print(
        f"[OK] {path}: episodes_checked={selected_count}, "
        f"frames_checked={checked_frames}, lowdim_keys={len(lowdim_specs)}, "
        f"frame_keys={len(frame_specs['ordered_keys'])}, full_stats={full_stats}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate LMDB dataset payload shapes, frame records, and stats."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sample-episodes", type=int, default=5)
    parser.add_argument(
        "--full-stats",
        action="store_true",
        help=(
            "Scan all episodes and compare low-dimensional min/max plus depth "
            "moments against stored normalizer statistics."
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    for path in args.paths:
        validate_path(path.expanduser().resolve(), args.sample_episodes, args.full_stats, args.atol)


if __name__ == "__main__":
    main()
