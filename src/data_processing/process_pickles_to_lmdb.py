import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from src.common.files import (
    expand_lmdb_shard_paths,
    get_processed_path,
    get_raw_paths,
    lmdb_shard_path,
)
from src.data_processing.process_pickles import (
    NORMALIZER_STATS_KEYS,
    TIMESERIES_KEYS,
    compute_normalizer_stats_from_dict,
    merge_normalizer_stats,
    process_pickle_file,
    serialize_normalizer_stats,
)
from src.data_processing.offline_image_annotations import IMAGE_ANNOTATION_MODES
from src.dataset.lmdb import (
    EPISODE_INDEX_KEY,
    DEFAULT_FRAME_COMPRESSION,
    DEFAULT_FRAME_COMPRESSION_LEVEL,
    LMDB_FORMAT_VERSION,
    META_KEY,
    build_frame_specs,
    episode_data_key,
    frame_key,
    json_dumps_bytes,
    open_lmdb_env,
    pack_frame,
    pack_named_arrays,
    require_lmdb,
    require_zstandard,
    read_lmdb_episode_index,
    read_lmdb_meta,
)
from src.dataset.depth_stats import (
    DEPTH_CAMERA_KEYS,
    DEPTH_NORMALIZER_STATS_ATTR,
    empty_depth_moments,
    finalize_depth_moments,
    update_depth_moments,
)
from src.common.gripper import (
    GRIPPER_OPEN_THRESHOLD_METERS,
    GRIPPER_WIDTH_ENCODING,
)
from src.common.pickle_compat import load_pickle_path
from src.real.legacy_timeline import reconstruct_legacy_real_trajectory
from src.real.v6_pickle_contract import V6_BUFFERED_SCHEMA


LOWDIM_KEYS = tuple(key for key in TIMESERIES_KEYS if key not in {
    "color_image1",
    "color_image2",
    "depth_image1",
    "depth_image2",
})
IMAGE_KEYS = ("color_image1", "color_image2", "depth_image1", "depth_image2")


def normalize_env_label(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    label = str(value)
    return label if label.strip() else None


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def log_lmdb_storage_layout(
    frame_specs,
    lowdim_specs,
    resize_image: bool,
    image_size: Optional[int],
):
    compression = frame_specs.get("compression")
    codec = compression.get("codec") if isinstance(compression, dict) else compression
    print(
        "[INFO] LMDB storage format: images are stored as per-frame byte payloads "
        f"(compression={codec or 'none'}), low-dimensional arrays are stored once "
        "per episode."
    )
    print(f"[INFO] resize_image={resize_image} (default: False)")
    print(f"[INFO] image_size={image_size} (cross-simulator contract: 224)")
    print("[INFO] Image frame layout:")
    for key in frame_specs["ordered_keys"]:
        spec = frame_specs["specs"][key]
        print(
            f"[INFO]   {key}: dtype={spec['dtype']}, shape={tuple(spec['shape'])}, "
            f"nbytes/frame={spec['nbytes']} ({format_bytes(int(spec['nbytes']))})"
        )
    print(
        "[INFO]   total image bytes/timestep="
        f"{frame_specs['total_nbytes']} ({format_bytes(int(frame_specs['total_nbytes']))})"
    )
    print("[INFO] Low-dimensional episode arrays:")
    for key in LOWDIM_KEYS:
        spec = lowdim_specs[key]
        print(
            f"[INFO]   {key}: dtype={spec['dtype']}, shape={tuple(spec['shape'])}"
        )


def log_episode_storage_debug(
    episode_data,
    frame_specs,
    packed_lowdim_nbytes: int,
):
    episode_length = int(episode_data["episode_length"])
    image_nbytes = int(frame_specs["total_nbytes"]) * episode_length
    total_nbytes = image_nbytes + packed_lowdim_nbytes
    print(
        f"[DEBUG] Example episode storage: timesteps={episode_length}, "
        f"image_bytes={image_nbytes} ({format_bytes(image_nbytes)}), "
        f"lowdim_bytes={packed_lowdim_nbytes} ({format_bytes(packed_lowdim_nbytes)}), "
        f"total={total_nbytes} ({format_bytes(total_nbytes)})"
    )
    for key in IMAGE_KEYS:
        array = np.asarray(episode_data[key])
        print(
            f"[DEBUG]   {key}: shape={tuple(array.shape)}, dtype={array.dtype}, "
            f"bytes/episode={array.nbytes} ({format_bytes(int(array.nbytes))})"
        )


def log_batch_storage_debug(
    batch_index: int,
    total_batches: int,
    batch_timesteps: int,
    batch_episodes: int,
    batch_image_bytes: int,
    batch_lowdim_bytes: int,
    running_total_bytes: int,
):
    batch_total_bytes = batch_image_bytes + batch_lowdim_bytes
    avg_bytes_per_timestep = (
        batch_total_bytes / batch_timesteps if batch_timesteps > 0 else 0.0
    )
    print(
        f"[DEBUG] Batch {batch_index}/{total_batches} payload estimate: "
        f"episodes={batch_episodes}, timesteps={batch_timesteps}, "
        f"image={format_bytes(batch_image_bytes)}, "
        f"lowdim={format_bytes(batch_lowdim_bytes)}, "
        f"total={format_bytes(batch_total_bytes)}, "
        f"avg/timestep={format_bytes(int(round(avg_bytes_per_timestep)))}, "
        f"running_total={format_bytes(running_total_bytes)}"
    )


def parse_task_episode_limits(entries: List[str]) -> Dict[str, int]:
    limits = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --task-episode-limit entry {entry!r}. Expected TASK=COUNT."
            )
        task, count = entry.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Invalid task name in entry {entry!r}.")
        limits[task] = int(count)
    return limits


def parse_datetime_from_pickle_name(path: Path) -> Optional[datetime]:
    name = path.name
    patterns = (
        (
            r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?",
            ("%Y-%m-%dT%H-%M-%S.%f", "%Y-%m-%dT%H-%M-%S"),
        ),
        (
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:\.\d+)?",
            ("%Y-%m-%d_%H-%M-%S.%f", "%Y-%m-%d_%H-%M-%S"),
        ),
        (
            r"\d{8}_\d{6}(?:\.\d+)?",
            ("%Y%m%d_%H%M%S.%f", "%Y%m%d_%H%M%S"),
        ),
    )
    for pattern, formats in patterns:
        for match in re.findall(pattern, name):
            for time_format in formats:
                try:
                    return datetime.strptime(match, time_format)
                except ValueError:
                    continue
    return None


def pickle_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def newest_pickle_sort_key(path: Path):
    parsed_time = parse_datetime_from_pickle_name(path)
    timestamp = (
        parsed_time.timestamp() if parsed_time is not None else pickle_mtime(path)
    )
    return (-timestamp, str(path))


def order_pickle_paths(
    paths: List[Path],
    randomize_order: bool,
    rng: random.Random,
) -> List[Path]:
    paths = list(paths)
    if randomize_order:
        rng.shuffle(paths)
        return paths
    return sorted(paths, key=newest_pickle_sort_key)


def pickle_identity(path: Path) -> str:
    path = Path(path).expanduser()
    try:
        resolved_path = path.resolve()
    except OSError:
        resolved_path = path.absolute()

    for candidate in (resolved_path, path):
        parts = candidate.parts
        if "raw" in parts:
            return "/".join(parts[parts.index("raw") + 1 :])
    return str(resolved_path)


def absolute_pickle_path(path: Path) -> str:
    path = Path(path).expanduser()
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def unnumbered_lmdb_base_path(path: Path) -> Path:
    match = re.match(r"^(?P<stem>.*)-\d+$", path.stem)
    if match:
        return path.with_name(f"{match.group('stem')}{path.suffix}")
    return path


def next_lmdb_shard_path(base_path: Path) -> Path:
    shard_index = 1
    while True:
        candidate = lmdb_shard_path(base_path, shard_index)
        if not candidate.exists():
            return candidate
        shard_index += 1


def infer_shard_index(base_path: Path, output_path: Path) -> Optional[int]:
    match = re.match(
        rf"^{re.escape(base_path.stem)}-(\d+){re.escape(base_path.suffix)}$",
        output_path.name,
    )
    if match:
        return int(match.group(1))
    return None


def resolve_output_path(
    base_path: Path,
    overwrite: bool,
    explicit_output_dir: bool,
) -> Path:
    if explicit_output_dir:
        return base_path
    if overwrite:
        return lmdb_shard_path(base_path, 1)
    return next_lmdb_shard_path(base_path)


def string_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        return [value.decode("utf-8")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in value
        ]
    return [str(value)]


def read_used_pickle_files(path: Path) -> List[str]:
    used_pickle_files = []
    try:
        meta = read_lmdb_meta(path)
        attrs = meta.get("attrs", {})
        used_pickle_files.extend(string_list(attrs.get("pickle_files")))
        used_pickle_files.extend(string_list(attrs.get("selected_pickle_files")))
        used_pickle_files.extend(string_list(attrs.get("pickle_paths")))
        used_pickle_files.extend(string_list(attrs.get("selected_pickle_paths")))
    except Exception as exc:
        print(f"[WARNING] Could not read LMDB metadata from {path}: {exc}")

    try:
        episode_index = read_lmdb_episode_index(path)
        used_pickle_files.extend(
            str(episode_meta["pickle_file"])
            for episode_meta in episode_index
            if "pickle_file" in episode_meta
        )
    except Exception as exc:
        print(f"[WARNING] Could not read LMDB episode index from {path}: {exc}")

    return used_pickle_files


def confirm_no_duplicate_pickles(
    selected_pickle_files: List[str],
    selected_pickle_paths: List[str],
    existing_lmdb_paths: List[Path],
):
    if not existing_lmdb_paths:
        return

    selected = set(selected_pickle_files) | set(selected_pickle_paths)
    duplicates_by_lmdb = {}
    for lmdb_path in existing_lmdb_paths:
        used = set(read_used_pickle_files(lmdb_path))
        overlap = sorted(selected & used)
        if overlap:
            duplicates_by_lmdb[lmdb_path] = overlap

    if not duplicates_by_lmdb:
        return

    print("[WARNING] Some selected pickle files already appear in existing LMDB shards:")
    for lmdb_path, overlaps in duplicates_by_lmdb.items():
        print(f"[WARNING]   {lmdb_path}")
        for pickle_file in overlaps[:20]:
            print(f"[WARNING]     {pickle_file}")
        if len(overlaps) > 20:
            print(f"[WARNING]     ... and {len(overlaps) - 20} more")

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Duplicate pickle files were detected and stdin is not interactive; aborting."
        )

    answer = input("Continue and write a shard with duplicate pickle files? [y/N] ")
    if answer.strip().lower() != "y":
        raise RuntimeError("Aborted because selected pickle files were already used.")


def log_first_pickle_shape(pickle_paths: List[Path]):
    total_files = len(pickle_paths)
    if total_files == 0:
        print("[WARNING] No pickle files found for the specified criteria.")
        return

    first_pickle_data = load_pickle_path(pickle_paths[0])
    print("[INFO] Shape of the first pickle file's data:")
    for key, value in first_pickle_data.items():
        if key in {"success", "task", "action_type"}:
            print(f"{key}: {value} (type: {type(value)})")
        elif key in {"rewards", "actions"}:
            print(f"{key}: shape {np.shape(value)}")
        elif key == "observations":
            print(f"{key}: number of observations {len(value)}")
            if len(value) > 0:
                for obs_key, obs_value in value[0].items():
                    if obs_key == "robot_state" and isinstance(obs_value, dict):
                        for sub_key, sub_value in obs_value.items():
                            print(f"  robot_state/{sub_key}: shape {np.shape(sub_value)}")
                    elif isinstance(obs_value, np.ndarray):
                        print(f"  {obs_key}: shape {obs_value.shape}")
                    else:
                        print(f"  {obs_key}: type {type(obs_value)}")


def gather_pickle_paths(args, task_episode_limits: Dict[str, int]) -> List[Path]:
    rng = random.Random(args.random_seed)

    if args.input_dir is not None:
        if task_episode_limits:
            raise ValueError(
                "--task-episode-limit is not supported together with --input-dir."
            )
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        pickle_paths = order_pickle_paths(
            list(input_dir.rglob("*.pkl*")),
            randomize_order=args.randomize_order,
            rng=rng,
        )
        print(f"Using explicit input directory: {input_dir}")
        return pickle_paths

    tasks = args.task if isinstance(args.task, list) else [args.task]
    selected_paths = []

    for task in tasks:
        task_paths = order_pickle_paths(
            get_raw_paths(
                controller=args.controller,
                domain=args.domain,
                task=task,
                demo_source=args.source,
                randomness=args.randomness,
                demo_outcome=args.demo_outcome,
                suffix=args.suffix,
            ),
            randomize_order=args.randomize_order,
            rng=rng,
        )

        task_limit = task_episode_limits.get(task)
        if task_limit is not None:
            task_paths = task_paths[:task_limit]

        print(f"[INFO] Task {task}: selected {len(task_paths)} pickle files")
        selected_paths.extend(task_paths)

    if args.randomize_order:
        rng.shuffle(selected_paths)
    else:
        selected_paths = order_pickle_paths(
            selected_paths,
            randomize_order=False,
            rng=rng,
        )

    return selected_paths


def load_episode_groups_manifest(
    manifest_path: Path,
    selected_pickle_paths: List[Path],
) -> Tuple[List[dict], dict]:
    """Resolve a portable list of logical episodes from aligned pickle segments.

    Manifest entries may contain absolute paths from another host.  Resolution
    deliberately uses the unique basename among the already-selected inputs so
    an audited grouping can be transferred together with the source dataset.
    Every selected pickle must appear exactly once; partial or duplicate
    grouping fails closed.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("--episode-groups-json must contain a JSON object.")
    raw_groups = payload.get("episode_groups")
    if raw_groups is None:
        raw_groups = payload.get("selective_episode_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError(
            "--episode-groups-json must contain a non-empty episode_groups or "
            "selective_episode_groups list."
        )

    by_name = defaultdict(list)
    for path in selected_pickle_paths:
        by_name[Path(path).name].append(Path(path).resolve())
    ambiguous = sorted(name for name, paths in by_name.items() if len(paths) != 1)
    if ambiguous:
        raise ValueError(
            "Episode grouping requires unique pickle basenames; ambiguous names: "
            + ", ".join(ambiguous[:20])
        )

    groups = []
    used = []
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise ValueError(f"Episode group {group_index} must be a JSON object.")
        raw_files = raw_group.get("pickle_files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError(
                f"Episode group {group_index} must contain non-empty pickle_files."
            )
        paths = []
        for raw_file in raw_files:
            name = Path(str(raw_file)).name
            matches = by_name.get(name, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Episode group {group_index} references unknown pickle {name!r}."
                )
            paths.append(matches[0])
            used.append(matches[0])
        source = raw_group.get("source", "logical")
        group_id = raw_group.get("group_id", f"{source}.stitched-g{group_index:03d}")
        groups.append(
            {
                "group_id": str(group_id),
                "source": str(source),
                "paths": paths,
                "segments": list(raw_group.get("segments", [])),
            }
        )

    selected_set = {Path(path).resolve() for path in selected_pickle_paths}
    used_set = set(used)
    if len(used) != len(used_set):
        raise ValueError("Episode grouping contains a pickle more than once.")
    missing = sorted(str(path) for path in selected_set - used_set)
    extra = sorted(str(path) for path in used_set - selected_set)
    if missing or extra:
        raise ValueError(
            "Episode grouping must cover every selected pickle exactly once; "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )
    return groups, payload


def combine_processed_episode_group(group: dict, episode_parts: List[dict]) -> dict:
    if not episode_parts:
        raise ValueError(f"Episode group {group['group_id']!r} is empty.")
    for scalar_key in ("task", "success", "env"):
        values = [part.get(scalar_key) for part in episode_parts]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"Episode group {group['group_id']!r} has inconsistent "
                f"{scalar_key}: {values!r}."
            )

    combined = {
        key: np.concatenate([np.asarray(part[key]) for part in episode_parts], axis=0)
        for key in (*IMAGE_KEYS, *LOWDIM_KEYS)
    }
    combined.update(
        {
            "episode_length": int(sum(part["episode_length"] for part in episode_parts)),
            "task": episode_parts[0]["task"],
            "success": episode_parts[0]["success"],
            "env": episode_parts[0].get("env"),
            "pickle_file": str(group["group_id"]),
            "source_pickle_files": [part["pickle_file"] for part in episode_parts],
            "source_pickle_schemas": sorted(
                {
                    part["source_pickle_schema"]
                    for part in episode_parts
                    if part.get("source_pickle_schema") is not None
                }
            ),
            "source": str(group.get("source", "")),
            "segments": list(group.get("segments", [])),
        }
    )
    if any(
        len(np.asarray(combined[key])) != combined["episode_length"]
        for key in (*IMAGE_KEYS, *LOWDIM_KEYS)
    ):
        raise RuntimeError(
            f"Episode group {group['group_id']!r} produced inconsistent lengths."
        )
    return combined


def process_episode_group(
    group,
    noop_threshold,
    resize_image,
    image_size,
    image_annotation_mode,
    required_source_image_annotation_mode,
    required_annotation_source,
    timeline_mode,
    timeline_frequency_hz,
    max_timeline_residual_ms,
    max_camera_residual_ms,
):
    if timeline_mode == "legacy-real-10hz":
        if len(group["paths"]) != 1:
            raise ValueError(
                "legacy-real-10hz requires exactly one raw pickle per episode"
            )
        path = group["paths"][0]
        trajectory = load_pickle_path(path)
        reconstructed, timeline_report = reconstruct_legacy_real_trajectory(
            trajectory,
            frequency_hz=timeline_frequency_hz,
            max_quantization_residual_ms=max_timeline_residual_ms,
            max_camera_residual_ms=max_camera_residual_ms,
            image_annotation_mode=image_annotation_mode,
        )
        processed = process_pickle_file(
            path,
            noop_threshold=noop_threshold,
            calculate_pos_action_from_delta=True,
            resize_image=resize_image,
            image_size=image_size,
            # The reconstructed in-memory trajectory has already rendered only
            # valid visual anchors.  Rendering here would mark zero placeholders.
            image_annotation_mode="none",
            required_source_image_annotation_mode=None,
            required_annotation_source=None,
            include_env_metadata=True,
            trajectory_data=reconstructed,
        )
        processed["timeline_report"] = timeline_report
        return processed

    parts = [
        process_pickle_file(
            path,
            noop_threshold=noop_threshold,
            calculate_pos_action_from_delta=True,
            resize_image=resize_image,
            image_size=image_size,
            image_annotation_mode=image_annotation_mode,
            required_source_image_annotation_mode=required_source_image_annotation_mode,
            required_annotation_source=required_annotation_source,
            include_env_metadata=True,
        )
        for path in group["paths"]
    ]
    return combine_processed_episode_group(group, parts)


def process_batch(
    batch_groups,
    noop_threshold,
    n_cpus,
    resize_image,
    image_size,
    image_annotation_mode,
    required_source_image_annotation_mode,
    required_annotation_source,
    timeline_mode,
    timeline_frequency_hz,
    max_timeline_residual_ms,
    max_camera_residual_ms,
):
    if n_cpus <= 1:
        return [
            process_episode_group(
                group,
                noop_threshold,
                resize_image=resize_image,
                image_size=image_size,
                image_annotation_mode=image_annotation_mode,
                required_source_image_annotation_mode=required_source_image_annotation_mode,
                required_annotation_source=required_annotation_source,
                timeline_mode=timeline_mode,
                timeline_frequency_hz=timeline_frequency_hz,
                max_timeline_residual_ms=max_timeline_residual_ms,
                max_camera_residual_ms=max_camera_residual_ms,
            )
            for group in batch_groups
        ]

    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        return list(
            executor.map(
                lambda group: process_episode_group(
                    group,
                    noop_threshold,
                    resize_image=resize_image,
                    image_size=image_size,
                    image_annotation_mode=image_annotation_mode,
                    required_source_image_annotation_mode=required_source_image_annotation_mode,
                    required_annotation_source=required_annotation_source,
                    timeline_mode=timeline_mode,
                    timeline_frequency_hz=timeline_frequency_hz,
                    max_timeline_residual_ms=max_timeline_residual_ms,
                    max_camera_residual_ms=max_camera_residual_ms,
                ),
                batch_groups,
            )
        )


def build_lowdim_specs(example_episode_data):
    return {
        key: {
            "dtype": str(np.asarray(example_episode_data[key]).dtype),
            "shape": list(np.asarray(example_episode_data[key]).shape),
        }
        for key in LOWDIM_KEYS
    }


def ensure_removed_output_path(output_path: Path):
    if not output_path.exists():
        return

    if output_path.is_dir():
        shutil.rmtree(output_path)
    else:
        output_path.unlink()


def main():
    require_lmdb()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller",
        "-c",
        type=str,
        required=True,
        choices=["osc", "diffik"],
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=["sim", "real", "distillation"],
        required=True,
    )
    parser.add_argument(
        "--task",
        "-f",
        type=str,
        nargs="+",
        required=True,
        help="One or more task names. Multiple tasks will be merged into one LMDB.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["scripted", "rollout", "teleop", "augmentation"],
        required=True,
    )
    parser.add_argument(
        "--randomness",
        "-r",
        type=str,
        choices=["low", "low_perturb", "med", "med_perturb", "high", "high_perturb"],
        required=True,
    )
    parser.add_argument(
        "--demo-outcome",
        "-o",
        type=str,
        choices=["success", "failure", "partial_success"],
        required=True,
    )
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--num-pickles",
        type=int,
        default=None,
        help="Maximum number of newest pickle files to process in this shard.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--randomize-order", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-cpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--map-size-gb", type=int, default=1024)
    parser.add_argument(
        "--frame-compression",
        choices=("none", "zstd"),
        default=DEFAULT_FRAME_COMPRESSION,
        help=(
            "Per-frame image payload compression. New LMDB datasets default to "
            f"{DEFAULT_FRAME_COMPRESSION}; pass 'none' only for compatibility or "
            "controlled benchmarks."
        ),
    )
    parser.add_argument(
        "--frame-compression-level",
        type=int,
        default=DEFAULT_FRAME_COMPRESSION_LEVEL,
        help=(
            "Zstd compression level when --frame-compression=zstd "
            f"(default: {DEFAULT_FRAME_COMPRESSION_LEVEL})."
        ),
    )
    parser.add_argument(
        "--resize-image",
        action="store_true",
        help="Resize images to standard dimensions (240x320x3).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Center-crop both RGB-D streams to IMAGE_SIZE x IMAGE_SIZE before "
            "storage. Use 224 to merge FurnitureBench, AutoMate, and ManiSkill."
        ),
    )
    parser.add_argument(
        "--image-annotation-mode",
        choices=IMAGE_ANNOTATION_MODES,
        default="none",
        help="Deterministically render saved 2D metadata onto color_image2 before LMDB encoding.",
    )
    parser.add_argument(
        "--annotation-source",
        choices=("scripted",),
        default=None,
        help=(
            "Policy-target provenance. Dataset-generation campaigns must pass "
            "--annotation-source scripted explicitly."
        ),
    )
    parser.add_argument(
        "--timeline-mode",
        choices=("pickle", "legacy-real-10hz"),
        default="pickle",
        help=(
            "Use the authoritative timeline already stored in the pickle, or "
            "explicitly reconstruct old deoxys raw_v2 real demonstrations."
        ),
    )
    parser.add_argument("--timeline-frequency-hz", type=float, default=10.0)
    parser.add_argument("--max-timeline-residual-ms", type=float, default=75.0)
    parser.add_argument("--max-camera-residual-ms", type=float, default=75.0)
    parser.add_argument(
        "--require-source-image-annotation-mode",
        choices=IMAGE_ANNOTATION_MODES,
        default=None,
        help=(
            "Reject source pickles unless their top-level image_annotation_mode "
            "matches this value. New two-stage campaigns should pass 'none'."
        ),
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=None,
        help="Optional JSON object recorded verbatim in LMDB metadata.",
    )
    parser.add_argument(
        "--episode-groups-json",
        type=Path,
        default=None,
        help=(
            "Optional audited grouping of input pickle segments into logical "
            "episodes. Every selected pickle must appear exactly once."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to the directory containing pkl files",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to save the LMDB directory",
        default=None,
    )
    parser.add_argument(
        "--task-episode-limit",
        type=str,
        nargs="*",
        default=None,
        help="Per-task episode limits, for example: one_leg=100 round_table=50",
    )
    parser.add_argument(
        "--debug-storage-stats",
        action="store_true",
        help="Print detailed LMDB payload estimates for image and low-dimensional data.",
    )
    args = parser.parse_args()

    if args.frame_compression == "zstd":
        require_zstandard()
    if args.image_size is not None and args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.resize_image and args.image_size is not None:
        raise ValueError("--resize-image and --image-size cannot be used together.")
    if args.timeline_frequency_hz <= 0:
        raise ValueError("--timeline-frequency-hz must be positive.")
    if args.timeline_mode == "legacy-real-10hz":
        if args.domain != "real":
            raise ValueError("legacy-real-10hz is only valid for --domain real")
        if args.annotation_source != "scripted":
            raise ValueError(
                "legacy-real-10hz requires --annotation-source scripted"
            )
        if args.episode_groups_json is not None:
            raise ValueError(
                "legacy-real-10hz preserves one source pickle per episode and "
                "cannot be combined with --episode-groups-json"
            )

    provenance = {}
    if args.provenance_json is not None:
        provenance_path = args.provenance_json.expanduser().resolve()
        provenance = json.loads(provenance_path.read_text())
        if not isinstance(provenance, dict):
            raise ValueError("--provenance-json must contain a JSON object.")

    assert not args.randomize_order or args.offset == 0, "Cannot offset with randomize"
    if args.offset < 0:
        raise ValueError(f"--offset must be non-negative, got {args.offset}.")
    if args.num_pickles is not None and args.num_pickles <= 0:
        raise ValueError(f"--num-pickles must be positive, got {args.num_pickles}.")
    if args.episode_groups_json is not None and (
        args.offset != 0
        or args.num_pickles is not None
        or args.randomize_order
        or args.task_episode_limit
    ):
        raise ValueError(
            "--episode-groups-json cannot be combined with offset, num-pickles, "
            "randomize-order, or task-episode-limit."
        )

    task_episode_limits = parse_task_episode_limits(args.task_episode_limit)
    pickle_paths = gather_pickle_paths(args, task_episode_limits)
    log_first_pickle_shape(pickle_paths)

    start = args.offset
    end = (
        args.offset + args.num_pickles
        if args.num_pickles is not None
        else len(pickle_paths)
    )
    pickle_paths = pickle_paths[start:end]
    print(f"Found {len(pickle_paths)} pickle files after filtering")
    if len(pickle_paths) == 0:
        raise ValueError("No pickle files selected; refusing to create an empty LMDB dataset.")

    episode_groups_payload: Dict[str, Any] = {}
    if args.episode_groups_json is not None:
        episode_groups_manifest_path = (
            args.episode_groups_json.expanduser().resolve()
        )
        episode_groups_manifest_sha256 = hashlib.sha256(
            episode_groups_manifest_path.read_bytes()
        ).hexdigest()
        episode_groups, episode_groups_payload = load_episode_groups_manifest(
            episode_groups_manifest_path,
            pickle_paths,
        )
        print(
            f"Loaded {len(episode_groups)} audited logical episodes from "
            f"{args.episode_groups_json.expanduser().resolve()}"
        )
    else:
        episode_groups_manifest_path = None
        episode_groups_manifest_sha256 = None
        episode_groups = [
            {
                "group_id": pickle_identity(path),
                "source": Path(path).name,
                "paths": [path],
                "segments": [],
            }
            for path in pickle_paths
        ]

    selected_pickle_files = [pickle_identity(path) for path in pickle_paths]
    selected_pickle_paths = [absolute_pickle_path(path) for path in pickle_paths]

    explicit_output_dir = args.output_dir is not None
    if args.output_dir is not None:
        base_output_path = Path(args.output_dir).expanduser().resolve()
        print(f"Using explicit output path: {base_output_path}")
    else:
        base_output_path = get_processed_path(
            controller=args.controller,
            domain=args.domain,
            task=args.task,
            demo_source=args.source,
            randomness=args.randomness,
            demo_outcome=args.demo_outcome,
            suffix=args.output_suffix,
            dataset_format="lmdb",
        )

    duplicate_scan_base = unnumbered_lmdb_base_path(base_output_path)
    existing_lmdb_paths = expand_lmdb_shard_paths(duplicate_scan_base)
    output_path = resolve_output_path(
        base_output_path,
        overwrite=args.overwrite,
        explicit_output_dir=explicit_output_dir,
    )
    shard_index = infer_shard_index(duplicate_scan_base, output_path)

    print(f"Base output path: {base_output_path}")
    print(f"Resolved output path: {output_path}")
    if output_path.exists():
        if not args.overwrite:
            raise ValueError(
                f"Output path already exists: {output_path}. Use --overwrite to overwrite."
            )

    confirm_no_duplicate_pickles(
        selected_pickle_files=selected_pickle_files,
        selected_pickle_paths=selected_pickle_paths,
        existing_lmdb_paths=existing_lmdb_paths,
    )

    if output_path.exists():
        ensure_removed_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    noop_threshold = 0.0
    n_cpus = min(os.cpu_count() or 1, args.n_cpus)
    batch_size = max(1, args.batch_size)
    time_created = datetime.now().astimezone().isoformat()
    env = open_lmdb_env(output_path, readonly=False)
    env.set_mapsize(int(args.map_size_gb) * (1024**3))

    episode_index = []
    normalizer_stats = {}
    depth_moments = empty_depth_moments()
    env_counts = defaultdict(int)
    frame_specs = None
    lowdim_specs = None
    global_frame_idx = 0
    global_episode_idx = 0
    selected_task_counts = {task: 0 for task in args.task}
    running_payload_bytes = 0
    timeline_totals = defaultdict(int)
    timeline_residual_max_ms = 0.0
    source_pickle_schema_counts = defaultdict(int)

    total_batches = (
        (len(episode_groups) + batch_size - 1) // batch_size
        if episode_groups
        else 0
    )
    print(
        f"Processing pickle files with {n_cpus} CPUs, batch_size={batch_size}, "
        f"noop_threshold={noop_threshold}, total_batches={total_batches}"
    )

    for batch_start in range(0, len(episode_groups), batch_size):
        batch_groups = episode_groups[batch_start : batch_start + batch_size]
        batch_results = process_batch(
            batch_groups,
            noop_threshold=noop_threshold,
            n_cpus=n_cpus,
            resize_image=args.resize_image,
            image_size=args.image_size,
            image_annotation_mode=args.image_annotation_mode,
            required_source_image_annotation_mode=args.require_source_image_annotation_mode,
            required_annotation_source=(
                args.annotation_source if args.timeline_mode == "pickle" else None
            ),
            timeline_mode=args.timeline_mode,
            timeline_frequency_hz=args.timeline_frequency_hz,
            max_timeline_residual_ms=args.max_timeline_residual_ms,
            max_camera_residual_ms=args.max_camera_residual_ms,
        )
        batch_image_bytes = 0
        batch_lowdim_bytes = 0
        batch_timesteps = 0

        with env.begin(write=True) as txn:
            for episode_data in batch_results:
                timeline_report = episode_data.get("timeline_report")
                if timeline_report is not None:
                    for key in (
                        "source_actions",
                        "timeline_steps",
                        "synthetic_noop_steps",
                        "valid_observations",
                    ):
                        timeline_totals[key] += int(timeline_report[key])
                    timeline_residual_max_ms = max(
                        timeline_residual_max_ms,
                        float(timeline_report["quantization_residual_ms_max"]),
                    )
                if frame_specs is None:
                    frame_specs = build_frame_specs(
                        {
                            key: episode_data[key][0]
                            for key in IMAGE_KEYS
                        }
                    )
                    if args.frame_compression == "zstd":
                        frame_specs["compression"] = {
                            "codec": "zstd",
                            "level": int(args.frame_compression_level),
                        }
                    lowdim_specs = build_lowdim_specs(episode_data)
                    log_lmdb_storage_layout(
                        frame_specs,
                        lowdim_specs,
                        resize_image=args.resize_image,
                        image_size=args.image_size,
                    )

                episode_length = int(episode_data["episode_length"])
                frame_start = global_frame_idx
                frame_end = frame_start + episode_length

                lowdim_payload = {
                    key: np.asarray(episode_data[key]) for key in LOWDIM_KEYS
                }
                packed_lowdim_payload = pack_named_arrays(lowdim_payload)
                txn.put(
                    episode_data_key(global_episode_idx),
                    packed_lowdim_payload,
                )
                packed_lowdim_nbytes = len(packed_lowdim_payload)
                batch_lowdim_bytes += packed_lowdim_nbytes
                batch_timesteps += episode_length

                if args.debug_storage_stats and global_episode_idx == 0:
                    log_episode_storage_debug(
                        episode_data,
                        frame_specs,
                        packed_lowdim_nbytes,
                    )

                for local_frame_idx in range(episode_length):
                    frame_payload = {
                        key: episode_data[key][local_frame_idx]
                        for key in IMAGE_KEYS
                    }
                    packed_frame = pack_frame(frame_payload, frame_specs)
                    txn.put(
                        frame_key(global_frame_idx + local_frame_idx),
                        packed_frame,
                    )
                    batch_image_bytes += len(packed_frame)

                env_label = normalize_env_label(episode_data.get("env"))
                source_schemas = episode_data.get("source_pickle_schemas")
                if source_schemas is None:
                    source_schema = episode_data.get("source_pickle_schema")
                    source_schemas = [] if source_schema is None else [source_schema]
                for source_schema in source_schemas:
                    source_pickle_schema_counts[str(source_schema)] += 1
                episode_index.append(
                    {
                        "episode_idx": global_episode_idx,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "task": episode_data["task"],
                        "success": int(episode_data["success"]),
                        "pickle_file": episode_data["pickle_file"],
                        "source_pickle_files": episode_data.get(
                            "source_pickle_files", [episode_data["pickle_file"]]
                        ),
                        "source": episode_data.get("source"),
                        "source_pickle_schemas": list(source_schemas),
                        "segments": episode_data.get("segments", []),
                        "env": env_label,
                    }
                )
                env_counts[env_label if env_label is not None else "<missing>"] += 1
                selected_task_counts.setdefault(episode_data["task"], 0)
                selected_task_counts[episode_data["task"]] += 1

                obs_valid = np.asarray(
                    lowdim_payload.get(
                        "obs_valid", np.ones(episode_length, dtype=np.bool_)
                    ),
                    dtype=np.bool_,
                )
                if not np.any(obs_valid):
                    raise ValueError(
                        f"Episode {global_episode_idx} has no valid visual observations."
                    )
                stats_payload = dict(lowdim_payload)
                for observation_key in ("robot_state", "skill", "parts_poses"):
                    stats_payload[observation_key] = np.asarray(
                        lowdim_payload[observation_key]
                    )[obs_valid]
                episode_stats = compute_normalizer_stats_from_dict(stats_payload)
                merge_normalizer_stats(normalizer_stats, episode_stats)
                for camera_name, depth_key in DEPTH_CAMERA_KEYS.items():
                    update_depth_moments(
                        depth_moments,
                        camera_name,
                        np.asarray(episode_data[depth_key])[obs_valid],
                    )

                global_frame_idx = frame_end
                global_episode_idx += 1

        running_payload_bytes += batch_image_bytes + batch_lowdim_bytes

        print(
            f"[INFO] Written batch {batch_start // batch_size + 1}/{total_batches}, "
            f"timesteps so far: {global_frame_idx}, episodes so far: {global_episode_idx}"
        )
        if args.debug_storage_stats:
            log_batch_storage_debug(
                batch_index=batch_start // batch_size + 1,
                total_batches=total_batches,
                batch_timesteps=batch_timesteps,
                batch_episodes=len(batch_results),
                batch_image_bytes=batch_image_bytes,
                batch_lowdim_bytes=batch_lowdim_bytes,
                running_total_bytes=running_payload_bytes,
            )

    serialized_normalizer_stats = serialize_normalizer_stats(normalizer_stats)
    depth_normalizer_stats = finalize_depth_moments(depth_moments)
    for camera_name, camera_stats in depth_normalizer_stats.items():
        if int(camera_stats["count"]) == 0:
            print(
                f"[WARNING] No finite non-zero depth pixels found for {camera_name}; "
                "this LMDB cannot be used for new RGBD training."
            )
        elif float(camera_stats["std"]) == 0.0:
            print(
                f"[WARNING] Depth standard deviation is zero for {camera_name}; "
                "this LMDB cannot be used for new RGBD training."
            )
    attrs = {
        "time_created": time_created,
        "time_finished": datetime.now().astimezone().isoformat(),
        "noop_threshold": noop_threshold,
        "rotation_mode": "rot_6d",
        "n_episodes": global_episode_idx,
        "n_timesteps": global_frame_idx,
        "mean_episode_length": (
            round(global_frame_idx / global_episode_idx) if global_episode_idx else 0
        ),
        "calculated_pos_action_from_delta": True,
        "randomize_order": args.randomize_order,
        "random_seed": args.random_seed,
        "pickle_order": "random" if args.randomize_order else "newest",
        "offset": args.offset,
        "num_pickles": args.num_pickles,
        "selected_pickle_count": len(pickle_paths),
        "episode_grouping_enabled": args.episode_groups_json is not None,
        "episode_groups_json": (
            None
            if episode_groups_manifest_path is None
            else str(episode_groups_manifest_path)
        ),
        "episode_groups_schema": episode_groups_payload.get("schema"),
        "episode_groups_manifest_sha256": episode_groups_manifest_sha256,
        "episode_groups_counts": episode_groups_payload.get("counts"),
        "episode_groups_policy": episode_groups_payload.get(
            "selective_stitch_policy"
        ),
        "selected_episode_group_count": len(episode_groups),
        "pickle_files": selected_pickle_files,
        "pickle_paths": selected_pickle_paths,
        "shard_index": shard_index,
        "shard_path": str(output_path),
        "demo_source": args.source,
        "controller": args.controller,
        "domain": args.domain if args.domain == "real" else "sim",
        "task": args.task if len(args.task) > 1 else args.task[0],
        "tasks": args.task,
        "selected_task_counts": selected_task_counts,
        "randomness": args.randomness,
        "demo_outcome": args.demo_outcome,
        "suffix": args.suffix,
        "output_suffix": args.output_suffix,
        "storage_format": "lmdb",
        "frame_compression": frame_specs.get("compression", {"codec": "none"}),
        "image_annotation_mode": args.image_annotation_mode,
        "source_image_annotation_mode": args.require_source_image_annotation_mode,
        "annotation_source": args.annotation_source,
        "timeline_mode": args.timeline_mode,
        "timeline_frequency_hz": args.timeline_frequency_hz,
        "max_timeline_residual_ms": args.max_timeline_residual_ms,
        "max_camera_residual_ms": args.max_camera_residual_ms,
        "timeline_totals": dict(timeline_totals),
        "timeline_quantization_residual_ms_max": timeline_residual_max_ms,
        "source_pickle_schema_counts": dict(sorted(source_pickle_schema_counts.items())),
        "contains_v6_offline_buffered": (
            V6_BUFFERED_SCHEMA in source_pickle_schema_counts
        ),
        "stored_image_size": args.image_size,
        "provenance": provenance,
        "normalizer_stats": serialized_normalizer_stats,
        "normalizer_stats_keys": list(NORMALIZER_STATS_KEYS),
        "gripper_width_encoding": GRIPPER_WIDTH_ENCODING,
        "gripper_width_open_threshold_m": GRIPPER_OPEN_THRESHOLD_METERS,
        DEPTH_NORMALIZER_STATS_ATTR: depth_normalizer_stats,
        "env_counts": dict(sorted(env_counts.items())),
    }
    meta = {
        "format": "robust_rearrangement_lmdb",
        "format_version": LMDB_FORMAT_VERSION,
        "attrs": attrs,
        "frame_specs": frame_specs,
        "lowdim_specs": lowdim_specs or {},
    }

    with env.begin(write=True) as txn:
        txn.put(META_KEY, json_dumps_bytes(meta))
        txn.put(EPISODE_INDEX_KEY, json_dumps_bytes(episode_index))

    env.sync()
    env.close()
    if args.debug_storage_stats:
        print(
            "[DEBUG] Final LMDB payload estimate (metadata excluded): "
            f"{format_bytes(running_payload_bytes)} across {global_frame_idx} timesteps"
        )
    print("[INFO] LMDB processing complete.")


if __name__ == "__main__":
    main()
