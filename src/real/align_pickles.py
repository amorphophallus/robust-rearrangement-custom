"""Build non-destructive, timestamp-aligned real-world trajectory pickles.

The legacy recorder sampled the current robot state and the latest completed
PromptDA result in the same Python loop.  Their wall-clock ages therefore differ.
This tool keeps action/control time as the step time, independently matches one
unique wrist and front source frame, and writes only contiguous valid segments.

Example (dry-run is the default)::

    python -m src.real.align_pickles \
      --input-dir data/raw/osc/real/one_leg/teleop/low/success/annotated \
      --output-dir data/aligned/osc/real/one_leg/teleop/low/success

Add ``--write`` after reviewing the report.  Feed that explicit output directory
to ``process_pickles_to_lmdb.py --input-dir``; the source pickles are untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from src.eval.real_skill_annotation_util import (
    RealSkillAnnotationSession,
    _atomic_pickle_dump,
    load_trajectory_pickle,
)
from src.real.time_alignment import (
    NS_PER_MS,
    TimestampMatch,
    contiguous_segments,
    monotonic_nearest_unique_match,
)


ALIGNMENT_SCHEMA = "rr_real_timestamp_alignment_v1"
LEGACY_ROTATION_CLIP_RAD = 0.35

WRIST_FIELDS = (
    "color_image1",
    "depth_image1",
    "depth_image1_realsense",
    "wrist_sensor_timestamp_ms",
    "wrist_frame_number",
)
FRONT_FIELDS = (
    "color_image2",
    "depth_image2",
    "depth_image2_realsense",
    "front_sensor_timestamp_ms",
    "front_frame_number",
    "camera_to_april",
    "camera_pose_samples",
    "camera_pose_samples_required",
    "parts_poses",
    "parts_founds",
    "parts_pose_valid",
    "parts_pose_age_ms",
    "parts_poses_frame",
)


def _sensor_time_ns(observation: Mapping[str, Any], camera: str) -> int:
    value = float(observation[f"{camera}_sensor_timestamp_ms"])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {camera} sensor timestamp")
    return int(round(value * NS_PER_MS))


def _unique_camera_sources(
    observations: Sequence[Mapping[str, Any]], camera: str
) -> Tuple[List[int], List[int]]:
    """Return source observation indices/times after logical frame deduplication."""

    indices: List[int] = []
    times: List[int] = []
    seen = set()
    for index, observation in enumerate(observations):
        timestamp_ns = _sensor_time_ns(observation, camera)
        frame_number = observation.get(f"{camera}_frame_number")
        identity = (frame_number, timestamp_ns)
        if identity in seen:
            continue
        seen.add(identity)
        indices.append(index)
        times.append(timestamp_ns)
    if times and any(right < left for left, right in zip(times, times[1:])):
        raise ValueError(f"{camera} source timestamps are not monotonic")
    return indices, times


def _camera_matches(
    observations: Sequence[Mapping[str, Any]],
    action_times_ns: Sequence[int],
    camera: str,
    max_residual_ms: float,
) -> Dict[int, TimestampMatch]:
    source_observation_indices, source_times_ns = _unique_camera_sources(
        observations, camera
    )
    compact_matches = monotonic_nearest_unique_match(
        action_times_ns,
        source_times_ns,
        max_residual_ms=max_residual_ms,
    )
    return {
        action_index: TimestampMatch(
            target_index=match.target_index,
            source_index=source_observation_indices[match.source_index],
            target_time_ns=match.target_time_ns,
            source_time_ns=match.source_time_ns,
        )
        for action_index, match in compact_matches.items()
    }


def _copy_present_fields(
    destination: MutableMapping[str, Any],
    source: Mapping[str, Any],
    fields: Iterable[str],
) -> None:
    for field in fields:
        if field in source:
            destination[field] = source[field]
        else:
            destination.pop(field, None)


def build_aligned_observation(
    observations: Sequence[Mapping[str, Any]],
    action_index: int,
    front_match: TimestampMatch,
    wrist_match: TimestampMatch,
) -> Dict[str, Any]:
    """Compose state-at-action-time with independently matched camera frames."""

    state_source = observations[action_index]
    front_source = observations[front_match.source_index]
    wrist_source = observations[wrist_match.source_index]
    aligned = dict(state_source)
    # Annotation dictionaries are mutated by the offline annotator; isolate
    # them from the source pickle while keeping large image arrays zero-copy.
    if isinstance(state_source.get("robot_state"), Mapping):
        aligned["robot_state"] = copy.deepcopy(state_source["robot_state"])
    _copy_present_fields(aligned, wrist_source, WRIST_FIELDS)
    _copy_present_fields(aligned, front_source, FRONT_FIELDS)

    action_time_ns = int(front_match.target_time_ns)
    aligned.update(
        {
            "step_timestamp_ns": action_time_ns,
            "action_wall_time_ns": action_time_ns,
            "state_source_index": int(action_index),
            "front_source_index": int(front_match.source_index),
            "wrist_source_index": int(wrist_match.source_index),
            "front_source_wall_time_ns": int(front_match.source_time_ns),
            "wrist_source_wall_time_ns": int(wrist_match.source_time_ns),
            "front_time_residual_ms": float(front_match.residual_ms),
            "wrist_time_residual_ms": float(wrist_match.residual_ms),
            "camera_anchor": "front",
            # Preserve the legacy field for readers while making its camera
            # meaning explicit.  Sensor timestamps are the alignment source.
            "camera_capture_wall_time_ns": front_source.get(
                "camera_capture_wall_time_ns"
            ),
            "front_camera_capture_wall_time_ns": front_source.get(
                "camera_capture_wall_time_ns"
            ),
            "wrist_camera_capture_wall_time_ns": wrist_source.get(
                "camera_capture_wall_time_ns"
            ),
            "front_prompt_depth_observed_wall_time_ns": front_source.get(
                "control_wall_time_ns"
            ),
            "wrist_prompt_depth_observed_wall_time_ns": wrist_source.get(
                "control_wall_time_ns"
            ),
            "prompt_depth_source_wall_time_ns": front_source.get(
                "prompt_depth_source_wall_time_ns",
                front_source.get("camera_capture_wall_time_ns"),
            ),
        }
    )
    if front_source.get("prompt_depth_ready_wall_time_ns") is not None:
        aligned["front_prompt_depth_ready_wall_time_ns"] = front_source[
            "prompt_depth_ready_wall_time_ns"
        ]
    else:
        aligned.pop("front_prompt_depth_ready_wall_time_ns", None)
    if wrist_source.get("prompt_depth_ready_wall_time_ns") is not None:
        aligned["wrist_prompt_depth_ready_wall_time_ns"] = wrist_source[
            "prompt_depth_ready_wall_time_ns"
        ]
    else:
        aligned.pop("wrist_prompt_depth_ready_wall_time_ns", None)
    return aligned


def align_trajectory(
    data: Mapping[str, Any],
    *,
    max_camera_residual_ms: float = 75.0,
    max_action_gap_ms: float = 150.0,
    min_segment_steps: int = 8,
    rerun_annotations: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Align one trajectory and return output segments plus a JSON-safe report."""

    observations = data.get("observations")
    actions = data.get("actions")
    if not isinstance(observations, list) or not observations:
        raise ValueError("trajectory observations must be a non-empty list")
    if actions is None:
        raise ValueError("trajectory is missing actions")
    action_count = len(actions)
    if len(observations) not in {action_count, action_count + 1}:
        raise ValueError(
            f"expected N or N+1 observations, got {len(observations)} and {action_count} actions"
        )
    if data.get("action_type") not in {None, "delta"}:
        raise ValueError(
            "legacy real alignment expects saved delta actions; got "
            f"action_type={data.get('action_type')!r}"
        )

    source_actions = np.asarray(actions, dtype=np.float64)
    if source_actions.shape != (action_count, 8):
        raise ValueError(
            f"expected saved actions with shape ({action_count}, 8), got "
            f"{source_actions.shape}"
        )
    source_rotvec = Rotation.from_quat(source_actions[:, 3:7]).as_rotvec()
    source_rotation_norm = float(np.linalg.norm(source_rotvec))
    legacy_rotation_scale = (
        min(1.0, LEGACY_ROTATION_CLIP_RAD / source_rotation_norm)
        if source_rotation_norm > 0
        else 1.0
    )

    saved_action_times = data.get("action_timestamps_ns")
    if saved_action_times is not None:
        if len(saved_action_times) != action_count or any(
            value is None for value in saved_action_times
        ):
            raise ValueError("action_timestamps_ns must contain one time per action")
        action_times_ns = [int(value) for value in saved_action_times]
        action_time_source = "action_timestamps_ns"
    else:
        action_times_ns = [
            int(observations[index]["control_wall_time_ns"])
            for index in range(action_count)
        ]
        action_time_source = "legacy observation control_wall_time_ns proxy"
    if any(right <= left for left, right in zip(action_times_ns, action_times_ns[1:])):
        raise ValueError("control_wall_time_ns must increase strictly")
    front_matches = _camera_matches(
        observations, action_times_ns, "front", max_camera_residual_ms
    )
    wrist_matches = _camera_matches(
        observations, action_times_ns, "wrist", max_camera_residual_ms
    )
    both = sorted(set(front_matches) & set(wrist_matches))
    segments = contiguous_segments(
        both,
        action_times_ns,
        max_gap_ms=max_action_gap_ms,
        min_steps=min_segment_steps,
    )
    retained = {index for segment in segments for index in segment}

    aligned_by_index = {
        index: build_aligned_observation(
            observations,
            index,
            front_matches[index],
            wrist_matches[index],
        )
        for index in retained
    }

    annotation_session = None
    if rerun_annotations and retained:
        task = data.get("furniture", data.get("task"))
        camera_info = data.get("camera_info")
        annotation_session = RealSkillAnnotationSession(
            str(task), camera_info, mode="offline"
        )
        for index in sorted(retained):
            annotation_session.annotate_observation(aligned_by_index[index])

    outputs: List[Dict[str, Any]] = []
    for segment_number, indices in enumerate(segments):
        output = dict(data)
        output["observations"] = [aligned_by_index[index] for index in indices]
        output["actions"] = [actions[index] for index in indices]
        if "actions_original" in data:
            output["actions_original"] = [
                data["actions_original"][index] for index in indices
            ]
        if "rewards" in data:
            output["rewards"] = [data["rewards"][index] for index in indices]
        metadata = copy.deepcopy(data.get("metadata", {}))
        if not isinstance(metadata, MutableMapping):
            metadata = {}
        metadata.update(
            {
                "schema": ALIGNMENT_SCHEMA,
                "alignment_created_at": datetime.now().astimezone().isoformat(),
                "alignment_step_semantics": (
                    "state and delta action at action master time; front/wrist "
                    "PromptDA source frames matched independently"
                ),
                "alignment_action_time_source": action_time_source,
                "alignment_camera_anchor": "front",
                "alignment_max_camera_residual_ms": max_camera_residual_ms,
                "alignment_max_action_gap_ms": max_action_gap_ms,
                "alignment_min_segment_steps": min_segment_steps,
                "alignment_segment_number": segment_number,
                "alignment_source_action_start": indices[0],
                "alignment_source_action_end_exclusive": indices[-1] + 1,
                "alignment_source_state_timestamp": (
                    "legacy proxy: robot state sampled immediately before "
                    "control_wall_time_ns; transport source/receive times unavailable"
                ),
                "alignment_annotations_rerun": bool(rerun_annotations),
                # Splitting an episode would otherwise change RR's historical
                # whole-episode rotation scaling.  The LMDB processor consumes
                # this override so aligned labels remain action-equivalent to
                # processing the unsplit source pickle.
                "legacy_rotation_episode_scale": legacy_rotation_scale,
                "legacy_rotation_episode_norm_rad": source_rotation_norm,
                "legacy_rotation_episode_clip_rad": LEGACY_ROTATION_CLIP_RAD,
                "legacy_rotation_scale_source": "unsplit_source_episode",
                "num_observations": len(indices),
                "num_actions": len(indices),
            }
        )
        output["metadata"] = metadata
        if annotation_session is not None:
            annotation_session.update_trajectory_metadata(output)
        outputs.append(output)

    step_report = []
    for index in range(action_count):
        front = front_matches.get(index)
        wrist = wrist_matches.get(index)
        if index in retained:
            reason = "kept"
        elif front is None and wrist is None:
            reason = "missing_both_cameras"
        elif front is None:
            reason = "missing_front"
        elif wrist is None:
            reason = "missing_wrist"
        else:
            reason = "short_segment"
        step_report.append(
            {
                "action_index": index,
                "action_time_ns": int(action_times_ns[index]),
                "front_source_index": None if front is None else front.source_index,
                "front_source_time_ns": None if front is None else front.source_time_ns,
                "front_residual_ms": None if front is None else front.residual_ms,
                "wrist_source_index": None if wrist is None else wrist.source_index,
                "wrist_source_time_ns": None if wrist is None else wrist.source_time_ns,
                "wrist_residual_ms": None if wrist is None else wrist.residual_ms,
                "reason": reason,
            }
        )
    report = {
        "input_observations": len(observations),
        "input_actions": action_count,
        "action_time_source": action_time_source,
        "matched_front": len(front_matches),
        "matched_wrist": len(wrist_matches),
        "matched_both": len(both),
        "retained_actions": len(retained),
        "legacy_rotation_episode_scale": legacy_rotation_scale,
        "retention_fraction": len(retained) / action_count if action_count else 0.0,
        "segments": [
            {
                "segment_number": number,
                "start_action_index": indices[0],
                "end_action_index_exclusive": indices[-1] + 1,
                "steps": len(indices),
            }
            for number, indices in enumerate(segments)
        ],
        "reason_counts": dict(Counter(step["reason"] for step in step_report)),
        "steps": step_report,
    }
    return outputs, report


def _base_pickle_name(path: Path) -> str:
    name = path.name
    for suffix in (".pkl.xz", ".pkl.gz", ".pickle", ".pkl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-camera-residual-ms", type=float, default=75.0)
    parser.add_argument("--max-action-gap-ms", type=float, default=150.0)
    parser.add_argument("--min-segment-steps", type=int, default=8)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write aligned pickles. Without this flag only the report is computed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="Do not rerun skill/guidance annotations (diagnostics only).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("output directory must not be the input directory or its child")
    paths = sorted(input_dir.rglob("*.pkl"))
    paths += sorted(input_dir.rglob("*.pkl.gz"))
    paths += sorted(input_dir.rglob("*.pkl.xz"))
    paths = sorted(set(paths))
    if not paths:
        raise ValueError(f"no pickle files found below {input_dir}")

    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_dir / "alignment_manifest.json"
    )
    if args.write:
        output_dir.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() and not args.overwrite:
            raise FileExistsError(manifest_path)

    manifest: Dict[str, Any] = {
        "schema": ALIGNMENT_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "write" if args.write else "dry_run",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "parameters": {
            "max_camera_residual_ms": args.max_camera_residual_ms,
            "max_action_gap_ms": args.max_action_gap_ms,
            "min_segment_steps": args.min_segment_steps,
            "annotations_rerun": not args.skip_annotations,
        },
        "files": [],
    }
    total_actions = 0
    total_retained = 0
    total_segments = 0
    for file_number, path in enumerate(paths, start=1):
        print(f"[{file_number}/{len(paths)}] aligning {path.name}", flush=True)
        data = load_trajectory_pickle(path)
        outputs, report = align_trajectory(
            data,
            max_camera_residual_ms=args.max_camera_residual_ms,
            max_action_gap_ms=args.max_action_gap_ms,
            min_segment_steps=args.min_segment_steps,
            rerun_annotations=args.write and not args.skip_annotations,
        )
        output_paths = []
        if args.write:
            base = _base_pickle_name(path)
            for segment_number, output in enumerate(outputs):
                destination = output_dir / f"{base}.aligned-s{segment_number:02d}.pkl"
                if destination.exists() and not args.overwrite:
                    raise FileExistsError(destination)
                _atomic_pickle_dump(output, destination)
                output_paths.append(str(destination))
        report["source_path"] = str(path)
        report["output_paths"] = output_paths
        manifest["files"].append(report)
        total_actions += report["input_actions"]
        total_retained += report["retained_actions"]
        total_segments += len(report["segments"])
        print(
            f"  retained {report['retained_actions']}/{report['input_actions']} "
            f"({report['retention_fraction']:.2%}) in {len(report['segments'])} segments",
            flush=True,
        )
    manifest["summary"] = {
        "input_files": len(paths),
        "input_actions": total_actions,
        "retained_actions": total_retained,
        "retention_fraction": total_retained / total_actions if total_actions else 0.0,
        "output_segments": total_segments,
    }
    if args.write:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote manifest: {manifest_path}")
    else:
        print(json.dumps(manifest["summary"], indent=2))
        print("dry-run only; add --write to create aligned pickles and manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
