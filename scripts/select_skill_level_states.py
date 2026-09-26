#!/usr/bin/env python3
"""Select 48 normalized-progress states for one clean skill-level stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.eval.progress_schema import get_task_progress_labels
from src.eval.skill_level import (
    ASSEMBLY_COMPLETION_STAGE_PAIR_INDEX,
    INCLUDED_SKILL_STAGES,
    assembly_pair_mask_value,
    stage_assembly_pair_index,
)
from src.eval.state_bank import load_state_record


STRATA = (
    ("early", 0.10, 0.33, 32),
    ("middle", 0.40, 0.65, 8),
    ("late", 0.70, 0.85, 8),
)

MAX_STATES_PER_SOURCE_EPISODE = 2
MIN_SAME_EPISODE_PROGRESS_GAP = 0.20
MIN_SAME_EPISODE_FRAME_GAP = 3
SAME_EPISODE_PROGRESS_GAP_OVERRIDES = {
    ("lamp", "hood-base-pick"): 0.15,
}
POOLED_STAGE_RANGES = {
    ("lamp", "hood-base-pick"): (0.00, 0.33),
    ("lamp", "hood-base-place"): (0.00, 0.90),
}
LAMP_BULB_SCREW_MIN_HOOD_PICK_FRAMES = 3


def progress_stratum(progress: float) -> str | None:
    for name, low, high, _count in STRATA:
        if low <= progress <= high:
            return name
    return None


def lamp_bulb_screw_episode_gate(
    episode_entries: list[dict],
    *,
    min_hood_pick_frames: int = LAMP_BULB_SCREW_MIN_HOOD_PICK_FRAMES,
) -> tuple[bool, str, dict]:
    """Require a non-degenerate screw -> hood pick -> hood place chain."""
    if min_hood_pick_frames < 1:
        raise ValueError("min_hood_pick_frames must be positive")
    ordered = sorted(episode_entries, key=lambda item: int(item["frame_index"]))
    segments: list[dict] = []
    for entry in ordered:
        stage = entry.get("skill_state")
        frame = int(entry["frame_index"])
        if not segments or segments[-1]["stage"] != stage:
            segments.append(
                {
                    "stage": stage,
                    "first_frame": frame,
                    "last_frame": frame,
                    "frame_count": 1,
                }
            )
        else:
            segments[-1]["last_frame"] = frame
            segments[-1]["frame_count"] += 1

    screw_index = next(
        (
            index
            for index, segment in enumerate(segments)
            if segment["stage"] == "bulb-base-screw"
        ),
        None,
    )
    audit = {
        "required_transition": [
            "bulb-base-screw",
            "hood-base-pick",
            "hood-base-place",
        ],
        "min_hood_pick_frames": int(min_hood_pick_frames),
        "observed_segments": segments,
    }
    if screw_index is None:
        return False, "missing_bulb_screw", audit
    if screw_index + 1 >= len(segments):
        return False, "missing_stage_after_bulb_screw", audit
    hood_pick = segments[screw_index + 1]
    if hood_pick["stage"] != "hood-base-pick":
        return False, "next_stage_not_hood_pick", audit
    if int(hood_pick["frame_count"]) < min_hood_pick_frames:
        return False, "hood_pick_too_short", audit
    if screw_index + 2 >= len(segments):
        return False, "missing_stage_after_hood_pick", audit
    if segments[screw_index + 2]["stage"] != "hood-base-place":
        return False, "hood_pick_not_followed_by_place", audit
    return True, "pass", audit

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", type=Path, nargs="+", required=True)
    parser.add_argument("--output-bank", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int, default=923048)
    parser.add_argument(
        "--lamp-bulb-fsm-pos-threshold",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help=(
            "For lamp/bulb-base-screw only, recompute the stage endpoint from "
            "three consecutive geometry matches using this FSM-only position "
            "threshold. The benchmark assembly threshold is not changed."
        ),
    )
    parser.add_argument(
        "--exclude-sha256",
        action="append",
        default=[],
        help=(
            "State SHA256 to exclude after a restore/semantic audit; repeat the "
            "flag for multiple states. Exclusions are recorded in campaign.json."
        ),
    )
    parser.add_argument(
        "--exclude-campaign",
        type=Path,
        action="append",
        default=[],
        help=(
            "Import excluded_state_sha256 from an earlier selector campaign; "
            "repeatable and recorded in the new campaign."
        ),
    )
    parser.add_argument(
        "--exclude-audit",
        type=Path,
        action="append",
        default=[],
        help=(
            "Import state_sha256 from the failures list of a restore audit; "
            "repeatable and recorded in the new campaign."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_record_is_success(record: dict) -> bool:
    assembled = record["physics"].get("runtime", {}).get("already_assembled")
    if assembled is None:
        return False
    values = np.asarray(assembled).reshape(-1)
    return bool(values.size > 0 and np.all(values.astype(bool)))


def record_assembly_pair_is_complete(record: dict, pair_index: int) -> bool:
    assembled = record["physics"].get("runtime", {}).get("already_assembled")
    if assembled is None:
        return False
    return assembly_pair_mask_value(assembled, pair_index)


def first_assembly_completion_entry(
    target: list[dict], *, task: str, stage: str
) -> dict | None:
    pair_index = stage_assembly_pair_index(task, stage)
    if pair_index is None:
        return None
    for entry in target:
        record = load_state_record(Path(entry["source_bank"]) / entry["path"])
        if record_assembly_pair_is_complete(record, pair_index):
            return entry
    return None


def lamp_bulb_relative_pose(record: dict) -> np.ndarray:
    physics = record["physics"]
    roots = np.asarray(physics["root_state"], dtype=np.float64)
    part_indices = physics["layout"]["part_actor_local_indices"]
    base = roots[int(part_indices["lamp_base"])]
    bulb = roots[int(part_indices["lamp_bulb"])]
    base_rot = Rotation.from_quat(base[3:7])
    bulb_rot = Rotation.from_quat(bulb[3:7])
    relative = np.eye(4, dtype=np.float64)
    relative[:3, :3] = (base_rot.inv() * bulb_rot).as_matrix()
    relative[:3, 3] = base_rot.inv().apply(bulb[:3] - base[:3])
    return relative


def lamp_bulb_fsm_completion_entry(
    episode_entries: list[dict],
    *,
    threshold: tuple[float, float, float],
    confirm_frames: int = 3,
) -> dict | None:
    """Return the frame that completes an independent bulb FSM geometry gate."""
    if len(threshold) != 3 or any(value <= 0 for value in threshold):
        raise ValueError("lamp bulb FSM threshold must contain three positive values")
    if confirm_frames < 1:
        raise ValueError("confirm_frames must be positive")

    from furniture_bench.furniture import furniture_factory

    furniture = furniture_factory("lamp")
    furniture.assembled_pos_threshold = list(threshold)
    target_poses = furniture.assembled_rel_poses[(0, 1)]
    consecutive = 0
    for entry in sorted(episode_entries, key=lambda item: int(item["frame_index"])):
        record = load_state_record(Path(entry["source_bank"]) / entry["path"])
        assembled_now = furniture.assembled(
            lamp_bulb_relative_pose(record), target_poses, pair=(0, 1)
        )
        consecutive = consecutive + 1 if assembled_now else 0
        if consecutive >= confirm_frames:
            return entry
    return None


def main() -> int:
    args = parse_args()
    excluded_sha256 = {str(value).lower() for value in args.exclude_sha256}
    exclusion_sources: list[str] = []
    for path in args.exclude_campaign:
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        excluded_sha256.update(
            str(value).lower()
            for value in payload.get("excluded_state_sha256", ())
        )
        exclusion_sources.append(str(resolved))
    for path in args.exclude_audit:
        resolved = path.expanduser().resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        excluded_sha256.update(
            str(row["state_sha256"]).lower()
            for row in payload.get("failures", ())
        )
        exclusion_sources.append(str(resolved))
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in excluded_sha256):
        raise ValueError("--exclude-sha256 values must be 64 lowercase/uppercase hex digits")
    sources = [path.expanduser().resolve() for path in args.source_bank]
    output = args.output_bank.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    campaigns = []
    entries = []
    all_entries = []
    for source_idx, source in enumerate(sources):
        campaign = json.loads((source / "campaign.json").read_text(encoding="utf-8"))
        if campaign.get("annotation_source") != "scripted":
            raise ValueError("state bank annotation_source must be scripted")
        if int(campaign.get("stride", 0)) != 1:
            raise ValueError("normalized-progress selection requires stride=1")
        campaigns.append(campaign)
        for entry in read_jsonl(source / "manifest.jsonl"):
            item = dict(entry)
            item["source_campaign_index"] = source_idx
            item["source_bank"] = str(source)
            item["source_episode_key"] = f"{source_idx}:{int(entry['episode_index'])}"
            all_entries.append(item)
            if str(entry.get("sha256", "")).lower() in excluded_sha256:
                continue
            entries.append(item)
    tasks = {str(entry["task"]) for entry in entries}
    if len(tasks) != 1:
        raise ValueError(f"source bank must contain one task, got {sorted(tasks)}")
    task = next(iter(tasks))
    if args.stage not in INCLUDED_SKILL_STAGES.get(task, ()):
        raise ValueError(f"stage {task}/{args.stage} is not in the clean analysis plan")
    progress_labels = get_task_progress_labels(task, "skill_states")
    min_same_episode_progress_gap = SAME_EPISODE_PROGRESS_GAP_OVERRIDES.get(
        (task, args.stage), MIN_SAME_EPISODE_PROGRESS_GAP
    )
    is_final_stage = bool(progress_labels and args.stage == progress_labels[-1])
    lamp_fsm_threshold = (
        None
        if args.lamp_bulb_fsm_pos_threshold is None
        else tuple(float(value) for value in args.lamp_bulb_fsm_pos_threshold)
    )
    if lamp_fsm_threshold is not None and (task, args.stage) != (
        "lamp",
        "bulb-base-screw",
    ):
        raise ValueError(
            "--lamp-bulb-fsm-pos-threshold is only valid for lamp/bulb-base-screw"
        )

    by_episode: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_episode[str(entry["source_episode_key"])].append(entry)
    for episode_entries in by_episode.values():
        episode_entries.sort(key=lambda item: int(item["frame_index"]))
    all_entries_by_episode: dict[str, list[dict]] = defaultdict(list)
    for entry in all_entries:
        all_entries_by_episode[str(entry["source_episode_key"])].append(entry)
    for episode_entries in all_entries_by_episode.values():
        episode_entries.sort(key=lambda item: int(item["frame_index"]))

    eligible: dict[str, list[dict]] = {}
    endpoint_modes: dict[str, str] = {}
    episode_gate_rejections: Counter[str] = Counter()
    episode_gate_pass_count = 0
    stage_index = progress_labels.index(args.stage)
    for episode, episode_entries in sorted(by_episode.items()):
        fsm_completion = None
        if (task, args.stage) == ("lamp", "bulb-base-screw") and lamp_fsm_threshold:
            fsm_completion = lamp_bulb_fsm_completion_entry(
                all_entries_by_episode[episode], threshold=lamp_fsm_threshold
            )
            if fsm_completion is None:
                episode_gate_rejections["missing_fsm_geometry_completion"] += 1
                continue
            episode_gate_pass_count += 1
        elif (task, args.stage) == ("lamp", "bulb-base-screw"):
            gate_pass, gate_reason, _gate_audit = lamp_bulb_screw_episode_gate(
                all_entries_by_episode[episode]
            )
            if not gate_pass:
                episode_gate_rejections[gate_reason] += 1
                continue
            episode_gate_pass_count += 1
        visits: dict[int, list[dict]] = defaultdict(list)
        for entry in episode_entries:
            if entry.get("skill_state") == args.stage:
                visits[int(entry.get("skill_visit_index", 0))].append(entry)
        valid_visit = None
        for visit_idx in sorted(visits):
            target = visits[visit_idx]
            if fsm_completion is not None:
                start_frame = int(target[0]["frame_index"])
                completion_frame = int(fsm_completion["frame_index"])
                if completion_frame < start_frame:
                    continue
                fsm_target = [
                    entry
                    for entry in target
                    if int(entry["frame_index"]) <= completion_frame
                ]
                stage_length = completion_frame - start_frame + 1
                if stage_length > 1 and fsm_target:
                    valid_visit = (
                        fsm_target,
                        stage_length,
                        "lamp_bulb_fsm_geometry_3frame",
                    )
                    break
                continue
            assembly_completion = first_assembly_completion_entry(
                target, task=task, stage=args.stage
            )
            if assembly_completion is not None:
                stage_length = int(assembly_completion["frame_index"]) - int(
                    target[0]["frame_index"]
                ) + 1
                # If the pair was complete before this label appeared, there
                # is no remaining stage execution to analyze.
                if stage_length > 1:
                    pair_index = ASSEMBLY_COMPLETION_STAGE_PAIR_INDEX[task][args.stage]
                    valid_visit = (
                        [
                            entry
                            for entry in target
                            if int(entry["frame_index"])
                            <= int(assembly_completion["frame_index"])
                        ],
                        stage_length,
                        f"assembly_mask_pair_{pair_index}",
                    )
                    break
                # The assembly bit was already set when this FSM label first
                # appeared.  There is no remaining current-stage behavior to
                # evaluate, so this visit must not fall through to the generic
                # transition or final-episode-success endpoints below.
                continue
            transition = next(
                (
                    entry
                    for entry in episode_entries
                    if int(entry["frame_index"]) > int(target[-1]["frame_index"])
                    and entry.get("skill_state") != args.stage
                ),
                None,
            )
            if transition is not None:
                next_stage = transition.get("skill_state")
                if next_stage not in progress_labels:
                    continue
                if progress_labels.index(next_stage) <= stage_index:
                    # A failed grasp/place can reset the FSM to an earlier
                    # stage.  That is not a successful exit from this visit.
                    continue
                stage_length = int(transition["frame_index"]) - int(
                    target[0]["frame_index"]
                )
                valid_visit = (target, stage_length, "next_stage_transition")
                break
            if not is_final_stage or visit_idx != max(visits):
                continue
            last = target[-1]
            if int(last["frame_index"]) != int(episode_entries[-1]["frame_index"]):
                continue
            if not final_record_is_success(
                load_state_record(Path(last["source_bank"]) / last["path"])
            ):
                continue
            stage_length = int(last["frame_index"]) - int(target[0]["frame_index"]) + 1
            valid_visit = (target, stage_length, "episode_success_terminal")
            break

        if valid_visit is None:
            continue
        target, stage_length, endpoint_mode = valid_visit
        start_frame = int(target[0]["frame_index"])
        last_offset = stage_length - 1
        if last_offset <= 0:
            continue
        candidates = []
        for entry in target:
            offset = int(entry["frame_index"]) - start_frame
            progress = offset / last_offset
            item = dict(entry)
            item["skill_frame_offset"] = offset
            item["stage_progress"] = progress
            item["stage_length_frames"] = stage_length
            item["stage_endpoint_mode"] = endpoint_mode
            candidates.append(item)
        eligible[episode] = candidates
        endpoint_modes[episode] = endpoint_mode

    rng = random.Random(args.seed)
    selected: list[dict] = []
    selected_by_episode: dict[str, list[dict]] = defaultdict(list)
    pooled_range = POOLED_STAGE_RANGES.get((task, args.stage))
    requested_counts = (
        {"pooled": 48}
        if pooled_range is not None
        else {name: count for name, _low, _high, count in STRATA}
    )

    def separated_from_prior(episode: str, candidate: dict) -> bool:
        for prior in selected_by_episode.get(episode, ()):
            if (
                abs(int(candidate["frame_index"]) - int(prior["frame_index"]))
                < MIN_SAME_EPISODE_FRAME_GAP
                or abs(float(candidate["stage_progress"]) - float(prior["stage_progress"]))
                + 1e-12 < min_same_episode_progress_gap
            ):
                return False
        return True

    def candidates_for(episode: str, low: float, high: float) -> list[dict]:
        if len(selected_by_episode.get(episode, ())) >= MAX_STATES_PER_SOURCE_EPISODE:
            return []
        prior_hashes = {
            item["sha256"] for item in selected_by_episode.get(episode, ())
        }
        return [
            item
            for item in eligible[episode]
            if low <= float(item["stage_progress"]) <= high
            and item["sha256"] not in prior_hashes
            and separated_from_prior(episode, item)
        ]

    def select_one(
        *,
        name: str,
        low: float,
        high: float,
        target_progress: float,
        selection_mode: str,
    ) -> bool:
        possible = {
            episode: candidates
            for episode in eligible
            if (candidates := candidates_for(episode, low, high))
        }
        if not possible:
            return False
        # Episode-balanced round robin: exhaust episodes with fewer selected
        # states before taking another state from an already represented one.
        min_count = min(
            len(selected_by_episode.get(episode, ())) for episode in possible
        )
        possible = {
            episode: candidates
            for episode, candidates in possible.items()
            if len(selected_by_episode.get(episode, ())) == min_count
        }
        ranked = []
        for episode, candidates in possible.items():
            candidate = min(
                candidates,
                key=lambda item: (
                    abs(float(item["stage_progress"]) - target_progress),
                    int(item["frame_index"]),
                ),
            )
            ranked.append(
                (
                    abs(float(candidate["stage_progress"]) - target_progress),
                    rng.random(),
                    episode,
                    candidate,
                )
            )
        _distance, _tie, episode, candidate = min(ranked, key=lambda row: row[:2])
        item = dict(candidate)
        item["selection_stratum"] = name
        item["selection_target_progress"] = target_progress
        item["selection_mode"] = selection_mode
        item["source_episode_selection_ordinal"] = len(
            selected_by_episode.get(episode, ())
        ) + 1
        selected.append(item)
        selected_by_episode[episode].append(item)
        return True

    if pooled_range is not None:
        low, high = pooled_range
        targets = [low + (index + 0.5) * (high - low) / 48 for index in range(48)]
        rng.shuffle(targets)
        for target_progress in targets:
            if not select_one(
                name="pooled",
                low=low,
                high=high,
                target_progress=target_progress,
                selection_mode="pooled_episode_balanced",
            ):
                raise RuntimeError(
                    "Pooled selection cannot make 48 states under the two-state "
                    f"episode cap and separation gates; total eligible={len(eligible)}, "
                    f"selected={len(selected)}"
                )
    else:
        for name, low, high, count in STRATA:
            for _index in range(count):
                if not select_one(
                    name=name,
                    low=low,
                    high=high,
                    target_progress=rng.uniform(low, high),
                    selection_mode="requested_stratum",
                ):
                    if name == "early":
                        raise RuntimeError(
                            f"Need {count} {name} states under the two-state episode "
                            f"cap and separation gates, selected {len(selected)} total; "
                            f"eligible episodes={len(eligible)}"
                        )
                    break

        total_requested = sum(requested_counts.values())
        shortfall = total_requested - len(selected)
        if shortfall > 0:
            early_low, early_high, _early_count = next(
                (low, high, count)
                for name, low, high, count in STRATA
                if name == "early"
            )
            for _index in range(shortfall):
                if not select_one(
                    name="early",
                    low=early_low,
                    high=early_high,
                    target_progress=rng.uniform(early_low, early_high),
                    selection_mode="later_quota_early_fallback",
                ):
                    raise RuntimeError(
                        "Later-stratum fallback cannot make 48 states under the "
                        "two-state episode cap and separation gates; "
                        f"need {shortfall}, selected={len(selected)}, "
                        f"eligible episodes={len(eligible)}"
                    )

    total_requested = sum(requested_counts.values())
    if len(selected) != total_requested:
        raise AssertionError(
            f"selector produced {len(selected)} states, expected {total_requested}"
        )

    output.mkdir(parents=True)
    (output / "states").mkdir()
    output_entries = []
    stratum_order = (
        {"pooled": 0}
        if pooled_range is not None
        else {name: idx for idx, (name, *_rest) in enumerate(STRATA)}
    )
    for item in sorted(
        selected,
        key=lambda entry: (
            stratum_order[entry["selection_stratum"]],
            entry["stage_progress"],
        ),
    ):
        source_state = Path(item["source_bank"]) / item["path"]
        destination = (
            output
            / "states"
            / f"c{int(item['source_campaign_index']):02d}__{source_state.name}"
        )
        shutil.copy2(source_state, destination)
        copied_hash = sha256(destination)
        if copied_hash != item["sha256"]:
            raise RuntimeError(f"hash mismatch after copying {source_state}")
        output_item = dict(item)
        output_item["source_path"] = str(source_state)
        output_item["path"] = str(destination.relative_to(output))
        output_entries.append(output_item)

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for item in output_entries:
            stream.write(json.dumps(item, sort_keys=True) + "\n")

    progress_values = np.asarray(
        [float(item["stage_progress"]) for item in output_entries],
        dtype=np.float64,
    )
    progress_histogram, progress_edges = np.histogram(
        progress_values, bins=np.linspace(0.0, 1.0, 11)
    )
    progress_summary = {
        "min": float(progress_values.min()),
        "q25": float(np.percentile(progress_values, 25)),
        "median": float(np.median(progress_values)),
        "q75": float(np.percentile(progress_values, 75)),
        "max": float(progress_values.max()),
    }
    selection_campaign = {
        "schema": campaigns[0].get("schema"),
        "annotation_source": "scripted",
        "task": task,
        "stage": args.stage,
        "source_banks": [str(source) for source in sources],
        "source_campaigns": campaigns,
        "selection_seed": args.seed,
        "excluded_state_sha256": sorted(excluded_sha256),
        "exclusion_sources": exclusion_sources,
        "progress_definition": "skill_frame_offset / (stage_length_frames - 1)",
        "selection_design": (
            "pooled_episode_balanced" if pooled_range is not None else "stratified"
        ),
        "strata": (
            [{"name": "pooled", "low": pooled_range[0], "high": pooled_range[1], "count": 48}]
            if pooled_range is not None
            else [
                {"name": name, "low": low, "high": high, "count": count}
                for name, low, high, count in STRATA
            ]
        ),
        "max_states_per_source_episode": MAX_STATES_PER_SOURCE_EPISODE,
        "min_same_episode_progress_gap": min_same_episode_progress_gap,
        "min_same_episode_frame_gap": MIN_SAME_EPISODE_FRAME_GAP,
        "later_strata_fallback_policy": (
            None
            if pooled_range is not None
            else "fill_missing_middle_or_late_slots_with_episode_balanced_early_states"
        ),
        "later_strata_fallback_used": any(
            item.get("selection_mode") == "later_quota_early_fallback"
            for item in output_entries
        ),
        "requested_stratum_counts": requested_counts,
        "realized_stratum_counts": dict(
            Counter(item["selection_stratum"] for item in output_entries)
        ),
        "selected_count": len(output_entries),
        "eligible_distinct_episodes": len(eligible),
        "selected_distinct_episodes": len(
            {item["source_episode_key"] for item in output_entries}
        ),
        "max_realized_states_per_source_episode": max(
            Counter(
                item["source_episode_key"] for item in output_entries
            ).values(),
            default=0,
        ),
        "progress_summary": progress_summary,
        "endpoint_modes": sorted(set(endpoint_modes.values())),
        "episode_gate": (
            {
                "name": (
                    "lamp_bulb_fsm_geometry_completion"
                    if lamp_fsm_threshold is not None
                    else "lamp_bulb_screw_normal_hood_transition"
                ),
                "fsm_position_threshold": (
                    list(lamp_fsm_threshold)
                    if lamp_fsm_threshold is not None
                    else None
                ),
                "fsm_confirm_frames": 3 if lamp_fsm_threshold is not None else None,
                "benchmark_threshold_unchanged": lamp_fsm_threshold is not None,
                "required_transition": (
                    None
                    if lamp_fsm_threshold is not None
                    else [
                        "bulb-base-screw",
                        "hood-base-pick",
                        "hood-base-place",
                    ]
                ),
                "min_hood_pick_frames": (
                    None
                    if lamp_fsm_threshold is not None
                    else LAMP_BULB_SCREW_MIN_HOOD_PICK_FRAMES
                ),
                "pass_count": episode_gate_pass_count,
                "rejection_counts": dict(episode_gate_rejections),
            }
            if (task, args.stage) == ("lamp", "bulb-base-screw")
            else None
        ),
    }
    (output / "campaign.json").write_text(
        json.dumps(selection_campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    distribution = {
        "count": len(output_entries),
        "distinct_episodes": len(
            {item["source_episode_key"] for item in output_entries}
        ),
        "max_states_per_episode": max(
            Counter(item["source_episode_key"] for item in output_entries).values(),
            default=0,
        ),
        "episode_state_count_histogram": dict(
            Counter(
                Counter(
                    item["source_episode_key"] for item in output_entries
                ).values()
            )
        ),
        "by_stratum": {
            name: sum(
                item["selection_stratum"] == name for item in output_entries
            )
            for name in (
                ("pooled",)
                if pooled_range is not None
                else tuple(name for name, *_rest in STRATA)
            )
        },
        "progress": progress_values.tolist(),
        "progress_summary": progress_summary,
        "progress_histogram": {
            "edges": progress_edges.tolist(),
            "counts": progress_histogram.astype(int).tolist(),
        },
        "stage_length_frames": [item["stage_length_frames"] for item in output_entries],
        "endpoint_mode_counts": dict(
            Counter(item["stage_endpoint_mode"] for item in output_entries)
        ),
    }
    (output / "distribution.json").write_text(
        json.dumps(distribution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(distribution, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
