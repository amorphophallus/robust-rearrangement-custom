#!/usr/bin/env python3
"""Validate and report the complete clean FurnitureBench skill-level matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from src.eval.skill_level import INCLUDED_SKILL_STAGES


PAIRED_COMPARISONS = (
    ("rgbd_gp", "rgbd", "GP - RGB-D"),
    ("rgbd_colored_gp", "rgbd_gp", "TAGPoint - GP"),
    ("rgbd_gp_skill", "rgbd_skill", "GP+skill - skill"),
    ("rgbd_grasp_part", "rgbd_gp", "grasp - GP"),
    (
        "rgbd_grasp_part_colored",
        "rgbd_grasp_part",
        "colored grasp - grasp",
    ),
)

REPEAT_SEEDS = [923001, 923002, 923003]
SKILL_COMPLETION_AUTHORITY = "scripted_fsm_forward_transition"
LAMP_BULB_FSM_POSITION_THRESHOLD = [0.01, 0.005, 0.01]

EXPECTED_POLICY_CONDITIONS = {
    "rgb": {
        "requires_skill_input": False,
        "guidance_point": False,
        "guidance_point_colored": False,
        "grasp": False,
        "grasp_colored": False,
        "grasp_part": False,
    },
    "rgbd": {
        "requires_skill_input": False,
        "guidance_point": False,
        "guidance_point_colored": False,
        "grasp": False,
        "grasp_colored": False,
        "grasp_part": False,
    },
    "rgbd_skill": {
        "requires_skill_input": True,
        "guidance_point": False,
        "guidance_point_colored": False,
        "grasp": False,
        "grasp_colored": False,
        "grasp_part": False,
    },
    "rgbd_gp": {
        "requires_skill_input": False,
        "guidance_point": True,
        "guidance_point_colored": False,
        "grasp": False,
        "grasp_colored": False,
        "grasp_part": False,
    },
    "rgbd_colored_gp": {
        "requires_skill_input": False,
        "guidance_point": True,
        "guidance_point_colored": True,
        "grasp": False,
        "grasp_colored": False,
        "grasp_part": False,
    },
    "rgbd_gp_skill": {
        "requires_skill_input": True,
        "guidance_point": True,
        "guidance_point_colored": False,
        "grasp": False,
        "grasp_colored": False,
        "grasp_part": False,
    },
    "rgbd_grasp_part": {
        "requires_skill_input": False,
        "guidance_point": False,
        "guidance_point_colored": False,
        "grasp": False,
        "grasp_colored": False,
        "grasp_part": True,
    },
    "rgbd_grasp_part_colored": {
        "requires_skill_input": False,
        "guidance_point": False,
        "guidance_point_colored": True,
        "grasp": False,
        "grasp_colored": True,
        "grasp_part": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("reports/clean_skill_level_checkpoint_registry_20260923.json"),
    )
    parser.add_argument("--state-banks-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-results-root",
        type=Path,
        default=None,
        help=(
            "Optional full-plan root supplying only RGB/RGB-D baseline cells. "
            "Used for an explicitly validated corrected-lineage merge."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=923999)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def mean_sd(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n_train": int(array.size),
    }


def metric_from_summary(summary: dict, key: str) -> float | None:
    value = summary.get(key, {}).get("mean")
    return None if value is None else float(value)


def describe_records(records: list[dict], field: str, *, scale: float = 1.0) -> dict:
    values = np.asarray(
        [float(row[field]) * scale for row in records if row.get(field) is not None],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None, "p90": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def aggregate_cell(checkpoint_rows: list[dict], records: list[dict]) -> dict:
    summaries = [row["summary"] for row in checkpoint_rows]
    output = {
        "n_train": len(summaries),
        "success_rate": mean_sd([summary["success_rate"] for summary in summaries]),
        "success_counts": {
            "completed": sum(int(summary["completed"]) for summary in summaries),
            "attempted": sum(int(summary["attempted"]) for summary in summaries),
        },
        "e_gt_cm": mean_sd([metric_from_summary(summary, "e_gt_cm") for summary in summaries]),
        "te_position_cm": mean_sd(
            [metric_from_summary(summary, "te_position_cm") for summary in summaries]
        ),
        "per_checkpoint": checkpoint_rows,
        "pooled_descriptive": {
            "e_gt_cm": describe_records(records, "e_gt_m", scale=100.0),
            "te_position_cm": describe_records(
                records, "te_position_m", scale=100.0
            ),
            "te_orientation_deg": describe_records(records, "te_orientation_deg"),
            "te_normalized_total": describe_records(
                records, "te_normalized_total"
            ),
        },
    }
    orientation = [metric_from_summary(summary, "te_orientation_deg") for summary in summaries]
    total = [metric_from_summary(summary, "te_normalized_total") for summary in summaries]
    if all(value is not None for value in orientation):
        output["te_orientation_deg"] = mean_sd(orientation)
    if all(value is not None for value in total):
        output["te_normalized_total"] = mean_sd(total)
    return output


def state_means(records: list[dict], field: str) -> dict[str, float]:
    grouped = {}
    for row in records:
        value = row.get(field)
        if field == "completed_current_stage":
            value = float(bool(value))
        if value is None:
            continue
        grouped.setdefault(str(row["state_sha256"]), []).append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def validate_and_compact_records(
    records: list[dict],
    cell_dir: Path,
    *,
    policy_receives_guidance: bool,
) -> list[dict]:
    compact = []
    trajectory_fields = (
        "ee_position_trajectory_robot_base_m",
        "ee_quaternion_trajectory_xyzw",
        "policy_input_clean_target_pose_trajectory_robot_base",
        "clean_target_pose_trajectory_robot_base",
    )
    statistic_fields = (
        "state_sha256",
        "repeat_index",
        "completed_current_stage",
        "e_gt_m",
        "te_position_m",
        "te_orientation_deg",
        "te_normalized_total",
    )
    for row in records:
        expected_steps = int(row["completion_steps"])
        lengths = {
            field: len(row.get(field, ())) for field in trajectory_fields
        }
        if any(length != expected_steps for length in lengths.values()):
            raise ValueError(
                f"trajectory length mismatch in {cell_dir}: "
                f"steps={expected_steps}, lengths={lengths}"
            )
        policy_targets = row["policy_input_clean_target_pose_trajectory_robot_base"]
        evaluation_targets = row["clean_target_pose_trajectory_robot_base"]
        if policy_receives_guidance:
            if any(target is None for target in policy_targets):
                raise ValueError(f"guided policy has a null input target in {cell_dir}")
        elif any(target is not None for target in policy_targets):
            raise ValueError(f"baseline policy unexpectedly received guidance in {cell_dir}")
        if any(target is None for target in evaluation_targets):
            raise ValueError(f"clean evaluation target trajectory is incomplete in {cell_dir}")
        if row.get("e_gt_m") is None or row.get("te_position_m") is None:
            raise ValueError(f"clean position metric is missing in {cell_dir}")
        if not math.isclose(
            float(row["e_gt_m"]),
            float(row["te_position_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"clean E_GT and TE.position diverge in {cell_dir}")
        compact.append({field: row.get(field) for field in statistic_fields})
    return compact


def paired_episode_cluster_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    episode_by_state: dict[str, str],
    samples: int,
    seed: int,
) -> dict:
    keys = sorted(set(left) & set(right))
    if len(keys) != 48:
        raise ValueError(f"paired comparison expected 48 states, found {len(keys)}")
    missing_episode_keys = sorted(set(keys) - set(episode_by_state))
    if missing_episode_keys:
        raise ValueError(
            f"missing source episode for {len(missing_episode_keys)} paired states"
        )
    differences_by_episode: dict[str, list[float]] = {}
    for key in keys:
        differences_by_episode.setdefault(str(episode_by_state[key]), []).append(
            float(left[key] - right[key])
        )
    episode_keys = sorted(differences_by_episode)
    cluster_sums = np.asarray(
        [sum(differences_by_episode[key]) for key in episode_keys],
        dtype=np.float64,
    )
    cluster_sizes = np.asarray(
        [len(differences_by_episode[key]) for key in episode_keys],
        dtype=np.float64,
    )
    differences = np.asarray(
        [left[key] - right[key] for key in keys], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    chunk = 1_000
    estimates = []
    remaining = int(samples)
    while remaining > 0:
        count = min(chunk, remaining)
        indices = rng.integers(
            0, len(episode_keys), size=(count, len(episode_keys))
        )
        estimates.append(
            cluster_sums[indices].sum(axis=1)
            / cluster_sizes[indices].sum(axis=1)
        )
        remaining -= count
    boot = np.concatenate(estimates)
    return {
        "paired_state_count": len(keys),
        "paired_episode_count": len(episode_keys),
        "max_states_per_episode": int(cluster_sizes.max()),
        "bootstrap_unit": "source_episode_cluster",
        "estimate": float(differences.mean()),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
    }


def format_mean_sd(stats: dict, *, scale: float = 1.0, digits: int = 2) -> str:
    return (
        f"{stats['mean'] * scale:.{digits}f} ± "
        f"{stats['sample_sd'] * scale:.{digits}f}"
    )


def markdown_table(title: str, task: str, stages: tuple[str, ...], conditions, cells, metric: str) -> list[str]:
    lines = [f"### {title}: `{task}`", ""]
    lines.append("| Condition | n_train | " + " | ".join(stages) + " |")
    lines.append("|---|---:|" + "---:|" * len(stages))
    for condition in conditions:
        values = []
        for stage in stages:
            cell = cells[condition["id"]][task][stage]
            if metric == "success_rate":
                counts = cell["success_counts"]
                values.append(
                    format_mean_sd(cell[metric], scale=100.0, digits=1)
                    + f"% ({counts['completed']}/{counts['attempted']})"
                )
            else:
                values.append(format_mean_sd(cell[metric], digits=2))
        n_train = cells[condition["id"]][task][stages[0]]["n_train"]
        lines.append(
            f"| {condition['label']} | {n_train} | " + " | ".join(values) + " |"
        )
    lines.append("")
    return lines


def markdown_optional_table(
    title: str,
    task: str,
    stages: tuple[str, ...],
    conditions,
    cells,
    metric: str,
) -> list[str]:
    lines = [f"### {title}: `{task}`", ""]
    lines.append("| Condition | n_train | " + " | ".join(stages) + " |")
    lines.append("|---|---:|" + "---:|" * len(stages))
    for condition in conditions:
        values = []
        for stage in stages:
            value = cells[condition["id"]][task][stage].get(metric)
            values.append("—" if value is None else format_mean_sd(value, digits=2))
        n_train = cells[condition["id"]][task][stages[0]]["n_train"]
        lines.append(
            f"| {condition['label']} | {n_train} | " + " | ".join(values) + " |"
        )
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    registry = read_json(args.registry.resolve())
    results_root = args.results_root.resolve()
    baseline_results_root = (
        args.baseline_results_root.resolve()
        if args.baseline_results_root is not None
        else results_root
    )
    corrected_lineage_merge = baseline_results_root != results_root
    banks_root = args.state_banks_root.resolve()
    checkpoint_cells: dict = {}
    all_records: dict = {}
    bank_audit: dict = {}
    state_episode_keys: dict = {}
    runtime_fingerprints: set[str] = set()
    furniture_bench_fingerprints: set[str] = set()
    root_commits: set[str] = set()
    checkpoint_registry = {}
    for condition in registry["conditions"]:
        condition_id = str(condition["id"])
        if condition_id not in EXPECTED_POLICY_CONDITIONS:
            raise ValueError(f"unregistered policy contract: {condition_id}")
        for checkpoint in condition["checkpoints"]:
            seed = str(checkpoint["train_seed"])
            path = Path(checkpoint["path"]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint_registry[(condition_id, seed)] = {
                "path": str(path),
                "sha256": sha256(path),
            }
    matrix_run = read_json(results_root / "matrix_run.json")
    baseline_matrix_run = read_json(baseline_results_root / "matrix_run.json")
    full_filters = {
        "condition": [],
        "train_seed": [],
        "task": [],
        "stage": [],
        "max_cells": None,
    }
    for label, run, expected_cells, expected_filters in (
        (
            "baseline",
            baseline_matrix_run,
            432,
            full_filters,
        ),
        (
            "corrected",
            matrix_run,
            324 if corrected_lineage_merge else 432,
            (
                {
                    **full_filters,
                    "condition": [
                        "rgbd_skill",
                        "rgbd_gp",
                        "rgbd_colored_gp",
                        "rgbd_gp_skill",
                        "rgbd_grasp_part",
                        "rgbd_grasp_part_colored",
                    ],
                }
                if corrected_lineage_merge
                else full_filters
            ),
        ),
    ):
        if (
            run.get("schema") != "rr-clean-skill-level-matrix-run-v1"
            or int(run.get("cell_count", -1)) != expected_cells
            or int(run.get("repeats", -1)) != 3
            or run.get("repeat_seeds") != REPEAT_SEEDS
            or run.get("filters") != expected_filters
        ):
            raise ValueError(f"{label} matrix_run.json contract is invalid")
    registry_source_key = "reports/clean_skill_level_checkpoint_registry_20260923.json"
    baseline_registry_source_hash = baseline_matrix_run.get(
        "runtime_source_sha256", {}
    ).get(registry_source_key)
    corrected_registry_source_hash = matrix_run.get("runtime_source_sha256", {}).get(
        registry_source_key
    )
    if not baseline_registry_source_hash or not corrected_registry_source_hash:
        raise ValueError("matrix contract lacks registry source fingerprint")
    registered_checkpoint_hashes = {
        value["path"]: value["sha256"] for value in checkpoint_registry.values()
    }
    for label, run in (("baseline", baseline_matrix_run), ("corrected", matrix_run)):
        contract_hashes = run.get("checkpoint_sha256", {})
        if any(
            registered_checkpoint_hashes.get(path) != digest
            for path, digest in contract_hashes.items()
        ):
            raise ValueError(
                f"{label} matrix contract checkpoint hashes differ from registry files"
            )

    for condition in registry["conditions"]:
        condition_id = condition["id"]
        checkpoint_cells[condition_id] = {}
        all_records[condition_id] = {}
        for task, stages in INCLUDED_SKILL_STAGES.items():
            checkpoint_cells[condition_id][task] = {}
            all_records[condition_id][task] = {}
            for stage in stages:
                bank_manifest = read_jsonl(banks_root / task / stage / "manifest.jsonl")
                if len(bank_manifest) != 48:
                    raise ValueError(f"state bank {task}/{stage} has {len(bank_manifest)} states")
                expected_hashes = {row["sha256"] for row in bank_manifest}
                stage_episode_by_state = {
                    str(row["sha256"]): str(row["source_episode_key"])
                    for row in bank_manifest
                }
                if len(stage_episode_by_state) != 48:
                    raise ValueError(
                        f"state bank {task}/{stage} has duplicate state hashes"
                    )
                prior_episode_map = state_episode_keys.setdefault(task, {}).setdefault(
                    stage, stage_episode_by_state
                )
                if prior_episode_map != stage_episode_by_state:
                    raise ValueError(
                        f"state bank {task}/{stage} changed while reporting"
                    )
                expected_manifest_sha256 = sha256(
                    banks_root / task / stage / "manifest.jsonl"
                )
                bank_campaign = read_json(banks_root / task / stage / "campaign.json")
                bank_audit.setdefault(task, {})[stage] = {
                    "requested": 48,
                    "valid": len(bank_manifest),
                    "early": sum(row.get("selection_stratum") == "early" for row in bank_manifest),
                    "middle": sum(row.get("selection_stratum") == "middle" for row in bank_manifest),
                    "late": sum(row.get("selection_stratum") == "late" for row in bank_manifest),
                    "pooled": sum(row.get("selection_stratum") == "pooled" for row in bank_manifest),
                    "distinct_source_episodes": len(
                        set(stage_episode_by_state.values())
                    ),
                    "max_states_per_source_episode": max(
                        (
                            list(stage_episode_by_state.values()).count(episode)
                            for episode in set(stage_episode_by_state.values())
                        ),
                        default=0,
                    ),
                    "selection_design": bank_campaign.get(
                        "selection_design", "stratified"
                    ),
                    "progress_summary": bank_campaign.get("progress_summary"),
                    "sampling_fallback_used": bool(
                        bank_campaign.get("later_strata_fallback_used", False)
                    ),
                }
                checkpoint_rows = []
                combined = []
                for checkpoint in condition["checkpoints"]:
                    seed = checkpoint["train_seed"]
                    cell_results_root = (
                        baseline_results_root
                        if condition_id in {"rgb", "rgbd"}
                        else results_root
                    )
                    cell_dir = cell_results_root / condition_id / seed / task / stage
                    run = read_json(cell_dir / "run.json")
                    summary = read_json(cell_dir / "summary.json")
                    records = read_jsonl(cell_dir / "records.jsonl")
                    registered_checkpoint = checkpoint_registry[(condition_id, str(seed))]
                    expected_policy_condition = EXPECTED_POLICY_CONDITIONS[condition_id]
                    expected_metric_type = (
                        "pose"
                        if expected_policy_condition["grasp"]
                        or expected_policy_condition["grasp_part"]
                        else "position"
                    )
                    if (
                        run.get("annotation_source") != "scripted"
                        or run.get("skill_completion_authority")
                        != SKILL_COMPLETION_AUTHORITY
                        or run.get("lamp_bulb_fsm_position_threshold")
                        != LAMP_BULB_FSM_POSITION_THRESHOLD
                        or run.get("condition") != condition_id
                        or run.get("task") != task
                        or run.get("skill_stage") != stage
                        or int(run.get("state_count", -1)) != 48
                        or int(run.get("repeats", -1)) != 3
                        or run.get("repeat_seeds") != REPEAT_SEEDS
                        or str(Path(run.get("checkpoint", "")).resolve())
                        != registered_checkpoint["path"]
                        or run.get("checkpoint_sha256")
                        != registered_checkpoint["sha256"]
                        or run.get("state_manifest_sha256")
                        != expected_manifest_sha256
                        or run.get("policy_condition") != expected_policy_condition
                        or run.get("metric_type") != expected_metric_type
                        or bool(run.get("depth_positive_meters"))
                        != (condition_id != "rgb")
                    ):
                        raise ValueError(f"run contract mismatch: {cell_dir}")
                    runtime_source = run.get("runtime_source_sha256")
                    if not isinstance(runtime_source, dict):
                        raise ValueError(f"runtime source fingerprint missing: {cell_dir}")
                    registry_source_hash = runtime_source.get(
                        registry_source_key
                    )
                    if not registry_source_hash:
                        raise ValueError(f"registry source fingerprint missing: {cell_dir}")
                    expected_registry_source_hash = (
                        baseline_registry_source_hash
                        if condition_id in {"rgb", "rgbd"}
                        else corrected_registry_source_hash
                    )
                    if registry_source_hash != expected_registry_source_hash:
                        raise ValueError(
                            f"unexpected registry lineage for {condition_id}: {cell_dir}"
                        )
                    normalized_runtime_source = dict(runtime_source)
                    normalized_runtime_source.pop(registry_source_key)
                    runtime_fingerprints.add(
                        json.dumps(normalized_runtime_source, sort_keys=True)
                    )
                    furniture_bench_fingerprints.add(
                        json.dumps(
                            run.get("furniture_bench_repository_state"),
                            sort_keys=True,
                        )
                    )
                    root_commits.add(str(run.get("git_commit")))
                    if (
                        not summary.get("complete")
                        or int(summary["attempted"]) != 144
                        or int(summary.get("expected_attempts", -1)) != 144
                        or int(summary.get("distinct_states", -1)) != 48
                        or summary.get("checkpoint_sha256")
                        != registered_checkpoint["sha256"]
                    ):
                        raise ValueError(f"incomplete result: {cell_dir}")
                    for metric in ("e_gt_cm", "te_position_cm"):
                        if int(summary.get(metric, {}).get("count", 0)) != 144:
                            raise ValueError(
                                f"incomplete {metric} metric in {cell_dir}"
                            )
                    if condition_id in {
                        "rgbd_grasp_part",
                        "rgbd_grasp_part_colored",
                    }:
                        for metric in (
                            "te_orientation_deg",
                            "te_normalized_total",
                        ):
                            if int(summary.get(metric, {}).get("count", 0)) != 144:
                                raise ValueError(
                                    f"incomplete {metric} metric in {cell_dir}"
                                )
                    keys = {
                        (str(row["state_sha256"]), int(row["repeat_index"]))
                        for row in records
                    }
                    if len(records) != 144 or len(keys) != 144:
                        raise ValueError(f"duplicate/missing records: {cell_dir}")
                    if {row["state_sha256"] for row in records} != expected_hashes:
                        raise ValueError(f"state-set mismatch: {cell_dir}")
                    if any(
                        row.get("condition") != condition_id
                        or row.get("checkpoint_sha256")
                        != registered_checkpoint["sha256"]
                        or int(row.get("repeat_index", -1)) not in (0, 1, 2)
                        or int(row.get("repeat_seed", -1))
                        != REPEAT_SEEDS[int(row.get("repeat_index", -1))]
                        for row in records
                    ):
                        raise ValueError(f"record provenance mismatch: {cell_dir}")
                    if int(summary.get("restore_audit", {}).get("gate_pass_count", 0)) != 144:
                        raise ValueError(f"restore gate incomplete: {cell_dir}")
                    checkpoint_rows.append(
                        {
                            "train_seed": seed,
                            "checkpoint": checkpoint["path"],
                            "summary": summary,
                        }
                    )
                    combined.extend(
                        validate_and_compact_records(
                            records,
                            cell_dir,
                            policy_receives_guidance=any(
                                expected_policy_condition[key]
                                for key in (
                                    "guidance_point",
                                    "guidance_point_colored",
                                    "grasp",
                                    "grasp_colored",
                                    "grasp_part",
                                )
                            ),
                        )
                    )
                checkpoint_cells[condition_id][task][stage] = checkpoint_rows
                all_records[condition_id][task][stage] = combined

    if len(runtime_fingerprints) != 1:
        raise ValueError("matrix mixes different runtime source fingerprints")
    if len(furniture_bench_fingerprints) != 1:
        raise ValueError("matrix mixes different FurnitureBench source states")
    if len(root_commits) != 1:
        raise ValueError("matrix mixes different root git commits")
    runtime_fingerprint = json.loads(next(iter(runtime_fingerprints)))
    furniture_bench_fingerprint = json.loads(
        next(iter(furniture_bench_fingerprints))
    )
    root_commit = next(iter(root_commits))
    def normalized_runtime(run: dict) -> dict:
        value = dict(run.get("runtime_source_sha256", {}))
        value.pop(registry_source_key, None)
        return value

    if normalized_runtime(matrix_run) != runtime_fingerprint:
        raise ValueError("corrected cell runtime hashes differ from matrix contract")
    if normalized_runtime(baseline_matrix_run) != runtime_fingerprint:
        raise ValueError("baseline cell runtime hashes differ from matrix contract")
    if matrix_run.get("furniture_bench_repository_state") != furniture_bench_fingerprint:
        raise ValueError("cell FurnitureBench state differs from matrix contract")
    if (
        baseline_matrix_run.get("furniture_bench_repository_state")
        != furniture_bench_fingerprint
    ):
        raise ValueError("baseline FurnitureBench state differs from matrix contract")
    if matrix_run.get("root_git_commit") != root_commit:
        raise ValueError("cell root commit differs from matrix contract")
    if baseline_matrix_run.get("root_git_commit") != root_commit:
        raise ValueError("baseline root commit differs from matrix contract")
    expected_bank_manifest_hashes = {
        str((banks_root / task / stage).resolve()): sha256(
            banks_root / task / stage / "manifest.jsonl"
        )
        for task, stages in INCLUDED_SKILL_STAGES.items()
        for stage in stages
    }
    if matrix_run.get("state_manifest_sha256") != expected_bank_manifest_hashes:
        raise ValueError("matrix contract state manifests differ from report inputs")
    if (
        baseline_matrix_run.get("state_manifest_sha256")
        != expected_bank_manifest_hashes
    ):
        raise ValueError("baseline state manifests differ from report inputs")

    cells = {
        condition["id"]: {
            task: {
                stage: aggregate_cell(
                    checkpoint_cells[condition["id"]][task][stage],
                    all_records[condition["id"]][task][stage],
                )
                for stage in stages
            }
            for task, stages in INCLUDED_SKILL_STAGES.items()
        }
        for condition in registry["conditions"]
    }

    comparisons = {}
    for task, stages in INCLUDED_SKILL_STAGES.items():
        comparisons[task] = {}
        for stage_idx, stage in enumerate(stages):
            comparisons[task][stage] = {}
            for comparison_idx, (left_id, right_id, label) in enumerate(PAIRED_COMPARISONS):
                left_rows = all_records[left_id][task][stage]
                right_rows = all_records[right_id][task][stage]
                comparison = {
                    "success_rate_difference": paired_episode_cluster_bootstrap(
                        state_means(left_rows, "completed_current_stage"),
                        state_means(right_rows, "completed_current_stage"),
                        episode_by_state=state_episode_keys[task][stage],
                        samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed + stage_idx * 101 + comparison_idx,
                    ),
                    "e_gt_cm_difference": paired_episode_cluster_bootstrap(
                        {key: value * 100.0 for key, value in state_means(left_rows, "e_gt_m").items()},
                        {key: value * 100.0 for key, value in state_means(right_rows, "e_gt_m").items()},
                        episode_by_state=state_episode_keys[task][stage],
                        samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed + 10_000 + stage_idx * 101 + comparison_idx,
                    ),
                }
                comparisons[task][stage][label] = comparison

    for task, stages in INCLUDED_SKILL_STAGES.items():
        for stage in stages:
            audits = [
                row["summary"]["restore_audit"]
                for condition in registry["conditions"]
                for row in checkpoint_cells[condition["id"]][task][stage]
            ]
            bank_audit[task][stage].update(
                {
                    "pinned_restore_attempts": sum(int(audit["count"]) for audit in audits),
                    "pinned_restore_pass": sum(
                        int(audit["gate_pass_count"]) for audit in audits
                    ),
                    "physics_restore_fallback_used": 0,
                    "excluded": 0,
                }
            )

    payload = {
        "schema": "rr-clean-skill-level-report-v1",
        "conditions": registry["conditions"],
        "cells": cells,
        "state_bank_restore_audit": bank_audit,
        "execution_provenance": {
            "root_git_commit": root_commit,
            "runtime_source_sha256": runtime_fingerprint,
            "registry_lineage": {
                "corrected_lineage_merge": corrected_lineage_merge,
                "baseline_results_root": str(baseline_results_root),
                "corrected_results_root": str(results_root),
                "baseline_registry_source_sha256": baseline_registry_source_hash,
                "corrected_registry_source_sha256": corrected_registry_source_hash,
                "baseline_conditions": ["rgb", "rgbd"],
                "corrected_conditions": [
                    "rgbd_skill",
                    "rgbd_gp",
                    "rgbd_colored_gp",
                    "rgbd_gp_skill",
                    "rgbd_grasp_part",
                    "rgbd_grasp_part_colored",
                ],
            },
            "furniture_bench_repository_state": furniture_bench_fingerprint,
            "matrix_run_contract": matrix_run,
            "baseline_matrix_run_contract": baseline_matrix_run,
        },
        "paired_comparisons": comparisons,
    }
    write_json_atomic(args.output_json.resolve(), payload)

    lines = [
        "# Clean skill-level analysis results",
        "",
        "## Protocol",
        "",
        "Current-stage success is defined exclusively as a forward transition of the scripted task FSM (including the final-stage transition to `done`); benchmark assembly success is retained only as a diagnostic. The lamp bulb FSM uses the fixed position threshold `[0.010, 0.005, 0.010]` m.",
        "",
        "每格为三个 training checkpoint 的均值 ± sample SD。SR 使用百分比并在括号中给出 pooled completed/attempted；E_GT/TE.position 使用 cm。失败 rollout 纳入 E_GT 与 TE。逐 checkpoint 数值保存在机器可读 JSON 的 `per_checkpoint`。",
        "",
        "## Current-stage success rate",
        "",
    ]
    for task, stages in INCLUDED_SKILL_STAGES.items():
        lines.extend(markdown_table("SR", task, stages, registry["conditions"], cells, "success_rate"))
    lines.extend(["## E_GT (cm)", ""])
    for task, stages in INCLUDED_SKILL_STAGES.items():
        lines.extend(markdown_table("E_GT", task, stages, registry["conditions"], cells, "e_gt_cm"))
    lines.extend(["## TE.position (cm)", ""])
    for task, stages in INCLUDED_SKILL_STAGES.items():
        lines.extend(markdown_table("TE.position", task, stages, registry["conditions"], cells, "te_position_cm"))
    grasp_conditions = [
        condition
        for condition in registry["conditions"]
        if condition["id"] in {"rgbd_grasp_part", "rgbd_grasp_part_colored"}
    ]
    lines.extend(["## TE.orientation (degree; grasp conditions)", ""])
    for task, stages in INCLUDED_SKILL_STAGES.items():
        lines.extend(
            markdown_optional_table(
                "TE.orientation", task, stages, grasp_conditions, cells, "te_orientation_deg"
            )
        )
    lines.extend(["## TE.normalized-total (grasp conditions)", ""])
    for task, stages in INCLUDED_SKILL_STAGES.items():
        lines.extend(
            markdown_optional_table(
                "TE.normalized-total",
                task,
                stages,
                grasp_conditions,
                cells,
                "te_normalized_total",
            )
        )
    lines.extend(
        [
            "## State-bank and restore audit",
            "",
            "| Task/stage | requested | valid | episodes | max/episode | early | middle | late | pooled | sampling fallback | pinned restore pass | alternate restore fallback | excluded |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task, stages in INCLUDED_SKILL_STAGES.items():
        for stage in stages:
            audit = bank_audit[task][stage]
            lines.append(
                f"| {task}/{stage} | {audit['requested']} | {audit['valid']} | "
                f"{audit['distinct_source_episodes']} | "
                f"{audit['max_states_per_source_episode']} | "
                f"{audit['early']} | {audit['middle']} | {audit['late']} | "
                f"{audit['pooled']} | "
                f"{int(audit['sampling_fallback_used'])} | "
                f"{audit['pinned_restore_pass']}/{audit['pinned_restore_attempts']} | "
                f"{audit['physics_restore_fallback_used']} | {audit['excluded']} |"
            )
    lines.extend(
        [
            "",
            "每个 cell 的 pooled median、IQR 与 p90 保存在机器可读 JSON 的 `pooled_descriptive` 字段。",
            "",
            "## Paired episode-cluster bootstrap",
            "",
            "预注册的五组比较按 state 配对；每个 state 内先平均 checkpoint 与 stochastic repeat，再以 source episode 为 cluster 重采样，同一 episode 的多个 state 始终一起进入 bootstrap。点估计仍是 48 个初始 state 的均值差。下表直接给出 estimate 与区间，机器可读值同时保存在 JSON。",
            "",
        ]
    )
    for task, stages in INCLUDED_SKILL_STAGES.items():
        lines.extend(
            [
                f"### Paired comparisons: `{task}`",
                "",
                "| Stage | Comparison | ΔSR pp [95% CI] | ΔE_GT cm [95% CI] |",
                "|---|---|---:|---:|",
            ]
        )
        for stage in stages:
            for label, comparison in comparisons[task][stage].items():
                sr = comparison["success_rate_difference"]
                error = comparison["e_gt_cm_difference"]
                lines.append(
                    f"| {stage} | {label} | "
                    f"{100.0 * sr['estimate']:.1f} "
                    f"[{100.0 * sr['ci95_low']:.1f}, {100.0 * sr['ci95_high']:.1f}] | "
                    f"{error['estimate']:.2f} "
                    f"[{error['ci95_low']:.2f}, {error['ci95_high']:.2f}] |"
                )
        lines.append("")
    args.output_markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.resolve().write_text("\n".join(lines), encoding="utf-8")
    print(f"report_json={args.output_json.resolve()}")
    print(f"report_markdown={args.output_markdown.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
