#!/usr/bin/env python3
"""Audit or run the clean FurnitureBench skill-level evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from src.eval.skill_level import INCLUDED_SKILL_STAGES


RUNTIME_RELATIVE_PATHS = (
    "scripts/evaluate_furniturebench_skill_level.py",
    "scripts/run_furniturebench_skill_level_matrix.py",
    "reports/clean_skill_level_checkpoint_registry_20260923.json",
    "src/eval/skill_level.py",
    "src/eval/state_bank.py",
    "src/eval/skill_annotation_util.py",
    "src/eval/rollout.py",
    "src/eval/progress_schema.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("reports/clean_skill_level_checkpoint_registry_20260923.json"),
    )
    parser.add_argument("--state-banks-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--expected-state-count", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--repeat-seeds", type=int, nargs="*", default=[923001, 923002, 923003]
    )
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--train-seed", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument(
        "--lamp-bulb-fsm-pos-threshold",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_repo_state(repo: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
        capture_output=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "status_porcelain": status.stdout.splitlines() if status.returncode == 0 else None,
        "tracked_diff_sha256": (
            hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
        ),
    }


def lightweight_matrix_fingerprint(args: argparse.Namespace, cells: list[dict]) -> dict:
    repo = Path.cwd().resolve()
    banks = sorted({Path(cell["state_bank"]) for cell in cells})
    root_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "root_git_commit": (
            root_commit.stdout.strip() if root_commit.returncode == 0 else None
        ),
        "runtime_source_sha256": {
            relative: sha256(repo / relative) for relative in RUNTIME_RELATIVE_PATHS
        },
        "furniture_bench_repository_state": git_repo_state(repo / "furniture-bench"),
        "registry_sha256": sha256(args.registry.resolve()),
        "state_manifest_sha256": {
            str(bank): sha256(bank / "manifest.jsonl") for bank in banks
        },
    }


def base_matrix_contract(args: argparse.Namespace, cells: list[dict]) -> dict:
    return {
        "schema": "rr-clean-skill-level-matrix-run-v1",
        "cell_count": len(cells),
        "repeats": int(args.repeats),
        "repeat_seeds": list(args.repeat_seeds),
        "n_envs": int(args.n_envs),
        "state_count": int(args.expected_state_count),
        "expected_state_count": int(args.expected_state_count),
        "lamp_bulb_fsm_position_threshold": args.lamp_bulb_fsm_pos_threshold,
        "filters": {
            "condition": list(args.condition),
            "train_seed": list(args.train_seed),
            "task": list(args.task),
            "stage": list(args.stage),
            "max_cells": args.max_cells,
        },
        **lightweight_matrix_fingerprint(args, cells),
    }


def build_matrix_contract(args: argparse.Namespace, cells: list[dict]) -> dict:
    checkpoint_paths = sorted({Path(cell["checkpoint"]) for cell in cells})
    return {
        **base_matrix_contract(args, cells),
        "checkpoint_sha256": {
            str(path): sha256(path) for path in checkpoint_paths
        },
    }


def validate_lightweight_fingerprint(
    args: argparse.Namespace, cells: list[dict], contract: dict
) -> None:
    current = lightweight_matrix_fingerprint(args, cells)
    for key, value in current.items():
        if contract.get(key) != value:
            raise RuntimeError(f"formal matrix fingerprint changed: {key}")


def validate_cell_run_contract(cell: dict, matrix_contract: dict, args: argparse.Namespace) -> None:
    run = load_json(Path(cell["output"]) / "run.json")
    checkpoint_path = str(Path(cell["checkpoint"]))
    expected = {
        "condition": cell["condition"],
        "task": cell["task"],
        "skill_stage": cell["stage"],
        "checkpoint_sha256": matrix_contract["checkpoint_sha256"][checkpoint_path],
        "state_manifest_sha256": matrix_contract["state_manifest_sha256"][
            str(Path(cell["state_bank"]))
        ],
        "repeats": int(args.repeats),
        "repeat_seeds": list(args.repeat_seeds),
        "n_envs": int(args.n_envs),
        "git_commit": matrix_contract["root_git_commit"],
        "runtime_source_sha256": matrix_contract["runtime_source_sha256"],
        "furniture_bench_repository_state": matrix_contract[
            "furniture_bench_repository_state"
        ],
        "lamp_bulb_fsm_position_threshold": args.lamp_bulb_fsm_pos_threshold,
    }
    mismatches = [key for key, value in expected.items() if run.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"cell run contract differs from formal matrix ({', '.join(mismatches)}): "
            f"{cell['output']}"
        )


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Multiple GPU shards update the shared status file.  A per-process temp
    # name preserves atomic replace without shards clobbering each other's temp.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def selected(value: str, filters: list[str]) -> bool:
    return not filters or value in filters


def build_cells(args: argparse.Namespace) -> list[dict]:
    registry = load_json(args.registry.resolve())
    banks_root = args.state_banks_root.resolve()
    results_root = args.results_root.resolve()
    cells = []
    for condition in registry["conditions"]:
        condition_id = str(condition["id"])
        if not selected(condition_id, args.condition):
            continue
        for checkpoint in condition["checkpoints"]:
            train_seed = str(checkpoint["train_seed"])
            if not selected(train_seed, args.train_seed):
                continue
            checkpoint_path = Path(checkpoint["path"]).expanduser().resolve()
            for task, stages in INCLUDED_SKILL_STAGES.items():
                if not selected(task, args.task):
                    continue
                for stage in stages:
                    if not selected(stage, args.stage):
                        continue
                    bank = banks_root / task / stage
                    output = results_root / condition_id / train_seed / task / stage
                    cells.append(
                        {
                            "condition": condition_id,
                            "condition_label": condition["label"],
                            "train_seed": train_seed,
                            "checkpoint": str(checkpoint_path),
                            "wrist_image_transform": checkpoint["wrist_image_transform"],
                            "task": task,
                            "stage": stage,
                            "state_bank": str(bank),
                            "output": str(output),
                        }
                    )
    return cells


def cell_status(
    cell: dict, repeats: int, expected_state_count: int
) -> tuple[str, str | None]:
    checkpoint = Path(cell["checkpoint"])
    bank = Path(cell["state_bank"])
    output = Path(cell["output"])
    if not checkpoint.is_file():
        return "blocked", "missing_checkpoint"
    for name in ("campaign.json", "manifest.jsonl"):
        if not (bank / name).is_file():
            return "blocked", f"missing_state_bank_{name}"
    manifest_count = sum(
        1 for line in (bank / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if manifest_count != expected_state_count:
        return "blocked", f"state_count_{manifest_count}"
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        expected = expected_state_count * int(repeats)
        if summary.get("complete") and int(summary.get("attempted", -1)) == expected:
            return "complete", None
        return "partial", None
    return "ready", None


def snapshot(
    cells: list[dict], repeats: int, expected_state_count: int
) -> dict:
    output = []
    counts = {}
    for cell in cells:
        status, reason = cell_status(cell, repeats, expected_state_count)
        row = {**cell, "status": status, "reason": reason}
        output.append(row)
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema": "rr-clean-skill-level-matrix-status-v1",
        "cell_count": len(output),
        "counts": counts,
        "cells": output,
    }


def main() -> int:
    args = parse_args()
    if len(args.repeat_seeds) != args.repeats:
        raise ValueError("repeat-seeds must contain exactly repeats values")
    if args.expected_state_count <= 0:
        raise ValueError("expected-state-count must be positive")
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.initialize_only and not args.run:
        raise ValueError("initialize-only requires --run")
    cells = build_cells(args)
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    status_path = args.results_root.resolve() / "matrix_status.json"
    current = snapshot(cells, args.repeats, args.expected_state_count)
    write_json_atomic(status_path, current)
    print(json.dumps({"cell_count": current["cell_count"], "counts": current["counts"]}, sort_keys=True))
    if not args.run:
        return 0

    contract_path = args.results_root.resolve() / "matrix_run.json"
    if contract_path.exists():
        existing_contract = load_json(contract_path)
        expected_base = base_matrix_contract(args, cells)
        if any(
            existing_contract.get(key) != value
            for key, value in expected_base.items()
        ):
            raise RuntimeError("formal matrix contract does not match existing run")
        expected_checkpoint_paths = sorted(
            {str(Path(cell["checkpoint"])) for cell in cells}
        )
        if sorted(existing_contract.get("checkpoint_sha256", {})) != expected_checkpoint_paths:
            raise RuntimeError("formal matrix checkpoint set does not match existing run")
        matrix_contract = existing_contract
    else:
        current_contract = build_matrix_contract(args, cells)
        write_json_atomic(contract_path, current_contract)
        matrix_contract = current_contract

    if args.initialize_only:
        print(f"MATRIX_CONTRACT_INITIALIZED={contract_path}")
        return 0

    indexed_shard_cells = [
        (index, cell)
        for index, cell in enumerate(cells)
        if index % args.shard_count == args.shard_index
    ]
    print(
        f"SHARD index={args.shard_index} count={args.shard_count} "
        f"cells={len(indexed_shard_cells)}",
        flush=True,
    )
    for index, cell in indexed_shard_cells:
        validate_lightweight_fingerprint(args, cells, matrix_contract)
        status, reason = cell_status(cell, args.repeats, args.expected_state_count)
        if status == "complete":
            validate_cell_run_contract(cell, matrix_contract, args)
            continue
        if status == "blocked":
            raise RuntimeError(f"cell blocked ({reason}): {cell}")
        cmd = [
            sys.executable,
            "scripts/evaluate_furniturebench_skill_level.py",
            "--checkpoint",
            cell["checkpoint"],
            "--condition",
            cell["condition"],
            "--state-bank-dir",
            cell["state_bank"],
            "--output-dir",
            cell["output"],
            "--gpu",
            str(args.gpu),
            "--n-envs",
            str(args.n_envs),
            "--repeats",
            str(args.repeats),
            "--repeat-seeds",
            *[str(value) for value in args.repeat_seeds],
            "--wrist-image-transform",
            cell["wrist_image_transform"],
            "--annotation-source",
            "scripted",
        ]
        if args.lamp_bulb_fsm_pos_threshold is not None:
            cmd.extend(
                [
                    "--lamp-bulb-fsm-pos-threshold",
                    *[str(value) for value in args.lamp_bulb_fsm_pos_threshold],
                ]
            )
        if status == "partial":
            cmd.append("--resume")
        print(
            f"CELL {index + 1}/{len(cells)} shard={args.shard_index}/"
            f"{args.shard_count} {cell['condition']} "
            f"{cell['train_seed']} {cell['task']}/{cell['stage']}",
            flush=True,
        )
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            write_json_atomic(
                status_path,
                snapshot(cells, args.repeats, args.expected_state_count),
            )
            return int(result.returncode)
        validate_cell_run_contract(cell, matrix_contract, args)
        write_json_atomic(
            status_path,
            snapshot(cells, args.repeats, args.expected_state_count),
        )

    final = snapshot(cells, args.repeats, args.expected_state_count)
    write_json_atomic(status_path, final)
    incomplete_shard = [
        cell
        for _index, cell in indexed_shard_cells
        if cell_status(cell, args.repeats, args.expected_state_count)[0]
        != "complete"
    ]
    if incomplete_shard:
        raise RuntimeError(
            f"matrix shard ended with {len(incomplete_shard)} incomplete cells"
        )
    if args.shard_count == 1 and final["counts"].get("complete", 0) != len(cells):
        raise RuntimeError("matrix runner ended without completing every selected cell")
    print(
        f"MATRIX_SHARD_COMPLETE={args.shard_index}/{args.shard_count} "
        f"cells={len(indexed_shard_cells)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
