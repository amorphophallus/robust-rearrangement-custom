#!/usr/bin/env python3
"""Run the N0/N2/N4-train by N0..N6-eval fixed-state matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from src.eval.skill_level import INCLUDED_SKILL_STAGES

NOISE_LEVELS = ("n0", "n1", "n2", "n3", "n4", "n5", "n6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--state-banks-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--state-count", type=int, default=8)
    parser.add_argument(
        "--state-start",
        type=int,
        default=0,
        help="Zero-based first state-bank entry; use this for incremental blocks.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repeat-seeds", type=int, nargs="*", default=[2026092501])
    parser.add_argument("--annotation-noise-seed", type=int, default=2026092501)
    parser.add_argument("--train-noise", action="append", default=[])
    parser.add_argument("--train-seed", action="append", default=[])
    parser.add_argument("--eval-noise", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--first-checkpoint-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected(value: str, filters: list[str]) -> bool:
    return not filters or value in filters


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_cells(args: argparse.Namespace) -> list[dict]:
    registry = load_json(args.registry.resolve())
    cells = []
    for condition in registry["train_conditions"]:
        train_noise = str(condition["train_noise_level"])
        if not selected(train_noise, args.train_noise):
            continue
        checkpoints = condition["checkpoints"][:1] if args.first_checkpoint_only else condition["checkpoints"]
        for checkpoint in checkpoints:
            train_seed = str(checkpoint["train_seed"])
            if not selected(train_seed, args.train_seed):
                continue
            for eval_noise in NOISE_LEVELS:
                if not selected(eval_noise, args.eval_noise):
                    continue
                for task, stages in INCLUDED_SKILL_STAGES.items():
                    if not selected(task, args.task):
                        continue
                    for stage in stages:
                        if not selected(stage, args.stage):
                            continue
                        cells.append(
                            {
                                "train_noise_level": train_noise,
                                "train_seed": train_seed,
                                "eval_noise_level": eval_noise,
                                "checkpoint": str(Path(checkpoint["path"]).expanduser().resolve()),
                                "wrist_image_transform": checkpoint.get("wrist_image_transform", "center-crop-224"),
                                "task": task,
                                "stage": stage,
                                "state_bank": str(args.state_banks_root.resolve() / task / stage),
                                "output": str(args.results_root.resolve() / train_noise / train_seed / eval_noise / task / stage),
                            }
                        )
    return cells


def cell_status(
    cell: dict, state_start: int, state_count: int, repeats: int
) -> tuple[str, str | None]:
    checkpoint = Path(cell["checkpoint"])
    bank = Path(cell["state_bank"])
    if not checkpoint.is_file():
        return "blocked", "missing_checkpoint"
    if not (bank / "manifest.jsonl").is_file():
        return "blocked", "missing_state_bank"
    available = sum(bool(line.strip()) for line in (bank / "manifest.jsonl").read_text().splitlines())
    if available < state_start + state_count:
        return "blocked", f"only_{available}_states"
    summary_path = Path(cell["output"]) / "summary.json"
    if not summary_path.is_file():
        return "ready", None
    summary = load_json(summary_path)
    if summary.get("complete") and int(summary.get("attempted", -1)) == state_count * repeats:
        return "complete", None
    return "partial", None


def snapshot(cells: list[dict], args: argparse.Namespace) -> dict:
    rows, counts = [], {}
    for cell in cells:
        status, reason = cell_status(
            cell, args.state_start, args.state_count, args.repeats
        )
        rows.append({**cell, "status": status, "reason": reason})
        counts[status] = counts.get(status, 0) + 1
    return {"schema": "rr-noisy-skill-level-matrix-status-v1", "counts": counts, "cells": rows}


def main() -> int:
    args = parse_args()
    if args.state_start < 0:
        raise ValueError("state-start must be non-negative")
    if len(args.repeat_seeds) != args.repeats:
        raise ValueError("repeat-seeds must contain exactly repeats values")
    cells = build_cells(args)
    if args.max_cells is not None:
        cells = cells[:args.max_cells]
    status_path = args.results_root.resolve() / "matrix_status.json"
    write_json_atomic(status_path, snapshot(cells, args))
    print(json.dumps({"cell_count": len(cells), "counts": snapshot(cells, args)["counts"]}, sort_keys=True))
    if not args.run:
        return 0

    contract_path = args.results_root.resolve() / "matrix_run.json"
    contract = {
        "schema": "rr-noisy-skill-level-matrix-run-v1",
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry.resolve()),
        "cell_count": len(cells),
        "state_count": args.state_count,
        "state_start": args.state_start,
        "repeats": args.repeats,
        "repeat_seeds": args.repeat_seeds,
        "annotation_noise_seed": args.annotation_noise_seed,
        "noise_levels": list(NOISE_LEVELS),
        "formula_report": "scripts/report_furniturebench_noisy_skill_level.py",
        "command": [sys.executable, *sys.argv],
    }
    if contract_path.exists() and load_json(contract_path) != contract:
        raise RuntimeError("matrix contract differs from existing run")
    if not contract_path.exists():
        write_json_atomic(contract_path, contract)

    for index, cell in enumerate(cells):
        status, reason = cell_status(
            cell, args.state_start, args.state_count, args.repeats
        )
        if status == "complete":
            continue
        if status == "blocked":
            raise RuntimeError(f"blocked cell ({reason}): {cell}")
        cmd = [
            sys.executable, "scripts/evaluate_furniturebench_skill_level.py",
            "--checkpoint", cell["checkpoint"],
            "--condition", f"tagpoint_train_{cell['train_noise_level']}",
            "--state-bank-dir", cell["state_bank"],
            "--output-dir", cell["output"],
            "--gpu", str(args.gpu), "--n-envs", str(args.n_envs),
            "--state-start", str(args.state_start),
            "--state-limit", str(args.state_count), "--repeats", str(args.repeats),
            "--rollout-seed-offset", str(args.state_start // args.n_envs),
            "--repeat-seeds", *[str(value) for value in args.repeat_seeds],
            "--wrist-image-transform", cell["wrist_image_transform"],
            "--annotation-source", "scripted",
            "--eval-noise-level", cell["eval_noise_level"],
            "--annotation-noise-seed", str(args.annotation_noise_seed),
        ]
        if status == "partial":
            cmd.append("--resume")
        print(
            f"CELL {index + 1}/{len(cells)} train={cell['train_noise_level']} "
            f"eval={cell['eval_noise_level']} {cell['task']}/{cell['stage']}", flush=True
        )
        result = subprocess.run(cmd, check=False)
        if result.returncode:
            write_json_atomic(status_path, snapshot(cells, args))
            return int(result.returncode)
        write_json_atomic(status_path, snapshot(cells, args))

    subprocess.run(
        [sys.executable, "scripts/report_furniturebench_noisy_skill_level.py", "--results-root", str(args.results_root.resolve())],
        check=True,
    )
    print("NOISY_SKILL_LEVEL_MATRIX_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
