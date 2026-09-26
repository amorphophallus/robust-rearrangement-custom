#!/usr/bin/env python3
"""Resumable full-rollout N0/N2/N4-train by N0..N6-eval pilot.

This deliberately uses one evaluator process per task: FurnitureBench's
multi-task wrapper starts a fresh child process for exactly this reason, but
does not persist its parent aggregate to ``--task-summary-out``.  Persisting
the task summaries independently makes every result auditable and resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


NOISE_STD_M = {
    "n0": 0.0, "n1": 0.003, "n2": 0.006, "n3": 0.012,
    "n4": 0.024, "n5": 0.048, "n6": 0.096,
}
TASKS = ("one_leg", "round_table", "lamp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--n-rollouts", type=int, default=12)
    parser.add_argument("--simulator-seed", type=int, default=0)
    parser.add_argument("--annotation-noise-seed", type=int, default=2026092501)
    parser.add_argument("--train-noise", action="append", default=[])
    parser.add_argument("--eval-noise", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Run all three registered seeds instead of the pilot's first seed.",
    )
    parser.add_argument("--max-cells", type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected(value: str, filters: list[str]) -> bool:
    return not filters or value in filters


def build_cells(args: argparse.Namespace) -> list[dict]:
    registry = load_json(args.registry.resolve())
    cells = []
    for condition in registry["train_conditions"]:
        train_noise = str(condition["train_noise_level"])
        if not selected(train_noise, args.train_noise):
            continue
        checkpoints = condition["checkpoints"] if args.all_checkpoints else condition["checkpoints"][:1]
        for checkpoint in checkpoints:
            checkpoint_path = Path(checkpoint["path"]).expanduser().resolve()
            train_seed = str(checkpoint["train_seed"])
            for eval_noise, pos_std_m in NOISE_STD_M.items():
                if not selected(eval_noise, args.eval_noise):
                    continue
                for task in TASKS:
                    if not selected(task, args.task):
                        continue
                    cells.append(
                        {
                            "train_noise_level": train_noise,
                            "train_seed": train_seed,
                            "eval_noise_level": eval_noise,
                            "eval_noise_pos_std_m": pos_std_m,
                            "checkpoint": str(checkpoint_path),
                            "wrist_image_transform": checkpoint.get("wrist_image_transform", "center-crop-224"),
                            "task": task,
                            "summary": str(
                                args.results_root.resolve() / "summaries" / train_noise /
                                train_seed / eval_noise / f"{task}.json"
                            ),
                        }
                    )
    return cells


def summary_error(cell: dict, args: argparse.Namespace) -> str | None:
    checkpoint = Path(cell["checkpoint"])
    if not checkpoint.is_file():
        return "missing_checkpoint"
    path = Path(cell["summary"])
    if not path.is_file():
        return "missing_summary"
    try:
        payload = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"invalid_summary:{exc}"
    checks = {
        "task": cell["task"],
        "n_envs": args.n_envs,
        "n_rollouts": args.n_rollouts,
        "eval_randomness": "low",
        "simulator_seed": args.simulator_seed,
        "annotation_source": "scripted",
        "success_criterion": "physics_reward",
        "action_type": "pos",
        "wrist_image_transform": cell["wrist_image_transform"],
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            return f"summary_{key}={payload.get(key)!r}, expected {expected!r}"
    if Path(payload.get("checkpoint_path", "")).resolve() != checkpoint:
        return "summary_checkpoint_path_mismatch"
    if not 0 <= int(payload.get("n_success", -1)) <= args.n_rollouts:
        return "invalid_success_count"
    noise = payload.get("annotation_noise_config", {})
    if (
        noise.get("pos_std_m") != cell["eval_noise_pos_std_m"]
        or noise.get("seed") != args.annotation_noise_seed
        or noise.get("mode") != "gaussian_clip_2sigma"
        or noise.get("apply_to") != "point"
    ):
        return "annotation_noise_config_mismatch"
    if not isinstance(payload.get("skill_state_counts"), dict) or not isinstance(payload.get("skill_completion_counts"), dict):
        return "missing_skill_level_counts"
    return None


def snapshot(cells: list[dict], args: argparse.Namespace) -> dict:
    rows, counts = [], {}
    for cell in cells:
        error = summary_error(cell, args)
        if error is None:
            state, reason = "complete", None
        elif error == "missing_summary":
            state, reason = "ready", None
        elif error == "missing_checkpoint":
            state, reason = "blocked", error
        else:
            state, reason = "invalid", error
        counts[state] = counts.get(state, 0) + 1
        rows.append({**cell, "status": state, "reason": reason})
    return {
        "schema": "rr-noisy-full-rollout-matrix-status-v1",
        "counts": counts,
        "cells": rows,
    }


def command_for(cell: dict, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-m", "src.eval.evaluate_model",
        "--wt-path", cell["checkpoint"], "--gpu", str(args.gpu),
        "--task", cell["task"], "--n-envs", str(args.n_envs),
        "--n-rollouts", str(args.n_rollouts), "--seed", str(args.simulator_seed),
        "--if-exists", "error", "--max-rollout-steps", "1000",
        "--action-type", "pos", "--observation-space", "image",
        "--wrist-image-transform", cell["wrist_image_transform"], "--randomness", "low",
        "--annotate-skill", "--enable-annotation-verify", "--annotation-source", "scripted",
        "--sim-front-camera-preset", "original", "--eepose-frame", "robot-base",
        "--tracking-metric-type", "position", "--max-saved-rollouts", "0",
        "--annotation-noise-pos-std-m", str(cell["eval_noise_pos_std_m"]),
        "--annotation-noise-seed", str(args.annotation_noise_seed),
        "--annotation-noise-mode", "gaussian_clip_2sigma",
        "--annotation-noise-apply-to", "point",
        "--rollout-suffix-model-name", (
            f"noisy_train_noisy_eval_20260925/pilot12/"
            f"train_{cell['train_noise_level']}_{cell['train_seed']}/"
            f"eval_{cell['eval_noise_level']}/{cell['task']}"
        ),
        "--task-summary-out", cell["summary"],
    ]


def main() -> int:
    args = parse_args()
    if args.n_envs != args.n_rollouts:
        raise ValueError("Pilot uses one 12-environment batch: --n-envs must equal --n-rollouts")
    cells = build_cells(args)
    if args.max_cells is not None:
        cells = cells[:args.max_cells]
    root = args.results_root.resolve()
    status_path = root / "matrix_status.json"
    write_json_atomic(status_path, snapshot(cells, args))
    print(json.dumps({"cell_count": len(cells), "counts": snapshot(cells, args)["counts"]}, sort_keys=True))
    if not args.run:
        return 0

    hashes = {}
    for cell in cells:
        checkpoint = Path(cell["checkpoint"])
        if not checkpoint.is_file():
            raise RuntimeError(f"missing checkpoint: {checkpoint}")
        hashes[str(checkpoint)] = sha256(checkpoint)
    contract = {
        "schema": "rr-noisy-full-rollout-matrix-run-v1",
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry.resolve()),
        "checkpoint_sha256": hashes,
        "cell_count": len(cells),
        "n_envs": args.n_envs,
        "n_rollouts_per_task_cell": args.n_rollouts,
        "simulator_seed": args.simulator_seed,
        "annotation_noise_seed": args.annotation_noise_seed,
        "noise_levels": NOISE_STD_M,
        "annotation_source": "scripted",
        "command": [sys.executable, *sys.argv],
    }
    contract_path = root / "matrix_run.json"
    if contract_path.exists() and load_json(contract_path) != contract:
        raise RuntimeError("matrix contract differs from existing run")
    if not contract_path.exists():
        write_json_atomic(contract_path, contract)

    for index, cell in enumerate(cells):
        error = summary_error(cell, args)
        if error is None:
            continue
        if error != "missing_summary":
            raise RuntimeError(f"cannot overwrite invalid cell ({error}): {cell}")
        summary = Path(cell["summary"])
        summary.parent.mkdir(parents=True, exist_ok=True)
        log_path = summary.with_suffix(".log")
        cmd = command_for(cell, args)
        print(
            f"CELL {index + 1}/{len(cells)} train={cell['train_noise_level']}/"
            f"{cell['train_seed']} eval={cell['eval_noise_level']} task={cell['task']}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write("COMMAND " + json.dumps(cmd) + "\n")
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            write_json_atomic(status_path, snapshot(cells, args))
            return int(result.returncode)
        error = summary_error(cell, args)
        if error is not None:
            write_json_atomic(status_path, snapshot(cells, args))
            raise RuntimeError(f"completed cell failed validation: {error}")
        write_json_atomic(status_path, snapshot(cells, args))

    print("NOISY_FULL_ROLLOUT_MATRIX_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
