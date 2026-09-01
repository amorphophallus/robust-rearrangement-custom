#!/usr/bin/env python3
"""Convert raw FurnitureBench pickles to LMDB with explicit offline annotation."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


TASKS = ("one_leg", "round_table", "lamp")
ANNOTATION_MODES = (
    "none",
    "guidance-point",
    "guidance-point-colored",
    "grasp-part",
    "grasp-part-colored",
)


def processing_environment():
    environment = os.environ.copy()
    conda_lib = str(Path(sys.prefix).resolve() / "lib")
    existing = environment.get("LD_LIBRARY_PATH", "")
    components = [component for component in existing.split(":") if component]
    environment["LD_LIBRARY_PATH"] = ":".join(
        [conda_lib, *[component for component in components if component != conda_lib]]
    )
    return environment


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "FurnitureBench pickle-to-LMDB runner. Image annotation happens only "
            "inside this conversion step and never modifies the source pickle."
        )
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--input-suffix", required=True)
    parser.add_argument("--output-suffix", required=True)
    parser.add_argument(
        "--image-annotation-mode",
        choices=ANNOTATION_MODES,
        required=True,
        help="Required explicit choice for images stored in the LMDB.",
    )
    parser.add_argument("--randomness", choices=("low", "med", "high"), default="low")
    parser.add_argument("--episodes-per-task", type=int, default=None)
    parser.add_argument("--task-episode-limit", action="append", default=[], metavar="TASK=N")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-cpus", type=int, default=1)
    parser.add_argument("--map-size-gb", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--provenance-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def task_limits(args):
    limits = {}
    if args.episodes_per_task is not None:
        if args.episodes_per_task <= 0:
            raise ValueError("--episodes-per-task must be positive")
        limits.update({task: args.episodes_per_task for task in args.tasks})
    for value in args.task_episode_limit:
        task, separator, raw_number = value.partition("=")
        if not separator or task not in args.tasks:
            raise ValueError(f"--task-episode-limit must use selected TASK=N: {value!r}")
        number = int(raw_number)
        if number <= 0:
            raise ValueError(f"--task-episode-limit must be positive: {value!r}")
        limits[task] = number
    return limits


def main():
    args = build_parser().parse_args()
    if args.image_size <= 0 or args.batch_size <= 0 or args.n_cpus <= 0:
        raise ValueError("--image-size, --batch-size, and --n-cpus must be positive")
    if not args.input_suffix.strip() or not args.output_suffix.strip():
        raise ValueError("input/output suffixes must be nonempty")

    repo_root = args.repo_root.expanduser().resolve()
    limits = task_limits(args)
    command = [
        sys.executable,
        "-m",
        "src.data_processing.process_pickles_to_lmdb",
        "-c",
        "diffik",
        "-d",
        "sim",
        "-f",
        *args.tasks,
        "-s",
        "rollout",
        "-r",
        args.randomness,
        "-o",
        "success",
        "--suffix",
        args.input_suffix,
        "--output-suffix",
        args.output_suffix,
        "--image-annotation-mode",
        args.image_annotation_mode,
        "--require-source-image-annotation-mode",
        "none",
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--n-cpus",
        str(args.n_cpus),
        "--map-size-gb",
        str(args.map_size_gb),
    ]
    if limits:
        command.extend(("--task-episode-limit", *[f"{task}={limits[task]}" for task in args.tasks if task in limits]))
    if args.output_dir is not None:
        command.extend(("--output-dir", str(args.output_dir.expanduser().resolve())))
    if args.provenance_json is not None:
        command.extend(("--provenance-json", str(args.provenance_json.expanduser().resolve())))
    if args.overwrite:
        command.append("--overwrite")

    print(f"repo_root={repo_root}")
    print(f"source_pickle_image_annotation_mode=none")
    print(f"lmdb_image_annotation_mode={args.image_annotation_mode}")
    print(f"command={shlex.join(command)}")
    if not args.dry_run:
        subprocess.run(
            command,
            cwd=repo_root,
            env=processing_environment(),
            check=True,
        )


if __name__ == "__main__":
    main()
