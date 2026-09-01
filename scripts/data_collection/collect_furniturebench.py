#!/usr/bin/env python3
"""Collect FurnitureBench rollouts without drawing into saved RGB frames."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


TASKS = ("one_leg", "round_table", "lamp")
DEFAULT_MAX_STEPS = {"one_leg": 700, "round_table": 1000, "lamp": 1000}
DEFAULT_POST_SUCCESS = {"one_leg": 200, "round_table": 100, "lamp": 20}
TASK_SEED_OFFSETS = {"one_leg": 0, "round_table": 100_000, "lamp": 200_000}


def parse_task_values(values: list[str], defaults: dict[str, int], name: str):
    result = dict(defaults)
    for value in values:
        task, separator, raw_number = value.partition("=")
        if not separator or task not in TASKS:
            raise ValueError(f"{name} must use TASK=N with TASK in {TASKS}: {value!r}")
        number = int(raw_number)
        if number < 0:
            raise ValueError(f"{name} must be non-negative: {value!r}")
        result[task] = number
    return result


def parse_checkpoint_values(values: list[str], repo_root: Path):
    result = {
        task: repo_root / "checkpoints" / "rppo" / task / "low" / "actor_chkpt.pt"
        for task in TASKS
    }
    for value in values:
        task, separator, raw_path = value.partition("=")
        if not separator or task not in TASKS:
            raise ValueError(f"--checkpoint must use TASK=PATH with TASK in {TASKS}: {value!r}")
        result[task] = Path(raw_path).expanduser().resolve()
    return result


def output_dir(data_root: Path, task: str, randomness: str, suffix: str):
    return data_root / "raw" / "diffik" / "sim" / task / "rollout" / randomness / suffix


def evaluation_environment(data_root: Path):
    """Expose the active Conda runtime libraries required by Isaac Gym."""
    environment = os.environ.copy()
    environment["DATA_DIR_RAW"] = str(data_root)
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
            "Sequential FurnitureBench rollout collector derived from gpu-snatcher's "
            "auto_data_preparation flow. Saved RGB-D stays raw; only skill and 2-D/3-D "
            "guidance metadata are recorded."
        )
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--target-successes", type=int, required=True)
    parser.add_argument("--output-suffix", required=True)
    parser.add_argument(
        "--annotation-source",
        choices=("scripted", "vlm"),
        required=True,
        help="Required provenance gate; never inferred from evaluate_model defaults.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20_901_000,
        help="Base simulator seed; each task receives a fixed disjoint offset.",
    )
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--randomness", choices=("low", "med", "high"), default="low")
    parser.add_argument("--action-type", choices=("pos", "delta"), default="pos")
    parser.add_argument("--observation-space", choices=("image", "state"), default="image")
    parser.add_argument("--if-exists", choices=("append", "overwrite", "skip", "error"), default="append")
    parser.add_argument("--max-rollout-steps", action="append", default=[], metavar="TASK=N")
    parser.add_argument("--rollout-after-success", action="append", default=[], metavar="TASK=N")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="TASK=PATH")
    parser.add_argument("--compress-pickles", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--vlm-base-url", default=os.environ.get("VLM_GUIDANCE_URL"))
    parser.add_argument("--vlm-query-interval", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.target_successes <= 0:
        raise ValueError("--target-successes must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.base_seed < 0:
        raise ValueError("--base-seed must be non-negative")
    if not args.output_suffix.strip() or "/" in args.output_suffix:
        raise ValueError("--output-suffix must be a nonempty single path component")
    if args.annotation_source == "vlm" and not args.vlm_base_url:
        raise ValueError("--annotation-source vlm requires --vlm-base-url or VLM_GUIDANCE_URL")

    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else Path(os.environ.get("DATA_DIR_RAW", repo_root / "data")).expanduser().resolve()
    )
    max_steps = parse_task_values(
        args.max_rollout_steps, DEFAULT_MAX_STEPS, "--max-rollout-steps"
    )
    post_success = parse_task_values(
        args.rollout_after_success,
        DEFAULT_POST_SUCCESS,
        "--rollout-after-success",
    )
    checkpoints = parse_checkpoint_values(args.checkpoint, repo_root)

    print(f"repo_root={repo_root}")
    print(f"data_root={data_root}")
    print(f"annotation_source={args.annotation_source}")
    print("saved_image_annotation_mode=none")
    print("saved_metadata=skill,guidance_point,guidance_point_2d,camera_info")

    for task in args.tasks:
        task_seed = args.base_seed + TASK_SEED_OFFSETS[task]
        if task_seed > 2_147_483_647:
            raise ValueError(f"Derived seed exceeds int32 range for {task}: {task_seed}")
        checkpoint = checkpoints[task]
        target_dir = output_dir(data_root, task, args.randomness, args.output_suffix)
        existing_pickles = list(target_dir.rglob("*.pkl*")) if target_dir.exists() else []
        if existing_pickles and not args.allow_existing_output:
            raise FileExistsError(
                f"Refusing to mix with {len(existing_pickles)} existing pickle(s): {target_dir}. "
                "Choose a new --output-suffix or pass --allow-existing-output explicitly."
            )
        if not checkpoint.is_file() and not args.dry_run:
            raise FileNotFoundError(checkpoint)

        command = [
            sys.executable,
            "-m",
            "src.eval.evaluate_model",
            "--gpu",
            str(args.gpu),
            "--seed",
            str(task_seed),
            "--n-envs",
            str(args.n_envs),
            "--n-rollouts",
            str(args.n_envs),
            "--target-successes",
            str(args.target_successes),
            "--max-saved-rollouts",
            str(args.target_successes),
            "-f",
            task,
            "--if-exists",
            args.if_exists,
            "--max-rollout-steps",
            str(max_steps[task]),
            "--action-type",
            args.action_type,
            "--observation-space",
            args.observation_space,
            "--randomness",
            args.randomness,
            "--wt-path",
            str(checkpoint),
            "--save-rollouts",
            "--save-depth-image",
            "--output-only-pickle",
            "--annotate-skill",
            "--enable-annotation-verify",
            "--annotation-source",
            args.annotation_source,
            "--save-rollouts-suffix",
            args.output_suffix,
            "--rollout-after-success",
            str(post_success[task]),
        ]
        if args.compress_pickles:
            command.append("--compress-pickles")
        if args.verbose:
            command.append("--verbose")
        if args.annotation_source == "vlm":
            command.extend(("--vlm-base-url", args.vlm_base_url))
            if args.vlm_query_interval is not None:
                command.extend(("--vlm-query-interval", str(args.vlm_query_interval)))

        forbidden = {
            "--guidance-point-on-image",
            "--grasp-annotation-on-image",
            "--grasp-part-annotate",
            "--skill-on-image",
        }
        if forbidden.intersection(command):
            raise RuntimeError("Collection command would mutate saved raw images")

        print(f"task={task} simulator_seed={task_seed} output_dir={target_dir}")
        print(f"command={shlex.join(command)}")
        if not args.dry_run:
            subprocess.run(
                command,
                cwd=repo_root,
                env=evaluation_environment(data_root),
                check=True,
            )


if __name__ == "__main__":
    main()
