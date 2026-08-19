#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import wandb


ENTITY = "huyue233-zhejiang-university"
PROJECT = "multi-task-rgbd-skill-med-0801"
TASKS = ["one_leg", "round_table", "lamp"]

EXPECTED = {
    1: ("rgbd", False, False, False, False, False, "rgbd"),
    2: ("rgbd-skill-point", True, False, False, False, False, "rgbd"),
    3: ("rgbd-skill-point-colored", True, False, True, False, False, "rgbd"),
    4: ("rgbd", False, True, False, False, False, "rgbd"),
    5: ("rgbd-skill-point", True, True, False, False, False, "rgbd"),
    6: ("rgbd", False, False, False, False, False, "image"),
    7: ("rgbd-skill-grasp-part", False, False, False, False, True, "rgbd"),
    8: ("rgbd-skill-grasp-part-colored", False, False, True, True, True, "rgbd"),
}


def expect_equal(errors, label, actual, expected):
    if actual != expected:
        errors.append(f"{label}={actual!r}, expected {expected!r}")


def validate_config(exp, run):
    cfg = run.config
    data = cfg.get("data", {})
    training = cfg.get("training", {})
    suffix, point, skill, point_colored, grasp_colored, grasp_part, observation = EXPECTED[exp]
    errors = []

    expect_equal(errors, "project", run.project, PROJECT)
    expect_equal(errors, "task", cfg.get("task"), TASKS)
    expect_equal(errors, "randomness", cfg.get("randomness"), "med")
    expect_equal(errors, "observation_type", cfg.get("observation_type"), observation)
    expect_equal(errors, "data.suffix", data.get("suffix"), suffix)
    expect_equal(errors, "data.storage_format", data.get("storage_format"), "lmdb")
    expect_equal(errors, "data.load_into_memory", data.get("load_into_memory"), False)
    expect_equal(errors, "data.dataloader_workers", data.get("dataloader_workers"), 4)
    expect_equal(errors, "data.ddp_shard_enabled", data.get("ddp_shard_enabled"), True)
    expect_equal(errors, "data.annotate_guidance_point", data.get("annotate_guidance_point"), point)
    expect_equal(errors, "data.annotate_skill_one_hot", data.get("annotate_skill_one_hot"), skill)
    expect_equal(errors, "data.annotate_guidance_point_colored", data.get("annotate_guidance_point_colored"), point_colored)
    expect_equal(errors, "data.annotate_grasp", data.get("annotate_grasp"), False)
    expect_equal(errors, "data.annotate_grasp_colored", data.get("annotate_grasp_colored"), grasp_colored)
    expect_equal(errors, "data.annotate_grasp_part", data.get("annotate_grasp_part"), grasp_part)
    expect_equal(errors, "training.batch_size", training.get("batch_size"), 512)
    expect_equal(errors, "training.num_epochs", training.get("num_epochs"), 3000)
    expect_equal(errors, "training.steps_per_epoch", training.get("steps_per_epoch"), 100)
    expect_equal(errors, "training.save_per_epoch", training.get("save_per_epoch"), 500)
    expect_equal(errors, "training.world_size", training.get("world_size"), 2)
    expect_equal(errors, "training.per_rank_batch_size", training.get("per_rank_batch_size"), 256)

    if exp == 6:
        vision_encoder = cfg.get("vision_encoder", {})
        # The approved launcher uses the `resnet` alias; Hydra resolves it to
        # the concrete registered model name `resnet18` in the run config.
        expect_equal(errors, "vision_encoder.model", vision_encoder.get("model"), "resnet18")
        expect_equal(errors, "vision_encoder.pretrained", vision_encoder.get("pretrained"), False)

    return errors


def resolve_run(api, run_id, run_name):
    if run_id and run_id != "-":
        return api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    candidates = list(
        api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"display_name": run_name},
            order="-created_at",
            per_page=5,
        )
    )
    if not candidates:
        raise RuntimeError(f"No W&B run found with display_name={run_name!r}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    args = parser.parse_args()

    with args.registry.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    api = wandb.Api(timeout=30)
    print("exp\tstate\tepoch\tconfig_ok\trun_id\trun_name\terrors")
    config_failed = False
    api_failed = False
    for row in rows:
        if row.get("status") not in {"started", "resumed"}:
            continue
        exp = int(row["exp"])
        try:
            run = resolve_run(api, row.get("wandb_run_id", "-"), row["wandb_run"])
            errors = validate_config(exp, run)
            config_failed |= bool(errors)
            print(
                "\t".join(
                    (
                        str(exp),
                        str(run.state),
                        str(run.summary.get("epoch", "-")),
                        "true" if not errors else "false",
                        str(run.id),
                        str(run.name),
                        " | ".join(errors).replace("\t", " "),
                    )
                )
            )
        except Exception as exc:
            print(
                f"{exp}\tapi-error\t-\tfalse\t{row.get('wandb_run_id', '-')}\t"
                f"{row['wandb_run']}\t{type(exc).__name__}: {exc}"
            )
            api_failed = True

    if api_failed:
        sys.exit(3)
    if config_failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
