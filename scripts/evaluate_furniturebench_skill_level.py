#!/usr/bin/env python3
"""Evaluate one policy checkpoint from a fixed FurnitureBench stage bank."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

# Isaac Gym native modules must be loaded before torch.
import isaacgym  # noqa: F401
import numpy as np
import torch
from omegaconf import OmegaConf

from src.behavior import get_actor
from src.behavior.base import (
    model_requires_skill_input,
    model_uses_grasp,
    model_uses_grasp_colored,
    model_uses_grasp_part,
    model_uses_guidance_point,
    model_uses_guidance_point_colored,
    validate_annotation_config,
)
from src.behavior.diffusion import DiffusionPolicy
from src.common.eepose import ROBOT_BASE
from src.common.tasks import task2idx
from src.common.vision import (
    CENTER_CROP_224_SPATIAL_TRANSFORM,
    LEGACY_224_SPATIAL_TRANSFORM,
)
from src.eval.evaluate_model import _extract_model_state_dict
from src.eval.annotation_noise import make_annotation_noise_config
from src.eval.skill_level import (
    RESTORE_ARM_DOF_POSITION_TOLERANCE,
    RESTORE_GRIPPER_DOF_POSITION_TOLERANCE,
    RESTORE_PART_ORIENTATION_TOLERANCE_DEG,
    RESTORE_PART_POSITION_TOLERANCE_M,
    calibrated_stage_timeout,
    rollout_skill_stage_batch,
    stable_annotation_noise_seed_offset,
    summarize_skill_level_records,
)
from src.eval.state_bank import load_state_record
from src.eval.rollout import _draw_guidance_points_for_all_envs
from src.gym import FULL_OBS, get_rl_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--state-bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--repeat-seeds", type=int, nargs="*", default=[923001, 923002, 923003]
    )
    parser.add_argument(
        "--rollout-seed-offset",
        type=int,
        default=0,
        help="Add an explicit offset to the per-batch rollout seed (for exact single-state replay).",
    )
    parser.add_argument("--state-limit", type=int, default=None)
    parser.add_argument(
        "--state-start",
        type=int,
        default=0,
        help="Zero-based first state-bank manifest entry to evaluate.",
    )
    parser.add_argument(
        "--state-index",
        type=int,
        default=None,
        help="Evaluate exactly one zero-based state-bank manifest entry.",
    )
    parser.add_argument("--max-stage-steps", type=int, default=None)
    parser.add_argument("--randomness", choices=("low",), default="low")
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument(
        "--wrist-image-transform",
        choices=("checkpoint", "legacy-resize", "center-crop-224"),
        default="checkpoint",
    )
    parser.add_argument("--annotation-source", choices=("scripted",), required=True)
    parser.add_argument(
        "--eval-noise-level",
        choices=("n0", "n1", "n2", "n3", "n4", "n5", "n6"),
        default="n0",
    )
    parser.add_argument("--annotation-noise-seed", type=int, default=2026092501)
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Optional front+wrist MP4 for one state and one repeat only.",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-annotate-guidance-point",
        action="store_true",
        help="Draw the policy guidance point on the exported video only.",
    )
    parser.add_argument(
        "--lamp-bulb-fsm-pos-threshold",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional lamp bulb FSM-only position threshold; benchmark success is not changed.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json_atomic(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


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


def runtime_source_hashes(repo: Path) -> dict[str, str]:
    relative_paths = (
        "scripts/evaluate_furniturebench_skill_level.py",
        "scripts/run_furniturebench_skill_level_matrix.py",
        "scripts/run_furniturebench_noisy_skill_level_matrix.py",
        "scripts/report_furniturebench_noisy_skill_level.py",
        "reports/clean_skill_level_checkpoint_registry_20260923.json",
        "reports/noisy_train_noisy_eval_checkpoint_registry_20260925.json",
        "src/eval/noisy_skill_level.py",
        "src/eval/skill_level.py",
        "src/eval/state_bank.py",
        "src/eval/skill_annotation_util.py",
        "src/eval/rollout.py",
        "src/eval/progress_schema.py",
    )
    return {relative: sha256(repo / relative) for relative in relative_paths}


def config_plain(cfg) -> dict:
    value = OmegaConf.to_container(cfg, resolve=True)
    return value if isinstance(value, dict) else {"config": value}


def _rgb_video_frame(observation, key: str) -> np.ndarray:
    value = observation[key][0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"expected a three-dimensional {key} frame, got {image.shape}")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0 + 1e-6:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _combined_video_frame(observation) -> np.ndarray:
    front = _rgb_video_frame(observation, "color_image1")
    wrist = _rgb_video_frame(observation, "color_image2")
    height = max(front.shape[0], wrist.shape[0])

    def pad(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == height:
            return image
        return np.pad(image, ((0, height - image.shape[0]), (0, 0), (0, 0)))

    return np.concatenate((pad(front), pad(wrist)), axis=1)


def main() -> int:
    args = parse_args()
    if args.n_envs <= 0 or args.repeats <= 0:
        raise ValueError("n-envs and repeats must be positive")
    if len(args.repeat_seeds) != args.repeats:
        raise ValueError("repeat-seeds must contain exactly repeats values")
    if args.video_fps <= 0:
        raise ValueError("video-fps must be positive")
    noise_std_m = {
        "n0": 0.0,
        "n1": 0.003,
        "n2": 0.006,
        "n3": 0.012,
        "n4": 0.024,
        "n5": 0.048,
        "n6": 0.096,
    }[args.eval_noise_level]
    annotation_noise_config = make_annotation_noise_config(
        pos_std_m=noise_std_m,
        ori_std_deg=0.0,
        seed=args.annotation_noise_seed,
        mode="gaussian_clip_2sigma",
        apply_to="point",
    )
    if args.lamp_bulb_fsm_pos_threshold is not None:
        values = tuple(float(value) for value in args.lamp_bulb_fsm_pos_threshold)
        if any(value <= 0 for value in values):
            raise ValueError("lamp-bulb-fsm-pos-threshold values must be positive")
        requested = ",".join(f"{value:g}" for value in values)
        inherited = os.environ.get("RR_LAMP_BULB_FSM_POS_THRESHOLD")
        if inherited is not None and inherited != requested:
            raise ValueError(
                "RR_LAMP_BULB_FSM_POS_THRESHOLD conflicts with the requested "
                "--lamp-bulb-fsm-pos-threshold"
            )
        os.environ["RR_LAMP_BULB_FSM_POS_THRESHOLD"] = requested

    checkpoint = args.checkpoint.expanduser().resolve()
    bank = args.state_bank_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    campaign = json.loads((bank / "campaign.json").read_text(encoding="utf-8"))
    manifest = read_jsonl(bank / "manifest.jsonl")
    if args.state_start < 0:
        raise ValueError("state-start must be non-negative")
    if args.state_limit is not None and args.state_index is not None:
        raise ValueError("state-limit and state-index are mutually exclusive")
    if args.state_index is not None and args.state_start != 0:
        raise ValueError("state-start and state-index are mutually exclusive")
    if args.state_limit is not None:
        if args.state_limit <= 0:
            raise ValueError("state-limit must be positive")
        manifest = manifest[args.state_start : args.state_start + args.state_limit]
    elif args.state_start:
        manifest = manifest[args.state_start :]
    elif args.state_index is not None:
        if args.state_index < 0 or args.state_index >= len(manifest):
            raise ValueError(
                f"state-index must be in [0, {len(manifest) - 1}], "
                f"got {args.state_index}"
            )
        manifest = [manifest[args.state_index]]
    if not manifest:
        raise ValueError("state bank is empty")
    if args.video_output is not None and (len(manifest) != 1 or args.repeats != 1):
        raise ValueError("--video-output requires exactly one state and one repeat")
    if campaign.get("annotation_source") != "scripted":
        raise ValueError("state bank must have scripted provenance")
    task = str(campaign["task"])
    stage = str(campaign["stage"])
    if any(str(row.get("task")) != task or str(row.get("skill_state")) != stage for row in manifest):
        raise ValueError("manifest task/stage does not match state-bank campaign")
    for row in manifest:
        path = bank / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"state hash mismatch: {path}")
    records = [load_state_record(bank / row["path"]) for row in manifest]

    device = torch.device(f"cuda:{args.gpu}")
    payload = torch.load(checkpoint, map_location=device)
    cfg = OmegaConf.create(payload["config"])
    validate_annotation_config(cfg)
    action_type = str(cfg.control.control_mode)
    if action_type not in {"pos", "delta"}:
        raise ValueError(f"unsupported checkpoint action type: {action_type}")
    actor = get_actor(cfg=cfg, device=device)
    if args.wrist_image_transform == "center-crop-224":
        actor.camera1_transform.spatial_transform = CENTER_CROP_224_SPATIAL_TRANSFORM
    elif args.wrist_image_transform == "legacy-resize":
        actor.camera1_transform.spatial_transform = LEGACY_224_SPATIAL_TRANSFORM
    resolved_transform = str(actor.camera1_transform.spatial_transform)
    if isinstance(actor, DiffusionPolicy):
        actor.inference_steps = int(args.inference_steps)
    actor.load_state_dict(_extract_model_state_dict(payload))
    actor.eval()
    actor.to(device)
    actor.set_task(task2idx[task])

    observation_type = str(cfg.get("observation_type", ""))
    vision_encoder = cfg.get("vision_encoder", {})
    vision_model = str(vision_encoder.get("model", "")) if hasattr(vision_encoder, "get") else ""
    depth_positive_meters = (
        observation_type.lower() == "rgbd" or vision_model.lower() == "resnet18_rgbd"
    )
    obs_keys = list(FULL_OBS) if depth_positive_meters else None
    if obs_keys is not None:
        for key in ("depth_image1", "depth_image2"):
            if key not in obs_keys:
                obs_keys.append(key)

    uses_gp = bool(model_uses_guidance_point(cfg))
    uses_gp_colored = bool(model_uses_guidance_point_colored(cfg))
    uses_grasp = bool(model_uses_grasp(cfg))
    uses_grasp_colored = bool(model_uses_grasp_colored(cfg))
    uses_grasp_part = bool(model_uses_grasp_part(cfg))
    requires_skill = bool(model_requires_skill_input(cfg))
    metric_type = "pose" if uses_grasp or uses_grasp_part else "position"
    calibrated_steps, timeout_calibration = calibrated_stage_timeout(stage, manifest)
    if args.max_stage_steps is None:
        max_steps = calibrated_steps
    else:
        max_steps = int(args.max_stage_steps)
        timeout_calibration = {
            **timeout_calibration,
            "method": "explicit_cli_override",
            "resolved_steps": max_steps,
        }

    env = get_rl_env(
        gpu_id=args.gpu,
        task=task,
        num_envs=args.n_envs,
        randomness=args.randomness,
        observation_space="image",
        max_env_steps=5_000,
        resize_img=False,
        act_rot_repr=str(cfg.control.act_rot_repr),
        action_type=action_type,
        april_tags=False,
        verbose=False,
        headless=True,
        obs_keys=obs_keys,
        depth_positive_meters=depth_positive_meters,
    )

    run_contract = {
        "schema": "rr-noisy-skill-level-v1",
        "condition": args.condition,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_config": config_plain(cfg),
        "git_commit": git_commit(),
        "git_repository_state": git_repo_state(Path.cwd()),
        "furniture_bench_repository_state": git_repo_state(
            Path.cwd() / "furniture-bench"
        ),
        "runtime_source_sha256": runtime_source_hashes(Path.cwd()),
        "command": [sys.executable, *sys.argv],
        "task": task,
        "skill_stage": stage,
        "state_bank": str(bank),
        "state_count": len(manifest),
        "state_selection": {
            "state_start": args.state_start,
            "state_limit": args.state_limit,
            "state_index": args.state_index,
        },
        "state_manifest_sha256": sha256(bank / "manifest.jsonl"),
        "repeats": int(args.repeats),
        "repeat_seeds": list(args.repeat_seeds),
        "rollout_seed_offset": int(args.rollout_seed_offset),
        "n_envs": int(args.n_envs),
        "max_stage_steps": max_steps,
        "stage_timeout_calibration": timeout_calibration,
        "randomness": args.randomness,
        "annotation_source": "scripted",
        "eval_noise_level": args.eval_noise_level,
        "annotation_noise": annotation_noise_config.to_dict(),
        "annotation_noise_pairing": {
            "key": "state_sha256+repeat_seed",
            "shared_standard_noise_across_levels": True,
        },
        "skill_completion_authority": "scripted_fsm_forward_transition",
        "lamp_bulb_fsm_position_threshold": args.lamp_bulb_fsm_pos_threshold,
        "metric_type": metric_type,
        "action_type": action_type,
        "act_rot_repr": str(cfg.control.act_rot_repr),
        "eepose_frame": ROBOT_BASE,
        "depth_positive_meters": depth_positive_meters,
        "requested_wrist_image_transform": args.wrist_image_transform,
        "resolved_wrist_image_transform": resolved_transform,
        "policy_condition": {
            "requires_skill_input": requires_skill,
            "guidance_point": uses_gp,
            "guidance_point_colored": uses_gp_colored,
            "grasp": uses_grasp,
            "grasp_colored": uses_grasp_colored,
            "grasp_part": uses_grasp_part,
        },
        "restore_protocol": {
            "mode": "pinned_gripper_settle",
            "restore_velocity": "zeroed",
            "settle_steps": 3,
            "part_position_max_abs_m": RESTORE_PART_POSITION_TOLERANCE_M,
            "part_orientation_max_abs_deg": RESTORE_PART_ORIENTATION_TOLERANCE_DEG,
            "arm_dof_position_max_abs": RESTORE_ARM_DOF_POSITION_TOLERANCE,
            "gripper_dof_position_max_abs": RESTORE_GRIPPER_DOF_POSITION_TOLERANCE,
        },
        "video": (
            None
            if args.video_output is None
            else {
                "path": str(args.video_output.expanduser().resolve()),
                "fps": int(args.video_fps),
                "annotate_guidance_point": bool(
                    args.video_annotate_guidance_point
                ),
            }
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "run.json"
    records_path = output / "records.jsonl"
    if contract_path.exists():
        if not args.resume:
            raise FileExistsError(f"output exists; pass --resume: {output}")
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        stable_keys = (
            "condition",
            "checkpoint_sha256",
            "task",
            "skill_stage",
            "state_manifest_sha256",
            "repeats",
            "repeat_seeds",
            "n_envs",
            "max_stage_steps",
            "metric_type",
            "resolved_wrist_image_transform",
            "furniture_bench_repository_state",
            "runtime_source_sha256",
            "restore_protocol",
            "skill_completion_authority",
            "lamp_bulb_fsm_position_threshold",
            "eval_noise_level",
            "annotation_noise",
            "annotation_noise_pairing",
        )
        if any(existing_contract.get(key) != run_contract.get(key) for key in stable_keys):
            raise ValueError("resume contract does not match existing output")
    else:
        write_json_atomic(contract_path, run_contract)

    existing = read_jsonl(records_path) if records_path.exists() else []
    completed_keys = {
        (str(row["state_sha256"]), int(row["repeat_index"])) for row in existing
    }
    all_rows = list(existing)
    video_writer = None
    frame_callback = None
    if args.video_output is not None:
        import imageio.v2 as imageio

        video_path = args.video_output.expanduser().resolve()
        if video_path.exists():
            raise FileExistsError(f"video output already exists: {video_path}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=int(args.video_fps))

        def frame_callback(step_idx, observation, annotation_bundles):
            del step_idx
            video_observation = observation
            if args.video_annotate_guidance_point:
                video_observation = deepcopy(observation)
                _draw_guidance_points_for_all_envs(
                    video_observation,
                    annotation_bundles,
                    annotate_wrist_camera=True,
                    guidance_point_colored=uses_gp_colored,
                )
            video_writer.append_data(_combined_video_frame(video_observation))
    for repeat_idx, repeat_seed in enumerate(args.repeat_seeds):
        for batch_idx, start in enumerate(range(0, len(records), args.n_envs)):
            source_records = records[start : start + args.n_envs]
            source_entries = manifest[start : start + args.n_envs]
            pending = [
                (str(entry["sha256"]), repeat_idx) not in completed_keys
                for entry in source_entries
            ]
            if not any(pending):
                continue
            if not all(pending):
                raise RuntimeError(
                    "partial batch found during resume; remove the incomplete cell output"
                )
            accepted_count = len(source_records)
            while len(source_records) < args.n_envs:
                source_records.append(source_records[-1])
                source_entries.append(source_entries[-1])
            batch_seed = (
                int(repeat_seed) * 1_000_003
                + int(batch_idx)
                + int(args.rollout_seed_offset)
            )
            noise_seed_offsets = [
                stable_annotation_noise_seed_offset(str(entry["sha256"]), repeat_seed)
                for entry in source_entries
            ]
            rows = rollout_skill_stage_batch(
                env=env,
                actor=actor,
                records=source_records,
                manifest_entries=source_entries,
                accepted_count=accepted_count,
                rollout_seed=batch_seed,
                max_steps=max_steps,
                metric_type=metric_type,
                annotate_guidance_point=uses_gp,
                annotate_grasp=uses_grasp,
                grasp_part_annotate=uses_grasp_part,
                guidance_point_colored=uses_gp_colored,
                grasp_annotation_colored=uses_grasp_colored,
                annotation_noise_config=annotation_noise_config,
                annotation_noise_seed_offsets=noise_seed_offsets,
                eepose_frame=ROBOT_BASE,
                gripper_settle_steps=3,
                frame_callback=frame_callback,
            )
            for row in rows:
                row.update(
                    {
                        "schema": "rr-noisy-skill-level-record-v1",
                        "condition": args.condition,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": run_contract["checkpoint_sha256"],
                        "repeat_index": repeat_idx,
                        "repeat_seed": int(repeat_seed),
                        "batch_index": batch_idx,
                        "eval_noise_level": args.eval_noise_level,
                    }
                )
            with records_path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            all_rows.extend(rows)
            completed_keys.update(
                (str(row["state_sha256"]), int(row["repeat_index"])) for row in rows
            )
            summary = {
                **summarize_skill_level_records(all_rows),
                "schema": "rr-noisy-skill-level-summary-v1",
                "condition": args.condition,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": run_contract["checkpoint_sha256"],
                "task": task,
                "skill_stage": stage,
                "metric_type": metric_type,
                "eval_noise_level": args.eval_noise_level,
                "expected_attempts": len(manifest) * args.repeats,
                "complete": len(all_rows) == len(manifest) * args.repeats,
            }
            write_json_atomic(output / "summary.json", summary)
            print(
                f"progress={len(all_rows)}/{summary['expected_attempts']} "
                f"success={summary['completed']}/{summary['attempted']}",
                flush=True,
            )

    if video_writer is not None:
        video_writer.close()

    final_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if not final_summary.get("complete"):
        raise RuntimeError("skill-level cell ended incomplete")
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
