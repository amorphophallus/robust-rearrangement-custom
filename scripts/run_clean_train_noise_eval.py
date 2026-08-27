from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.generate_annotation_noise_report import generate_report


@dataclass(frozen=True)
class ConditionConfig:
    condition_id: str
    condition: str
    family: str
    checkpoint: Path
    flags: tuple[str, ...]
    apply_to: str
    data_suffix: str


@dataclass(frozen=True)
class NoiseLevel:
    noise_id: str
    noise_label: str
    pos_std_m: float
    ori_std_deg: float
    perturbation: str = "gaussian"


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "bc" / "one_leg+round_table+lamp" / "low"
AUTO_EVAL_DEFAULT = Path("/home/huyue/projects/gpu-snatcher/auto_eval.sh")
MANIFEST_DEFAULT = REPO_ROOT / "logs" / "annotation_noise_clean_train_fresh36_manifest.jsonl"
REPORT_DEFAULT = REPO_ROOT / "reports" / "annotation_noise_clean_train_fresh36.md"
FIGURES_DEFAULT = REPO_ROOT / "reports" / "figures" / "fresh36"
DATA_DEFAULT = REPO_ROOT / "reports" / "data" / "fresh36"
GROUP_LOGS_DEFAULT = REPO_ROOT / "logs" / "annotation_noise_clean_train_fresh36_groups"
GUIDANCE_BANK_DEFAULT = REPO_ROOT / "logs" / "annotation_noise_guidance_bank"

CONDITIONS = [
    ConditionConfig(
        condition_id="gp",
        condition="rgbd+GP",
        family="point",
        checkpoint=CHECKPOINT_ROOT / "multi-task-rgbd-skill-low-0610_icy-vortex-9_latest_3000.pt",
        flags=("--annotate-skill", "--guidance-point-on-image"),
        apply_to="point",
        data_suffix="rgbd-point",
    ),
    ConditionConfig(
        condition_id="colored_gp",
        condition="rgbd+colored GP",
        family="point",
        checkpoint=CHECKPOINT_ROOT / "multi-task-rgbd-skill-low-0610_absurd-voice-2_latest_3000.pt",
        flags=(
            "--annotate-skill",
            "--guidance-point-on-image",
            "--guidance-point-colored",
        ),
        apply_to="point",
        data_suffix="rgbd-point-colored",
    ),
    ConditionConfig(
        condition_id="gp_skill",
        condition="rgbd+GP+skill",
        family="point",
        checkpoint=CHECKPOINT_ROOT / "multi-task-rgbd-skill-low-0610_fresh-tree-11_latest_3000.pt",
        flags=("--annotate-skill", "--guidance-point-on-image"),
        apply_to="point",
        data_suffix="rgbd-point",
    ),
    ConditionConfig(
        condition_id="grasp_part",
        condition="rgbd+grasp-part",
        family="grasp-part",
        checkpoint=CHECKPOINT_ROOT / "multi-task-rgbd-skill-low-grasp-annotation_morning-glitter-1_last_.pt",
        flags=("--annotate-skill", "--grasp-part-annotate"),
        apply_to="all",
        data_suffix="rgbd-grasp-part",
    ),
    ConditionConfig(
        condition_id="grasp_part_colored",
        condition="rgbd+grasp-part-colored",
        family="grasp-part",
        checkpoint=CHECKPOINT_ROOT / "multi-task-rgbd-skill-low-grasp-annotation_eternal-cosmos-2_last_.pt",
        flags=(
            "--annotate-skill",
            "--grasp-part-annotate",
            "--guidance-point-colored",
            "--grasp-annotation-colored",
        ),
        apply_to="all",
        data_suffix="rgbd-grasp-part-colored",
    ),
]

POINT_NOISE_LEVELS = [
    NoiseLevel("n0", "0mm", 0.0, 0.0),
    NoiseLevel("n1", "3mm", 0.003, 0.0),
    NoiseLevel("n2", "6mm", 0.006, 0.0),
    NoiseLevel("n3", "12mm", 0.012, 0.0),
    NoiseLevel("n4", "24mm", 0.024, 0.0),
]

GRASP_NOISE_LEVELS = [
    NoiseLevel("n0", "0mm/0deg", 0.0, 0.0),
    NoiseLevel("n1", "3mm/2.5deg", 0.003, 2.5),
    NoiseLevel("n2", "6mm/5deg", 0.006, 5.0),
    NoiseLevel("n3", "12mm/10deg", 0.012, 10.0),
    NoiseLevel("n4", "24mm/20deg", 0.024, 20.0),
]

SHUFFLED_GUIDANCE = NoiseLevel(
    "shuffle", "shuffled-guidance", 0.0, 0.0, perturbation="shuffle"
)


def _noise_levels_for_family(
    family: str, *, include_shuffled: bool = False
) -> list[NoiseLevel]:
    levels = list(POINT_NOISE_LEVELS if family == "point" else GRASP_NOISE_LEVELS)
    if include_shuffled:
        levels.append(SHUFFLED_GUIDANCE)
    return levels


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _manifest_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("condition_id")), str(row.get("noise_id")))
        current = latest.get(key)
        if current is None or str(row.get("started_at", "")) > str(
            current.get("started_at", "")
        ):
            latest[key] = row
    return latest


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _resolve_conditions(requested: str | None) -> list[ConditionConfig]:
    if not requested:
        return CONDITIONS
    requested_ids = {item.strip() for item in requested.split(",") if item.strip()}
    return [condition for condition in CONDITIONS if condition.condition_id in requested_ids]


def _resolve_noise_ids(requested: str | None) -> set[str] | None:
    if not requested:
        return None
    return {item.strip() for item in requested.split(",") if item.strip()}


def _latest_json_after(log_dir: Path, start_ts: float) -> Path | None:
    if not log_dir.exists():
        return None
    candidates = [path for path in log_dir.glob("*.json") if path.stat().st_mtime >= start_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _safe_path_part(value: str) -> str:
    safe = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", safe)
    return safe.strip("._") or "unknown"


def _task_group_log_dir(task_group: str, checkpoint_name: str) -> Path:
    return (
        REPO_ROOT
        / "logs"
        / "evaluate_model"
        / _safe_path_part(task_group)
        / _safe_path_part(checkpoint_name)
    )


def _group_log_path(
    group_logs_dir: Path,
    condition: ConditionConfig,
    noise: NoiseLevel,
) -> Path:
    return (
        group_logs_dir
        / _safe_path_part(condition.condition_id)
        / f"{_safe_path_part(noise.noise_id)}_{_safe_path_part(noise.noise_label)}.log"
    )


def _rollout_suffix_model_name(condition: ConditionConfig, noise: NoiseLevel) -> str:
    return (
        f"{_safe_path_part(condition.condition_id)}"
        f"/{_safe_path_part(noise.noise_id)}_{_safe_path_part(noise.noise_label)}"
    )


def _effective_rollout_suffix_model_name(
    condition: ConditionConfig,
    noise: NoiseLevel,
) -> str:
    suffix = _rollout_suffix_model_name(condition, noise)
    if noise.perturbation == "shuffle":
        return f"{suffix}_shuffle_seed0"
    if noise.pos_std_m <= 0.0 and noise.ori_std_deg <= 0.0:
        return suffix
    pos_tag = str(noise.pos_std_m).replace(".", "p")
    ori_tag = str(noise.ori_std_deg).replace(".", "p")
    return f"{suffix}_noise_pos{pos_tag}_ori{ori_tag}_seed0"


def _rollout_group_dirs(
    *,
    task_group: str,
    randomness: str,
    condition: ConditionConfig,
    noise: NoiseLevel,
) -> list[Path]:
    suffix = _effective_rollout_suffix_model_name(condition, noise)
    rollout_dirs = []
    for task in task_group.split("+"):
        base = (
            REPO_ROOT
            / "data"
            / "raw"
            / "diffik"
            / "sim"
            / task
            / "rollout"
            / randomness
            / condition.data_suffix
        )
        if "+" in task_group:
            base = base / task_group
        rollout_dirs.append(base / suffix)
    return rollout_dirs


def _clean_rollout_group(
    *,
    task_group: str,
    randomness: str,
    condition: ConditionConfig,
    noise: NoiseLevel,
) -> None:
    for rollout_dir in _rollout_group_dirs(
        task_group=task_group,
        randomness=randomness,
        condition=condition,
        noise=noise,
    ):
        if rollout_dir.exists():
            shutil.rmtree(rollout_dir)


def _evict_rollout_group_cache(
    *,
    task_group: str,
    randomness: str,
    condition: ConditionConfig,
    noise: NoiseLevel,
) -> dict[str, int]:
    stats = {"files": 0, "bytes": 0, "errors": 0}
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        stats["errors"] = 1
        return stats

    for rollout_dir in _rollout_group_dirs(
        task_group=task_group,
        randomness=randomness,
        condition=condition,
        noise=noise,
    ):
        if not rollout_dir.exists():
            continue
        for path in rollout_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                stats["files"] += 1
                stats["bytes"] += size
            except OSError:
                stats["errors"] += 1
    return stats


def _delete_rollout_group_pickles(
    *,
    task_group: str,
    randomness: str,
    condition: ConditionConfig,
    noise: NoiseLevel,
) -> dict[str, int]:
    stats = {"files": 0, "bytes": 0, "errors": 0}
    for rollout_dir in _rollout_group_dirs(
        task_group=task_group,
        randomness=randomness,
        condition=condition,
        noise=noise,
    ):
        if not rollout_dir.exists():
            continue
        for path in rollout_dir.rglob("*.pkl"):
            try:
                size = path.stat().st_size
                path.unlink()
                stats["files"] += 1
                stats["bytes"] += size
            except OSError:
                stats["errors"] += 1
    return stats


def _build_command(
    *,
    auto_eval_path: Path,
    task_group: str,
    checkpoint: Path,
    flags: tuple[str, ...],
    n_envs: int,
    n_rollouts: int,
    randomness: str,
    condition: ConditionConfig,
    noise: NoiseLevel,
    apply_to: str,
    save_rollouts_count: int,
    guidance_bank_dir: Path | None = None,
    guidance_bank_out_dir: Path | None = None,
) -> list[str]:
    command = [
        str(auto_eval_path),
        "--steps",
        "eval",
        "--n-envs",
        str(n_envs),
        "--n-rollouts",
        str(n_rollouts),
        "--task",
        task_group,
        "--randomness",
        randomness,
        "--overwrite-wt-path",
        str(checkpoint),
        "--rollout-suffix-model-name",
        _rollout_suffix_model_name(condition, noise),
    ]
    if save_rollouts_count > 0:
        command.extend(["--max-saved-rollouts", str(save_rollouts_count)])
    command.extend(flags)
    if guidance_bank_out_dir is not None:
        command.extend(["--guidance-bank-out-dir", str(guidance_bank_out_dir)])
    if noise.perturbation == "shuffle":
        if guidance_bank_dir is None:
            raise ValueError("shuffled guidance requires a guidance bank directory")
        command.extend(
            [
                "--annotation-shuffle-guidance",
                "--annotation-shuffle-bank",
                str(guidance_bank_dir),
                "--annotation-shuffle-seed",
                "0",
                "--noise-apply-to",
                apply_to,
            ]
        )
    elif noise.pos_std_m > 0.0 or noise.ori_std_deg > 0.0:
        command.extend(
            [
                "--noise-pos-std-m",
                str(noise.pos_std_m),
                "--noise-ori-std-deg",
                str(noise.ori_std_deg),
                "--noise-seed",
                "0",
                "--noise-mode",
                "gaussian_clip_2sigma",
                "--noise-apply-to",
                apply_to,
            ]
        )
    return command


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"summary is not a JSON object: {path}")
    return payload


def _validate_guidance_bank(bank_dir: Path, tasks: list[str]) -> None:
    issues = []
    for task in tasks:
        path = bank_dir / f"{task}.json"
        if not path.exists():
            issues.append(f"missing {path}")
            continue
        try:
            payload = _load_summary(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"invalid {path}: {exc}")
            continue
        records = payload.get("records") or []
        guidance_frame = str(payload.get("guidance_frame", "")).replace("_", "-")
        if guidance_frame != "robot-base":
            issues.append(f"non-canonical or missing guidance_frame in {path}")
        if not records:
            issues.append(f"empty {path}")
        elif any(str(record.get("task")) != task for record in records):
            issues.append(f"task mismatch in {path}")
    if issues:
        raise ValueError("invalid shuffled-guidance bank: " + "; ".join(issues))


def _validate_summary(
    *,
    summary_path: Path | None,
    condition: ConditionConfig,
    noise: NoiseLevel,
    task_group: str,
    n_envs: int,
    n_rollouts: int,
    randomness: str,
    require_tracking: bool = True,
) -> list[str]:
    errors: list[str] = []
    if summary_path is None or not summary_path.exists():
        return ["aggregate summary JSON was not created"]

    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"aggregate summary JSON is unreadable: {exc}"]

    expected_total = n_rollouts * len(task_group.split("+"))
    if int(payload.get("n_rollouts", -1)) != expected_total:
        errors.append(
            f"aggregate n_rollouts={payload.get('n_rollouts')} expected={expected_total}"
        )
    if int(payload.get("n_envs", -1)) != n_envs:
        errors.append(f"n_envs={payload.get('n_envs')} expected={n_envs}")
    expected_checkpoint_name = condition.checkpoint.stem
    if payload.get("checkpoint_name") != expected_checkpoint_name:
        errors.append(
            f"checkpoint_name={payload.get('checkpoint_name')!r} "
            f"expected={expected_checkpoint_name!r}"
        )
    expected_task_value = task_group if "+" in task_group else task_group.split("+")[0]
    actual_task_value = payload.get("task_group") or payload.get("task")
    if actual_task_value != expected_task_value:
        errors.append(
            f"task={actual_task_value!r} expected={expected_task_value!r}"
        )
    if payload.get("eval_randomness") != randomness:
        errors.append(
            f"eval_randomness={payload.get('eval_randomness')!r} expected={randomness!r}"
        )
    if payload.get("observation_space") != "image":
        errors.append(
            f"observation_space={payload.get('observation_space')!r} expected='image'"
        )
    if payload.get("action_type") != "pos":
        errors.append(f"action_type={payload.get('action_type')!r} expected='pos'")

    train_data_cfg = (payload.get("training_config") or {}).get("data") or {}
    for key in ("annotation_noise_pos_std_m", "annotation_noise_ori_std_deg"):
        try:
            train_noise = float(train_data_cfg.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            train_noise = float("nan")
        if not math.isfinite(train_noise) or train_noise != 0.0:
            errors.append(f"training_config.data.{key}={train_data_cfg.get(key)!r} expected=0")

    noise_cfg = payload.get("annotation_noise_config") or {}
    expected_enabled = (
        noise.perturbation == "shuffle"
        or noise.pos_std_m > 0.0
        or noise.ori_std_deg > 0.0
    )
    checks: dict[str, Any] = {
        "pos_std_m": noise.pos_std_m,
        "ori_std_deg": noise.ori_std_deg,
        "enabled": expected_enabled,
    }
    if noise.perturbation == "shuffle":
        checks.update(
            {
                "apply_to": condition.apply_to,
                "mode": "shuffle",
                "shuffle_seed": 0,
            }
        )
    elif expected_enabled:
        checks.update(
            {
                "apply_to": condition.apply_to,
                "mode": "gaussian_clip_2sigma",
                "seed": 0,
            }
        )
    for key, expected in checks.items():
        actual = noise_cfg.get(key)
        if isinstance(expected, float):
            try:
                matches = abs(float(actual) - expected) < 1e-9
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            errors.append(f"noise.{key}={actual!r} expected={expected!r}")

    annotation_cfg = payload.get("eval_annotation_config") or {}
    expected_flags = {
        "annotate_skill": True,
        "guidance_point_on_image": "--guidance-point-on-image" in condition.flags,
        "guidance_point_colored": "--guidance-point-colored" in condition.flags,
        "grasp_part_annotate": "--grasp-part-annotate" in condition.flags,
        "grasp_annotation_colored": "--grasp-annotation-colored" in condition.flags,
    }
    for key, expected in expected_flags.items():
        if bool(annotation_cfg.get(key)) != expected:
            errors.append(
                f"annotation.{key}={annotation_cfg.get(key)!r} expected={expected!r}"
            )

    tasks = task_group.split("+")
    per_task = payload.get("per_task") or {}
    if len(tasks) == 1 and not per_task:
        per_task = {tasks[0]: payload}
    for task in tasks:
        task_payload = per_task.get(task)
        if not isinstance(task_payload, dict):
            errors.append(f"missing per_task.{task}")
            continue
        if int(task_payload.get("n_rollouts", -1)) != n_rollouts:
            errors.append(
                f"{task}.n_rollouts={task_payload.get('n_rollouts')} expected={n_rollouts}"
            )
        if int(task_payload.get("n_envs", -1)) != n_envs:
            errors.append(
                f"{task}.n_envs={task_payload.get('n_envs')} expected={n_envs}"
            )
        if task_payload.get("eval_randomness") != randomness:
            errors.append(
                f"{task}.eval_randomness={task_payload.get('eval_randomness')!r} "
                f"expected={randomness!r}"
            )
        if require_tracking:
            tracking_payload = task_payload.get("tracking_error") or {}
            tracking = tracking_payload.get("overall") or {}
            metric_type = tracking_payload.get("metric_type", "pose")
            expected_metric_type = "position" if condition.family == "point" else "pose"
            if metric_type != expected_metric_type:
                errors.append(
                    f"{task}.tracking_error.metric_type={metric_type!r} "
                    f"expected={expected_metric_type!r}"
                )
            if condition.family == "point" and any(
                key in tracking for key in ("mean_ori_deg", "mean_total")
            ):
                errors.append(f"{task}.point tracking contains orientation/total metrics")
            if int(tracking.get("count", 0)) <= 0:
                errors.append(f"{task}.tracking_error is missing or empty")
            if int(tracking_payload.get("episode_count", -1)) != n_rollouts:
                errors.append(
                    f"{task}.tracking_error.episode_count="
                    f"{tracking_payload.get('episode_count')!r} expected={n_rollouts}"
                )
            if int(tracking_payload.get("incomplete_episode_count", -1)) != 0:
                errors.append(
                    f"{task}.tracking_error.incomplete_episode_count="
                    f"{tracking_payload.get('incomplete_episode_count')!r} expected=0"
                )
            if tracking_payload.get("complete") is not True:
                errors.append(
                    f"{task}.tracking_error.complete="
                    f"{tracking_payload.get('complete')!r} expected=True"
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-group", default="one_leg+round_table+lamp")
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=36,
        help="Fresh rollouts per task for every selected condition/noise group.",
    )
    parser.add_argument(
        "--guidance-bank-dir", type=Path, default=GUIDANCE_BANK_DEFAULT
    )
    perturbation_group = parser.add_mutually_exclusive_group()
    perturbation_group.add_argument(
        "--shuffled-only",
        action="store_true",
        help="Run only the five shuffled-guidance groups.",
    )
    perturbation_group.add_argument(
        "--include-shuffled",
        action="store_true",
        help="Run n0-n4 and shuffled guidance serially in one process.",
    )
    parser.add_argument("--randomness", default="low")
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--noise-ids", default=None)
    parser.add_argument("--auto-eval-path", type=Path, default=AUTO_EVAL_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DEFAULT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--group-logs-dir", type=Path, default=GROUP_LOGS_DEFAULT)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--save-rollouts-count", type=int, default=8)
    parser.add_argument("--keep-rollout-cache", action="store_true")
    parser.add_argument(
        "--delete-rollout-pickles",
        action="store_true",
        help=(
            "After a group passes summary validation, delete its large rollout "
            "pickle files while retaining videos and text diagnostics."
        ),
    )
    parser.add_argument(
        "--initial-min-free-disk-gib",
        type=float,
        default=500.0,
        help="Free-space threshold when the manifest has no completed groups.",
    )
    parser.add_argument(
        "--resume-min-free-disk-gib",
        type=float,
        default=80.0,
        help="Free-space threshold when resuming a partially completed manifest.",
    )
    args = parser.parse_args()

    if args.save_rollouts_count <= 0:
        raise ValueError(
            "--save-rollouts-count must be positive: auto_eval interprets zero as "
            "unlimited saving; use 8 to save only the first eight rollouts per task"
        )

    env = os.environ.copy()
    env.setdefault("DATA_DIR_RAW", str(REPO_ROOT / "data"))
    selected_conditions = _resolve_conditions(args.conditions)
    selected_noise_ids = _resolve_noise_ids(args.noise_ids)
    manifest_rows = _read_jsonl(args.manifest)
    manifest_index = _manifest_lookup(manifest_rows)
    if (args.shuffled_only or args.include_shuffled) and not args.dry_run:
        _validate_guidance_bank(args.guidance_bank_dir, args.task_group.split("+"))
    if not args.dry_run:
        free_disk_gib = shutil.disk_usage(REPO_ROOT).free / (1024**3)
        has_completed_groups = any(
            row.get("status") == "ok"
            and int(row.get("n_rollouts", -1)) == args.n_rollouts
            and int(row.get("tracking_rollouts_per_task", -1)) == args.n_rollouts
            for row in manifest_rows
        )
        min_free_disk_gib = (
            args.resume_min_free_disk_gib
            if has_completed_groups
            else args.initial_min_free_disk_gib
        )
        if free_disk_gib < min_free_disk_gib:
            raise RuntimeError(
                f"only {free_disk_gib:.1f} GiB free; "
                f"this {'resume' if has_completed_groups else 'initial run'} "
                f"requires {min_free_disk_gib:.1f} GiB"
            )

    if not args.auto_eval_path.exists():
        raise FileNotFoundError(f"Missing auto_eval script: {args.auto_eval_path}")

    for condition in selected_conditions:
        if not condition.checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {condition.checkpoint}")
        noise_levels = (
            [SHUFFLED_GUIDANCE]
            if args.shuffled_only
            else _noise_levels_for_family(
                condition.family,
                include_shuffled=args.include_shuffled,
            )
        )
        for noise in noise_levels:
            if selected_noise_ids is not None and noise.noise_id not in selected_noise_ids:
                continue
            key = (condition.condition_id, noise.noise_id)
            existing = manifest_index.get(key)
            if not args.rerun and existing is not None and existing.get("status") == "ok":
                existing_summary_value = str(existing.get("summary_json", "") or "")
                existing_summary = (
                    Path(existing_summary_value) if existing_summary_value else None
                )
                existing_errors = _validate_summary(
                    summary_path=existing_summary,
                    condition=condition,
                    noise=noise,
                    task_group=args.task_group,
                    n_envs=args.n_envs,
                    n_rollouts=args.n_rollouts,
                    randomness=args.randomness,
                )
                if not existing_errors:
                    print(
                        f"[skip] condition={condition.condition_id} "
                        f"noise={noise.noise_id} summary={existing_summary}",
                        flush=True,
                    )
                    continue
                print(
                    f"[rerun-invalid] condition={condition.condition_id} "
                    f"noise={noise.noise_id} validation={existing_errors}",
                    flush=True,
                )

            command = _build_command(
                auto_eval_path=args.auto_eval_path,
                task_group=args.task_group,
                checkpoint=condition.checkpoint,
                flags=condition.flags,
                n_envs=args.n_envs,
                n_rollouts=args.n_rollouts,
                randomness=args.randomness,
                condition=condition,
                noise=noise,
                apply_to=condition.apply_to,
                save_rollouts_count=args.save_rollouts_count,
                guidance_bank_dir=(
                    args.guidance_bank_dir
                    if args.shuffled_only or args.include_shuffled
                    else None
                ),
                guidance_bank_out_dir=(
                    args.guidance_bank_dir
                    if condition.condition_id == "gp_skill" and noise.noise_id == "n0"
                    else None
                ),
            )
            checkpoint_name = condition.checkpoint.stem
            log_dir = _task_group_log_dir(args.task_group, checkpoint_name)
            group_log = _group_log_path(args.group_logs_dir, condition, noise)
            started_at = datetime.now().isoformat(timespec="seconds")
            start_ts = datetime.now().timestamp()
            row = {
                "started_at": started_at,
                "condition_id": condition.condition_id,
                "condition": condition.condition,
                "family": condition.family,
                "noise_id": noise.noise_id,
                "noise_label": noise.noise_label,
                "pos_std_mm": noise.pos_std_m * 1000.0,
                "ori_std_deg": noise.ori_std_deg,
                "apply_to": condition.apply_to,
                "task_group": args.task_group,
                "randomness": args.randomness,
                "n_envs": args.n_envs,
                "n_rollouts": args.n_rollouts,
                "tracking_rollouts_per_task": args.n_rollouts,
                "perturbation": noise.perturbation,
                "save_rollouts_count": args.save_rollouts_count,
                "checkpoint": str(condition.checkpoint),
                "checkpoint_name": checkpoint_name,
                "command": command,
                "group_log": str(group_log),
                "status": "dry_run" if args.dry_run else "started",
            }
            print(
                f"[run] {condition.condition} {noise.noise_label} "
                f"checkpoint={checkpoint_name} log={group_log}",
                flush=True,
            )
            if args.dry_run:
                continue

            _clean_rollout_group(
                task_group=args.task_group,
                randomness=args.randomness,
                condition=condition,
                noise=noise,
            )
            group_log.parent.mkdir(parents=True, exist_ok=True)
            with group_log.open("w") as log_file:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            if not args.keep_rollout_cache:
                cache_eviction = _evict_rollout_group_cache(
                    task_group=args.task_group,
                    randomness=args.randomness,
                    condition=condition,
                    noise=noise,
                )
                row["cache_eviction"] = cache_eviction
                print(
                    f"[cache] evicted files={cache_eviction['files']} "
                    f"bytes={cache_eviction['bytes']} errors={cache_eviction['errors']}",
                    flush=True,
                )
            row["ended_at"] = datetime.now().isoformat(timespec="seconds")
            row["returncode"] = completed.returncode
            if completed.returncode == 0:
                summary_path = _latest_json_after(log_dir, start_ts)
                row["summary_json"] = str(summary_path) if summary_path else ""
                validation_errors = _validate_summary(
                    summary_path=summary_path,
                    condition=condition,
                    noise=noise,
                    task_group=args.task_group,
                    n_envs=args.n_envs,
                    n_rollouts=args.n_rollouts,
                    randomness=args.randomness,
                )
                row["validation_errors"] = validation_errors
                if validation_errors:
                    row["status"] = "failed"
                    row["returncode"] = 2
                    print(
                        f"[failed] {condition.condition_id} {noise.noise_id} "
                        f"validation={validation_errors} log={group_log}",
                        flush=True,
                    )
                else:
                    row["status"] = "ok"
                    if args.delete_rollout_pickles:
                        pickle_cleanup = _delete_rollout_group_pickles(
                            task_group=args.task_group,
                            randomness=args.randomness,
                            condition=condition,
                            noise=noise,
                        )
                        row["pickle_cleanup"] = pickle_cleanup
                        print(
                            f"[pickle-cleanup] files={pickle_cleanup['files']} "
                            f"bytes={pickle_cleanup['bytes']} "
                            f"errors={pickle_cleanup['errors']}",
                            flush=True,
                        )
                    print(
                        f"[ok] {condition.condition_id} {noise.noise_id} "
                        f"summary={row['summary_json']} log={group_log}",
                        flush=True,
                    )
            else:
                row["status"] = "failed"
                row["summary_json"] = ""
                print(
                    f"[failed] {condition.condition_id} {noise.noise_id} "
                    f"returncode={completed.returncode} log={group_log}",
                    flush=True,
                )
            _append_manifest(args.manifest, row)
            manifest_index[key] = row
            if row["status"] != "ok" and not args.continue_on_error:
                raise SystemExit(int(row.get("returncode", 1) or 1))

    if args.dry_run:
        print("[dry-run] commands printed; manifest and report unchanged", flush=True)
        return

    generate_report(
        manifest_path=args.manifest,
        report_path=args.report,
        figures_dir=args.figures_dir,
        data_dir=args.data_dir,
    )
    print(f"[done] report written to {args.report}", flush=True)


if __name__ == "__main__":
    main()
