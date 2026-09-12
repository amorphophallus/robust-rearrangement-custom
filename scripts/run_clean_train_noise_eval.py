from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
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


@dataclass(frozen=True)
class ReplicateConfig:
    replicate_id: int
    simulator_seed: int
    annotation_seed: int


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "bc" / "one_leg+round_table+lamp" / "low"
AUTO_EVAL_DEFAULT = Path("/home/huyue/projects/gpu-snatcher/auto_eval.sh")
MANIFEST_DEFAULT = REPO_ROOT / "logs" / "annotation_noise_clean_train_fresh36_manifest.jsonl"
REPORT_DEFAULT = REPO_ROOT / "reports" / "annotation_noise_clean_train_fresh36.md"
FIGURES_DEFAULT = REPO_ROOT / "reports" / "figures" / "fresh36"
DATA_DEFAULT = REPO_ROOT / "reports" / "data" / "fresh36"
GROUP_LOGS_DEFAULT = REPO_ROOT / "logs" / "annotation_noise_clean_train_fresh36_groups"
GUIDANCE_BANK_DEFAULT = REPO_ROOT / "logs" / "annotation_noise_guidance_bank"
VLM_COVER_MANIFEST_DEFAULT = REPO_ROOT / "logs" / "annotation_noise_vlm_cover_108" / "manifest.jsonl"
VLM_COVER_REPORT_DEFAULT = REPO_ROOT / "reports" / "annotation_noise_vlm_cover_108.md"
VLM_COVER_FIGURES_DEFAULT = REPO_ROOT / "reports" / "figures" / "vlm_cover_108"
VLM_COVER_DATA_DEFAULT = REPO_ROOT / "reports" / "data" / "vlm_cover_108"
VLM_COVER_GROUPS_DEFAULT = REPO_ROOT / "logs" / "annotation_noise_vlm_cover_108" / "groups"

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
    NoiseLevel("n5", "48mm", 0.048, 0.0),
    NoiseLevel("n6", "96mm", 0.096, 0.0),
    NoiseLevel("n7", "192mm", 0.192, 0.0),
]

GRASP_NOISE_LEVELS = [
    NoiseLevel("n0", "0mm/0deg", 0.0, 0.0),
    NoiseLevel("n1", "3mm/2.5deg", 0.003, 2.5),
    NoiseLevel("n2", "6mm/5deg", 0.006, 5.0),
    NoiseLevel("n3", "12mm/10deg", 0.012, 10.0),
    NoiseLevel("n4", "24mm/20deg", 0.024, 20.0),
    NoiseLevel("n5", "48mm/40deg", 0.048, 40.0),
    NoiseLevel("n6", "96mm/60deg", 0.096, 60.0),
    NoiseLevel("n7", "192mm/90deg", 0.192, 90.0),
]

FIXED_R180 = NoiseLevel(
    "r180", "orientation-only-180deg", 0.0, 180.0,
    perturbation="fixed_geodesic",
)

SHUFFLED_GUIDANCE = NoiseLevel(
    "shuffle", "shuffled-guidance", 0.0, 0.0, perturbation="shuffle"
)


def _noise_levels_for_family(
    family: str, *, include_shuffled: bool = False, legacy_only: bool = True
) -> list[NoiseLevel]:
    levels = list(POINT_NOISE_LEVELS if family == "point" else GRASP_NOISE_LEVELS)
    if legacy_only:
        levels = levels[:5]
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


def _manifest_key(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    annotation_seed = row.get("annotation_seed")
    if annotation_seed is None:
        annotation_seed = row.get("shuffle_seed", 0)
    return (
        str(row.get("condition_id")),
        str(row.get("noise_id")),
        int(row.get("replicate_id", 0)),
        int(row.get("simulator_seed", 0)),
        int(annotation_seed or 0),
    )


def _manifest_lookup(
    rows: list[dict[str, Any]], *, replicate_aware: bool = False
) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            _manifest_key(row)
            if replicate_aware
            else (str(row.get("condition_id")), str(row.get("noise_id")))
        )
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


def _upsert_manifest(path: Path, row: dict[str, Any]) -> None:
    """Keep the profile manifest unique even after retrying a failed key."""
    key = _manifest_key(row)
    rows = [item for item in _read_jsonl(path) if _manifest_key(item) != key]
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows)
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_provenance() -> dict[str, Any]:
    source_paths = (
        REPO_ROOT / "scripts" / "run_clean_train_noise_eval.py",
        REPO_ROOT / "scripts" / "audit_clean_train_noise_eval.py",
        REPO_ROOT / "scripts" / "generate_annotation_noise_report.py",
        REPO_ROOT / "scripts" / "generate_vlm_cover_108_report.py",
        REPO_ROOT / "src" / "eval" / "annotation_noise.py",
        REPO_ROOT / "src" / "eval" / "evaluate_model.py",
        REPO_ROOT / "src" / "eval" / "rollout.py",
        REPO_ROOT / "src" / "eval" / "skill_annotation_util.py",
    )
    return {
        "hostname": os.uname().nodename,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": (_command_output(["git", "status", "--short"]) or "clean"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in source_paths
            if path.exists()
        },
    }


def _resolve_conditions(requested: str | None) -> list[ConditionConfig]:
    if not requested:
        return CONDITIONS
    requested_ids = {item.strip() for item in requested.split(",") if item.strip()}
    return [condition for condition in CONDITIONS if condition.condition_id in requested_ids]


def _resolve_noise_ids(requested: str | None) -> set[str] | None:
    if not requested:
        return None
    return {item.strip() for item in requested.split(",") if item.strip()}


def _resolve_replicate_ids(requested: str | None) -> set[int] | None:
    if not requested:
        return None
    values = {int(item.strip()) for item in requested.split(",") if item.strip()}
    unknown = values - {0, 1, 2}
    if unknown:
        raise ValueError(f"unsupported replicate ids: {sorted(unknown)}")
    return values


def _vlm_cover_schedule(
    conditions: list[ConditionConfig],
) -> list[tuple[ConditionConfig, NoiseLevel, ReplicateConfig]]:
    """Return the fixed 111-invocation order, excluding reused legacy cells."""
    replicates = {seed: ReplicateConfig(seed, seed, seed) for seed in (0, 1, 2)}
    scheduled: list[tuple[ConditionConfig, NoiseLevel, ReplicateConfig]] = []
    for condition in conditions:
        for noise in _noise_levels_for_family(condition.family, legacy_only=False)[5:]:
            scheduled.append((condition, noise, replicates[0]))
    for seed in (1, 2):
        for condition in conditions:
            for noise in _noise_levels_for_family(condition.family, legacy_only=False):
                scheduled.append((condition, noise, replicates[seed]))
    for seed in (0, 1, 2):
        for condition in conditions:
            if condition.family == "grasp-part":
                scheduled.append((condition, FIXED_R180, replicates[seed]))
    for seed in (1, 2):
        for condition in conditions:
            scheduled.append((condition, SHUFFLED_GUIDANCE, replicates[seed]))
    return scheduled


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
    replicate: ReplicateConfig | None = None,
) -> Path:
    seed_suffix = ""
    if replicate is not None:
        seed_suffix = (
            f"_rep{replicate.replicate_id}_sim{replicate.simulator_seed}"
            f"_ann{replicate.annotation_seed}"
        )
    return (
        group_logs_dir
        / _safe_path_part(condition.condition_id)
        / f"{_safe_path_part(noise.noise_id)}_{_safe_path_part(noise.noise_label)}{seed_suffix}.log"
    )


def _group_summary_path(
    group_logs_dir: Path,
    condition: ConditionConfig,
    noise: NoiseLevel,
    replicate: ReplicateConfig,
) -> Path:
    return (
        group_logs_dir
        / _safe_path_part(condition.condition_id)
        / (
            f"{_safe_path_part(noise.noise_id)}_{_safe_path_part(noise.noise_label)}"
            f"_rep{replicate.replicate_id}_sim{replicate.simulator_seed}"
            f"_ann{replicate.annotation_seed}.summary.json"
        )
    )


def _group_command_metadata_path(
    group_logs_dir: Path,
    condition: ConditionConfig,
    noise: NoiseLevel,
    replicate: ReplicateConfig,
) -> Path:
    return _group_summary_path(
        group_logs_dir, condition, noise, replicate
    ).with_suffix(".command.json")


def _rollout_suffix_model_name(
    condition: ConditionConfig,
    noise: NoiseLevel,
    replicate: ReplicateConfig | None = None,
) -> str:
    suffix = (
        f"{_safe_path_part(condition.condition_id)}"
        f"/{_safe_path_part(noise.noise_id)}_{_safe_path_part(noise.noise_label)}"
    )
    if replicate is not None:
        suffix += (
            f"/rep{replicate.replicate_id}_sim{replicate.simulator_seed}"
            f"_ann{replicate.annotation_seed}"
        )
    return suffix


def _effective_rollout_suffix_model_name(
    condition: ConditionConfig,
    noise: NoiseLevel,
    replicate: ReplicateConfig | None = None,
) -> str:
    if replicate is not None:
        return _rollout_suffix_model_name(condition, noise, replicate)
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
    replicate: ReplicateConfig | None = None,
) -> list[Path]:
    suffix = _effective_rollout_suffix_model_name(condition, noise, replicate)
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
    replicate: ReplicateConfig | None = None,
) -> None:
    for rollout_dir in _rollout_group_dirs(
        task_group=task_group,
        randomness=randomness,
        condition=condition,
        noise=noise,
        replicate=replicate,
    ):
        if rollout_dir.exists():
            shutil.rmtree(rollout_dir)


def _evict_rollout_group_cache(
    *,
    task_group: str,
    randomness: str,
    condition: ConditionConfig,
    noise: NoiseLevel,
    replicate: ReplicateConfig | None = None,
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
        replicate=replicate,
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
    replicate: ReplicateConfig | None = None,
) -> dict[str, int]:
    stats = {"files": 0, "bytes": 0, "errors": 0}
    for rollout_dir in _rollout_group_dirs(
        task_group=task_group,
        randomness=randomness,
        condition=condition,
        noise=noise,
        replicate=replicate,
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
    replicate: ReplicateConfig | None = None,
) -> list[str]:
    explicit_replicate = replicate is not None
    replicate = replicate or ReplicateConfig(0, 0, 0)
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
        _rollout_suffix_model_name(
            condition, noise, replicate if explicit_replicate else None
        ),
        "--seed",
        str(replicate.simulator_seed),
        "--annotation-source",
        "scripted",
    ]
    # auto_eval historically treats the literal string "0" as "omit the flag".
    # "00" is the same integer while ensuring the evaluator receives the explicit
    # no-persistence setting required by the vlm-cover profile.
    saved_count_arg = "00" if save_rollouts_count == 0 else str(save_rollouts_count)
    command.extend(["--max-saved-rollouts", saved_count_arg])
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
                str(replicate.annotation_seed),
                "--noise-apply-to",
                apply_to,
            ]
        )
    else:
        command.extend(
            [
                "--noise-pos-std-m",
                str(noise.pos_std_m),
                "--noise-ori-std-deg",
                str(noise.ori_std_deg),
                "--noise-seed",
                str(replicate.annotation_seed),
                "--noise-mode",
                (
                    "fixed_geodesic"
                    if noise.perturbation == "fixed_geodesic"
                    else "gaussian_clip_2sigma"
                ),
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
    replicate: ReplicateConfig | None = None,
    expected_saved_rollouts: int | None = None,
    require_noise_stats: bool = False,
) -> list[str]:
    errors: list[str] = []
    if summary_path is None or not summary_path.exists():
        return ["aggregate summary JSON was not created"]

    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"aggregate summary JSON is unreadable: {exc}"]
    explicit_replicate = replicate is not None
    replicate = replicate or ReplicateConfig(0, 0, 0)

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
    if payload.get("annotation_source") != "scripted":
        errors.append(
            f"annotation_source={payload.get('annotation_source')!r} expected='scripted'"
        )
    if int(payload.get("simulator_seed", -1)) != replicate.simulator_seed:
        errors.append(
            f"simulator_seed={payload.get('simulator_seed')!r} "
            f"expected={replicate.simulator_seed}"
        )

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
                "shuffle_seed": replicate.annotation_seed,
            }
        )
    elif expected_enabled or explicit_replicate:
        checks.update(
            {
                "apply_to": condition.apply_to,
                "mode": (
                    "fixed_geodesic"
                    if noise.perturbation == "fixed_geodesic"
                    else "gaussian_clip_2sigma"
                ),
                "seed": replicate.annotation_seed,
            }
        )
        if noise.perturbation == "fixed_geodesic":
            checks["target_geodesic_deg"] = noise.ori_std_deg
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
        if int(task_payload.get("simulator_seed", -1)) != replicate.simulator_seed:
            errors.append(
                f"{task}.simulator_seed={task_payload.get('simulator_seed')!r} "
                f"expected={replicate.simulator_seed}"
            )
        if expected_saved_rollouts is not None:
            if int(task_payload.get("n_saved_rollouts", -1)) != expected_saved_rollouts:
                errors.append(
                    f"{task}.n_saved_rollouts={task_payload.get('n_saved_rollouts')!r} "
                    f"expected={expected_saved_rollouts}"
                )
        if require_noise_stats:
            stats = task_payload.get("annotation_noise_stats") or {}
            if int(stats.get("phase_count", 0)) <= 0:
                errors.append(f"{task}.annotation_noise_stats is missing or empty")
            for key in (
                "position_norm_m",
                "rotation_geodesic_deg",
                "workspace_valid_rate",
                "front_projection_visible_rate",
                "invalid_nonfinite_rate",
            ):
                if key not in stats:
                    errors.append(f"{task}.annotation_noise_stats.{key} is missing")
            if noise.perturbation == "fixed_geodesic":
                samples = stats.get("phase_samples") or []
                realized = [
                    float(sample.get("realized_ori_geodesic_deg"))
                    for sample in samples
                    if sample.get("apply_ori") is True
                ]
                if not realized:
                    errors.append(f"{task}.r180 has no applied orientation samples")
                elif any(abs(value - noise.ori_std_deg) > 1e-3 for value in realized):
                    errors.append(
                        f"{task}.r180 realized geodesic is not {noise.ori_std_deg}±1e-3"
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
            if require_noise_stats and tracking_payload.get("target_source") != (
                "scripted_displayed_annotation"
            ):
                errors.append(
                    f"{task}.tracking_error.target_source="
                    f"{tracking_payload.get('target_source')!r} "
                    "expected='scripted_displayed_annotation'"
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["legacy-fresh36", "vlm-cover-108"],
        default="legacy-fresh36",
    )
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
    parser.add_argument(
        "--replicates",
        default=None,
        help="Comma-separated replicate ids (0,1,2); mainly for resume and pilots.",
    )
    parser.add_argument("--auto-eval-path", type=Path, default=AUTO_EVAL_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--legacy-manifest", type=Path, default=MANIFEST_DEFAULT,
        help="Read-only seed-0 source used only by the vlm-cover report/audit.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--group-logs-dir", type=Path, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--save-rollouts-count", type=int, default=None)
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
        default=None,
        help="Free-space threshold when the manifest has no completed groups.",
    )
    parser.add_argument(
        "--resume-min-free-disk-gib",
        type=float,
        default=80.0,
        help="Free-space threshold when resuming a partially completed manifest.",
    )
    args = parser.parse_args()

    is_vlm_cover = args.profile == "vlm-cover-108"
    args.manifest = args.manifest or (
        VLM_COVER_MANIFEST_DEFAULT if is_vlm_cover else MANIFEST_DEFAULT
    )
    args.report = args.report or (
        VLM_COVER_REPORT_DEFAULT if is_vlm_cover else REPORT_DEFAULT
    )
    args.figures_dir = args.figures_dir or (
        VLM_COVER_FIGURES_DEFAULT if is_vlm_cover else FIGURES_DEFAULT
    )
    args.data_dir = args.data_dir or (
        VLM_COVER_DATA_DEFAULT if is_vlm_cover else DATA_DEFAULT
    )
    args.group_logs_dir = args.group_logs_dir or (
        VLM_COVER_GROUPS_DEFAULT if is_vlm_cover else GROUP_LOGS_DEFAULT
    )
    if args.save_rollouts_count is None:
        args.save_rollouts_count = 0 if is_vlm_cover else 8
    if args.save_rollouts_count < 0:
        raise ValueError("--save-rollouts-count must be non-negative")
    if args.initial_min_free_disk_gib is None:
        args.initial_min_free_disk_gib = 150.0 if is_vlm_cover else 500.0
    if is_vlm_cover and (args.shuffled_only or args.include_shuffled):
        raise ValueError(
            "vlm-cover-108 has a fixed Shuffle schedule; filter it with --noise-ids"
        )

    env = os.environ.copy()
    env.setdefault("DATA_DIR_RAW", str(REPO_ROOT / "data"))
    selected_conditions = _resolve_conditions(args.conditions)
    selected_noise_ids = _resolve_noise_ids(args.noise_ids)
    selected_replicates = _resolve_replicate_ids(args.replicates)
    manifest_rows = _read_jsonl(args.manifest)
    manifest_index = _manifest_lookup(manifest_rows, replicate_aware=is_vlm_cover)
    if (
        args.shuffled_only
        or args.include_shuffled
        or (is_vlm_cover and (selected_noise_ids is None or "shuffle" in selected_noise_ids))
    ) and not args.dry_run:
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

    if is_vlm_cover:
        execution_items: list[
            tuple[ConditionConfig, NoiseLevel, ReplicateConfig | None]
        ] = list(_vlm_cover_schedule(selected_conditions))
    else:
        execution_items = []
        for condition in selected_conditions:
            noise_levels = (
                [SHUFFLED_GUIDANCE]
                if args.shuffled_only
                else _noise_levels_for_family(
                    condition.family,
                    include_shuffled=args.include_shuffled,
                )
            )
            execution_items.extend((condition, noise, None) for noise in noise_levels)

    execution_items = [
        (condition, noise, replicate)
        for condition, noise, replicate in execution_items
        if (selected_noise_ids is None or noise.noise_id in selected_noise_ids)
        and (
            selected_replicates is None
            or replicate is None
            or replicate.replicate_id in selected_replicates
        )
    ]

    checkpoint_hashes: dict[Path, str] = {}
    for condition in selected_conditions:
        if not condition.checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {condition.checkpoint}")
        checkpoint_hashes[condition.checkpoint] = _sha256(condition.checkpoint)
    run_provenance = _run_provenance()

    for condition, noise, replicate in execution_items:
            key = (
                _manifest_key(
                    {
                        "condition_id": condition.condition_id,
                        "noise_id": noise.noise_id,
                        "replicate_id": replicate.replicate_id,
                        "simulator_seed": replicate.simulator_seed,
                        "annotation_seed": replicate.annotation_seed,
                    }
                )
                if replicate is not None
                else (condition.condition_id, noise.noise_id)
            )
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
                    replicate=replicate,
                    expected_saved_rollouts=(0 if is_vlm_cover else None),
                    require_noise_stats=is_vlm_cover,
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
                    if noise.perturbation == "shuffle"
                    else None
                ),
                guidance_bank_out_dir=(
                    args.guidance_bank_dir
                    if (
                        not is_vlm_cover
                        and condition.condition_id == "gp_skill"
                        and noise.noise_id == "n0"
                    )
                    else None
                ),
                replicate=replicate,
            )
            checkpoint_name = condition.checkpoint.stem
            log_dir = _task_group_log_dir(args.task_group, checkpoint_name)
            group_log = _group_log_path(
                args.group_logs_dir, condition, noise, replicate
            )
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
                "checkpoint_sha256": checkpoint_hashes[condition.checkpoint],
                "checkpoint_name": checkpoint_name,
                "profile": args.profile,
                "replicate_id": replicate.replicate_id if replicate else 0,
                "simulator_seed": replicate.simulator_seed if replicate else 0,
                "annotation_seed": replicate.annotation_seed if replicate else 0,
                "seed_kind": (
                    "shuffle_seed" if noise.perturbation == "shuffle" else "noise_seed"
                ),
                "resolved_annotation_source": "scripted",
                "run_provenance": run_provenance,
                "command": command,
                "group_log": str(group_log),
                "status": "dry_run" if args.dry_run else "started",
            }
            print(
                f"[run] {condition.condition} {noise.noise_label} "
                f"checkpoint={checkpoint_name} log={group_log}",
                flush=True,
            )
            print(f"[command] {shlex.join(command)}", flush=True)
            if args.dry_run:
                continue

            if is_vlm_cover:
                assert replicate is not None
                command_metadata_path = _group_command_metadata_path(
                    args.group_logs_dir, condition, noise, replicate
                )
                command_metadata_path.parent.mkdir(parents=True, exist_ok=True)
                command_metadata_path.write_text(
                    json.dumps(
                        {
                            "recorded_at": datetime.now().isoformat(timespec="seconds"),
                            "profile": args.profile,
                            "condition_id": condition.condition_id,
                            "noise_id": noise.noise_id,
                            "replicate_id": replicate.replicate_id,
                            "simulator_seed": replicate.simulator_seed,
                            "annotation_seed": replicate.annotation_seed,
                            "resolved_annotation_source": "scripted",
                            "checkpoint": str(condition.checkpoint),
                            "checkpoint_sha256": checkpoint_hashes[condition.checkpoint],
                            "run_provenance": run_provenance,
                            "command": command,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                row["command_metadata"] = str(command_metadata_path)

            _clean_rollout_group(
                task_group=args.task_group,
                randomness=args.randomness,
                condition=condition,
                noise=noise,
                replicate=replicate,
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
                    replicate=replicate,
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
                if is_vlm_cover and summary_path is not None:
                    assert replicate is not None
                    stable_summary_path = _group_summary_path(
                        args.group_logs_dir, condition, noise, replicate
                    )
                    stable_summary_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(summary_path, stable_summary_path)
                    summary_path = stable_summary_path
                row["summary_json"] = str(summary_path) if summary_path else ""
                validation_errors = _validate_summary(
                    summary_path=summary_path,
                    condition=condition,
                    noise=noise,
                    task_group=args.task_group,
                    n_envs=args.n_envs,
                    n_rollouts=args.n_rollouts,
                    randomness=args.randomness,
                    replicate=replicate,
                    expected_saved_rollouts=(0 if is_vlm_cover else None),
                    require_noise_stats=is_vlm_cover,
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
                            replicate=replicate,
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
            if is_vlm_cover:
                _upsert_manifest(args.manifest, row)
            else:
                _append_manifest(args.manifest, row)
            manifest_index[key] = row
            if row["status"] != "ok" and not args.continue_on_error:
                raise SystemExit(int(row.get("returncode", 1) or 1))

    if args.dry_run:
        print("[dry-run] commands printed; manifest and report unchanged", flush=True)
        return

    if not args.skip_report:
        if is_vlm_cover:
            from scripts.audit_clean_train_noise_eval import audit_vlm_cover_108

            audit_payload, audit_returncode = audit_vlm_cover_108(
                manifest_path=args.manifest,
                legacy_manifest_path=args.legacy_manifest,
                require_complete=True,
                n_rollouts=args.n_rollouts,
            )
            if audit_returncode != 0:
                raise RuntimeError(
                    "vlm-cover-108 audit failed before report generation: "
                    f"missing={len(audit_payload['missing'])} "
                    f"issues={len(audit_payload['issues'])}"
                )
        generate_report(
            manifest_path=args.manifest,
            report_path=args.report,
            figures_dir=args.figures_dir,
            data_dir=args.data_dir,
            profile=args.profile,
            legacy_manifest_path=(args.legacy_manifest if is_vlm_cover else None),
        )
        print(f"[done] report written to {args.report}", flush=True)


if __name__ == "__main__":
    main()
