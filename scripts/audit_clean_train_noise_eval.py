from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scripts.run_clean_train_noise_eval import (
        CONDITIONS,
        FIXED_R180,
        MANIFEST_DEFAULT,
        ReplicateConfig,
        SHUFFLED_GUIDANCE,
        VLM_COVER_MANIFEST_DEFAULT,
        _manifest_lookup,
        _manifest_key,
        _noise_levels_for_family,
        _read_jsonl,
        _rollout_group_dirs,
        _validate_summary,
    )
except ModuleNotFoundError:  # Allow `python scripts/audit_clean_train_noise_eval.py`.
    from run_clean_train_noise_eval import (
        CONDITIONS,
        FIXED_R180,
        MANIFEST_DEFAULT,
        ReplicateConfig,
        SHUFFLED_GUIDANCE,
        VLM_COVER_MANIFEST_DEFAULT,
        _manifest_lookup,
        _manifest_key,
        _noise_levels_for_family,
        _read_jsonl,
        _rollout_group_dirs,
        _validate_summary,
    )


VIDEO_SUFFIXES = ("_cam1", "_cam2", "_dep1", "_dep2")


def _condition_by_id():
    return {condition.condition_id: condition for condition in CONDITIONS}


def _noise_by_id(condition):
    levels = _noise_levels_for_family(condition.family, legacy_only=False)
    levels.extend([SHUFFLED_GUIDANCE])
    if condition.family == "grasp-part":
        levels.append(FIXED_R180)
    return {level.noise_id: level for level in levels}


def _expected_cover_new_keys() -> set[tuple[str, str, int, int, int]]:
    keys = set()
    for condition in CONDITIONS:
        for noise_id in ("n5", "n6", "n7"):
            keys.add((condition.condition_id, noise_id, 0, 0, 0))
        for seed in (1, 2):
            for noise_id in ("n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"):
                keys.add((condition.condition_id, noise_id, seed, seed, seed))
            keys.add((condition.condition_id, "shuffle", seed, seed, seed))
        if condition.family == "grasp-part":
            for seed in (0, 1, 2):
                keys.add((condition.condition_id, "r180", seed, seed, seed))
    return keys


def audit_vlm_cover_108(
    *,
    manifest_path: Path,
    legacy_manifest_path: Path,
    require_complete: bool,
    n_rollouts: int = 36,
) -> tuple[dict[str, Any], int]:
    """Strictly validate new replicates and pooled legacy+new 108-rollout cells."""
    new_rows = _read_jsonl(manifest_path)
    legacy_rows = _read_jsonl(legacy_manifest_path)
    expected_new = _expected_cover_new_keys()
    issues: list[str] = []
    missing: list[str] = []

    counts: dict[tuple[str, str, int, int, int], int] = {}
    for row in new_rows:
        key = _manifest_key(row)
        counts[key] = counts.get(key, 0) + 1
    for key, count in counts.items():
        if count != 1:
            issues.append(f"duplicate manifest key {key}: count={count}")
        if key not in expected_new:
            issues.append(f"unexpected manifest key {key}")

    new_index = _manifest_lookup(new_rows, replicate_aware=True)
    conditions = _condition_by_id()
    for key in sorted(expected_new):
        row = new_index.get(key)
        key_text = "/".join(map(str, key))
        if row is None or row.get("status") != "ok":
            missing.append(key_text)
            continue
        condition = conditions[key[0]]
        noise = _noise_by_id(condition)[key[1]]
        for field, expected in {
            "profile": "vlm-cover-108",
            "n_envs": 3,
            "n_rollouts": n_rollouts,
            "tracking_rollouts_per_task": n_rollouts,
            "randomness": "low",
            "save_rollouts_count": 0,
            "resolved_annotation_source": "scripted",
        }.items():
            if row.get(field) != expected:
                issues.append(
                    f"{key_text}: manifest.{field}={row.get(field)!r} expected={expected!r}"
                )
        checkpoint_hash = str(row.get("checkpoint_sha256", ""))
        if len(checkpoint_hash) != 64:
            issues.append(f"{key_text}: invalid checkpoint_sha256")
        replicate = ReplicateConfig(key[2], key[3], key[4])
        summary_value = str(row.get("summary_json", "") or "")
        summary_path = Path(summary_value) if summary_value else None
        for issue in _validate_summary(
            summary_path=summary_path,
            condition=condition,
            noise=noise,
            task_group="one_leg+round_table+lamp",
            n_envs=3,
            n_rollouts=n_rollouts,
            randomness="low",
            replicate=replicate,
            expected_saved_rollouts=0,
            require_noise_stats=True,
        ):
            issues.append(f"{key_text}: {issue}")

    # No two completed replicates may point at the same output or summary.
    for field in ("summary_json", "group_log"):
        seen: dict[str, tuple[str, str, int, int, int]] = {}
        for row in new_rows:
            value = str(row.get(field, "") or "")
            if not value:
                continue
            key = _manifest_key(row)
            if value in seen and seen[value] != key:
                issues.append(f"{field} collision: {seen[value]} and {key}: {value}")
            seen[value] = key
    suffix_seen: dict[str, tuple[str, str, int, int, int]] = {}
    for row in new_rows:
        command = list(row.get("command") or [])
        try:
            suffix = str(command[command.index("--rollout-suffix-model-name") + 1])
        except (ValueError, IndexError):
            issues.append(f"{_manifest_key(row)}: command has no rollout suffix")
            continue
        key = _manifest_key(row)
        if suffix in suffix_seen and suffix_seen[suffix] != key:
            issues.append(
                f"rollout suffix collision: {suffix_seen[suffix]} and {key}: {suffix}"
            )
        suffix_seen[suffix] = key

    legacy_index = _manifest_lookup(legacy_rows)
    pooled_cells: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        noise_ids = [f"n{idx}" for idx in range(8)] + ["shuffle"]
        if condition.family == "grasp-part":
            noise_ids.append("r180")
        for noise_id in noise_ids:
            replicate_rows = []
            for seed in (0, 1, 2):
                if seed == 0 and noise_id in {"n0", "n1", "n2", "n3", "n4", "shuffle"}:
                    row = legacy_index.get((condition.condition_id, noise_id))
                    if row is None or row.get("status") != "ok":
                        legacy_key = f"legacy/{condition.condition_id}/{noise_id}/0"
                        if legacy_key not in missing:
                            missing.append(legacy_key)
                else:
                    row = new_index.get(
                        (condition.condition_id, noise_id, seed, seed, seed)
                    )
                if row is not None and row.get("status") == "ok":
                    replicate_rows.append(row)
            for task in ("one_leg", "round_table", "lamp"):
                rollout_count = 0
                for row in replicate_rows:
                    summary_value = str(row.get("summary_json", "") or "")
                    if not summary_value or not Path(summary_value).exists():
                        continue
                    summary = json.loads(Path(summary_value).read_text())
                    task_payload = (summary.get("per_task") or {}).get(task, {})
                    rollout_count += int(task_payload.get("n_rollouts", 0))
                expected_tracking_n = (
                    72
                    if noise_id in {"n0", "n1", "n2", "n3", "n4", "shuffle"}
                    else 108
                )
                pooled_cells.append(
                    {
                        "condition_id": condition.condition_id,
                        "noise_id": noise_id,
                        "task": task,
                        "replicate_count": len(replicate_rows),
                        "rollout_count": rollout_count,
                        "tracking_n": expected_tracking_n,
                    }
                )
                if len(replicate_rows) == 3 and rollout_count != 3 * n_rollouts:
                    issues.append(
                        f"pooled {condition.condition_id}/{noise_id}/{task}: "
                        f"rollout_count={rollout_count} expected={3 * n_rollouts}"
                    )

    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "profile": "vlm-cover-108",
        "manifest": str(manifest_path),
        "legacy_manifest": str(legacy_manifest_path),
        "expected_new_invocations": len(expected_new),
        "completed_new_invocations": sum(
            key in new_index and new_index[key].get("status") == "ok"
            for key in expected_new
        ),
        "missing": missing,
        "issues": issues,
        "pooled_cells": pooled_cells,
        "complete": not missing and not issues,
    }
    if issues:
        return payload, 1
    if require_complete and missing:
        return payload, 2
    return payload, 0


def _saved_rollout_count(rollout_dir: Path) -> int:
    """Count rollout basenames whether pickle payloads were retained or cleaned."""
    if not rollout_dir.exists():
        return 0

    rollout_ids = {path.stem for path in rollout_dir.rglob("*.pkl")}
    for path in rollout_dir.rglob("*.mp4"):
        stem = path.stem
        for suffix in VIDEO_SUFFIXES:
            if stem.endswith(suffix):
                rollout_ids.add(stem[: -len(suffix)])
                break
    return len(rollout_ids)


def _load_state(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(item) for item in payload.get("validated_groups", [])}


def _write_state(path: Path | None, validated_groups: set[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "validated_groups": sorted(validated_groups),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def audit(
    *,
    manifest_path: Path,
    state_path: Path | None,
    require_complete: bool,
    n_rollouts: int = 36,
    include_shuffled: bool = False,
    shuffled_rollouts: int = 36,
) -> tuple[dict[str, Any], int]:
    manifest_index = _manifest_lookup(_read_jsonl(manifest_path))
    previously_validated = _load_state(state_path)
    validated_groups = set(previously_validated)
    completed: list[str] = []
    newly_validated: list[str] = []
    missing: list[str] = []
    issues: list[str] = []

    for condition in CONDITIONS:
        for noise in _noise_levels_for_family(
            condition.family, include_shuffled=include_shuffled
        ):
            group_id = f"{condition.condition_id}/{noise.noise_id}"
            group_rollouts = (
                shuffled_rollouts if noise.perturbation == "shuffle" else n_rollouts
            )
            row = manifest_index.get((condition.condition_id, noise.noise_id))
            if row is None or row.get("status") != "ok":
                missing.append(group_id)
                continue

            completed.append(group_id)
            row_checks = {
                "condition": condition.condition,
                "family": condition.family,
                "task_group": "one_leg+round_table+lamp",
                "randomness": "low",
                "n_envs": 3,
                "n_rollouts": group_rollouts,
                "save_rollouts_count": 8,
                "checkpoint": str(condition.checkpoint),
            }
            group_issues = []
            for key, expected in row_checks.items():
                if row.get(key) != expected:
                    group_issues.append(
                        f"manifest.{key}={row.get(key)!r} expected={expected!r}"
                    )

            summary_text = str(row.get("summary_json", "") or "").strip()
            summary_path = Path(summary_text) if summary_text else None
            group_issues.extend(
                _validate_summary(
                    summary_path=summary_path,
                    condition=condition,
                    noise=noise,
                    task_group="one_leg+round_table+lamp",
                    n_envs=3,
                    n_rollouts=group_rollouts,
                    randomness="low",
                    require_tracking=noise.perturbation == "shuffle",
                )
            )
            rollout_dirs = _rollout_group_dirs(
                task_group="one_leg+round_table+lamp",
                randomness="low",
                condition=condition,
                noise=noise,
            )
            for rollout_dir in rollout_dirs:
                saved_count = _saved_rollout_count(rollout_dir)
                if saved_count != 8:
                    group_issues.append(
                        f"saved_rollouts={saved_count} expected=8 dir={rollout_dir}"
                    )

            if group_issues:
                validated_groups.discard(group_id)
                issues.extend(f"{group_id}: {issue}" for issue in group_issues)
            else:
                validated_groups.add(group_id)
                if group_id not in previously_validated:
                    newly_validated.append(group_id)

    _write_state(state_path, validated_groups)
    expected_count = 25 + (5 if include_shuffled else 0)
    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "completed_count": len(completed),
        "validated_count": len(validated_groups),
        "expected_count": expected_count,
        "newly_validated": newly_validated,
        "missing": missing,
        "issues": issues,
        "complete": len(validated_groups) == expected_count and not issues,
    }
    if issues:
        return payload, 1
    if require_complete and not payload["complete"]:
        return payload, 2
    return payload, 0


def _limit_reported_issues(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    limited = dict(payload)
    issues = list(payload.get("issues", []))
    limited["issue_count"] = len(issues)
    if limit >= 0 and len(issues) > limit:
        limited["issues"] = issues[:limit]
        limited["issues_truncated"] = len(issues) - limit
    else:
        limited["issues_truncated"] = 0
    return limited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["legacy-fresh36", "vlm-cover-108"],
        default="legacy-fresh36",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--legacy-manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--n-rollouts", type=int, default=36)
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument("--shuffled-rollouts", type=int, default=36)
    parser.add_argument(
        "--max-reported-issues",
        type=int,
        default=-1,
        help="Limit issues printed in JSON; negative keeps all issues.",
    )
    args = parser.parse_args()

    if args.profile == "vlm-cover-108":
        payload, returncode = audit_vlm_cover_108(
            manifest_path=args.manifest or VLM_COVER_MANIFEST_DEFAULT,
            legacy_manifest_path=args.legacy_manifest,
            require_complete=args.require_complete,
            n_rollouts=args.n_rollouts,
        )
    else:
        if args.manifest is None:
            parser.error("--manifest is required for legacy-fresh36")
        payload, returncode = audit(
            manifest_path=args.manifest,
            state_path=args.state,
            require_complete=args.require_complete,
            n_rollouts=args.n_rollouts,
            include_shuffled=args.include_shuffled,
            shuffled_rollouts=args.shuffled_rollouts,
        )
    print(
        json.dumps(
            _limit_reported_issues(payload, args.max_reported_issues),
            sort_keys=True,
        )
    )
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
