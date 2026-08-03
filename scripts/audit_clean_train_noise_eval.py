from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.run_clean_train_noise_eval import (
    CONDITIONS,
    _manifest_lookup,
    _noise_levels_for_family,
    _read_jsonl,
    _rollout_group_dirs,
    _validate_summary,
)


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
) -> tuple[dict[str, Any], int]:
    manifest_index = _manifest_lookup(_read_jsonl(manifest_path))
    previously_validated = _load_state(state_path)
    validated_groups = set(previously_validated)
    completed: list[str] = []
    newly_validated: list[str] = []
    missing: list[str] = []
    issues: list[str] = []

    for condition in CONDITIONS:
        for noise in _noise_levels_for_family(condition.family):
            group_id = f"{condition.condition_id}/{noise.noise_id}"
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
                "n_rollouts": 12,
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
                    n_rollouts=12,
                    randomness="low",
                )
            )

            rollout_dirs = _rollout_group_dirs(
                task_group="one_leg+round_table+lamp",
                randomness="low",
                condition=condition,
                noise=noise,
            )
            for rollout_dir in rollout_dirs:
                saved_count = sum(1 for _ in rollout_dir.rglob("*.pkl")) if rollout_dir.exists() else 0
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
    payload = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "completed_count": len(completed),
        "validated_count": len(validated_groups),
        "expected_count": 25,
        "newly_validated": newly_validated,
        "missing": missing,
        "issues": issues,
        "complete": len(validated_groups) == 25 and not issues,
    }
    if issues:
        return payload, 1
    if require_complete and not payload["complete"]:
        return payload, 2
    return payload, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    payload, returncode = audit(
        manifest_path=args.manifest,
        state_path=args.state,
        require_complete=args.require_complete,
    )
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
