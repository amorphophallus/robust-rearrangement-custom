from __future__ import annotations

from typing import Iterable, Optional


TASK_PROGRESS_SCHEMA: dict[str, dict[str, list[str]]] = {
    "one_leg": {
        "skill_states": [
            "top-leg-pick",
            "top-leg-push",
            "leg-top-pick",
            "leg-top-place",
            "leg-top-insert",
            "leg-top-screw",
        ],
        "assembly_steps": [
            "top-leg",
        ],
    },
    "round_table": {
        "skill_states": [
            "top-leg-push",
            "leg-top-pick",
            "leg-top-place",
            "leg-top-insert",
            "leg-top-screw",
            "base-leg-pick",
            "base-leg-place",
            "base-leg-insert",
            "base-leg-screw",
        ],
        "assembly_steps": [
            "top-leg",
            "leg-base",
        ],
    },
    "lamp": {
        "skill_states": [
            "base-bulb-push",
            "bulb-base-pick",
            "bulb-base-place",
            "bulb-base-insert",
            "bulb-base-screw",
            "hood-base-pick",
            "hood-base-place",
        ],
        "assembly_steps": [
            "base-bulb",
            "base-hood",
        ],
    },
}


def get_task_progress_labels(task_name: Optional[str], kind: str) -> list[str]:
    if task_name is None:
        return []
    return list(TASK_PROGRESS_SCHEMA.get(str(task_name), {}).get(kind, ()))


def normalize_progress_counts(
    counts: Optional[dict[str, int]],
    expected_labels: Iterable[str],
) -> dict[str, int]:
    normalized = {}
    raw_counts = counts or {}

    for label in expected_labels:
        normalized[str(label)] = int(raw_counts.get(label, 0))

    for key, value in raw_counts.items():
        key = str(key)
        if key in normalized:
            continue
        normalized[key] = int(value)

    return normalized
