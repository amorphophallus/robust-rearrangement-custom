"""End-effector pose frame selection shared by training and evaluation code.

The policy-facing field names stay ``ee_pos`` and ``ee_quat``.  This module only
selects which stored coordinate frame is exposed through those stable names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROBOT_BASE = "robot-base"
SIM_LOCAL = "sim-local"
ORIGINAL = "original"
REAL_TIP = "real-tip"

EEPPOSE_FRAME_HELP = (
    "EE pose exposed to the policy. 'robot-base' is the canonical default; "
    "'original' restores the domain-specific legacy representation. In sim, "
    "'original' and 'sim-local' are equivalent; on real, 'original' and "
    "'real-tip' are equivalent."
)


def resolve_eepose_frame(frame_spec: str, *, original_frame: str) -> str:
    """Resolve a CLI EE-frame specification to a concrete representation.

    ``original`` is a domain-dependent alias.  Callers pass the domain's legacy
    frame via ``original_frame``; users select either ``original`` or the
    concrete frame name as separate CLI values.
    """

    value = str(frame_spec).strip().lower().replace("_", "-")
    aliases = {
        "base": ROBOT_BASE,
        "robot": ROBOT_BASE,
        "robot-base": ROBOT_BASE,
        "sim": SIM_LOCAL,
        "sim-local": SIM_LOCAL,
        "tip": REAL_TIP,
        "real-tip": REAL_TIP,
        "virtual-tip": REAL_TIP,
        "original": ORIGINAL,
    }
    resolved = aliases.get(value)
    if resolved is None:
        valid = "robot-base, original, sim-local, real-tip"
        raise ValueError(f"unsupported eepose frame {frame_spec!r}; expected {valid}")
    if resolved == ORIGINAL:
        return resolve_eepose_frame(original_frame, original_frame=original_frame)
    return resolved


def select_policy_eepose(
    robot_state: Mapping[str, Any],
    frame_spec: str,
    *,
    original_frame: str,
) -> dict[str, Any]:
    """Return a shallow copy with the requested pose under stable policy keys.

    The function is deliberately array-library agnostic: numpy arrays and torch
    tensors are both passed through unchanged.
    """

    selected = dict(robot_state)
    resolved = resolve_eepose_frame(frame_spec, original_frame=original_frame)
    if resolved == ROBOT_BASE:
        suffix = ""
    elif resolved == SIM_LOCAL:
        suffix = "_sim"
    elif resolved == REAL_TIP:
        suffix = "_original"
    else:  # pragma: no cover - guarded by resolve_eepose_frame
        raise AssertionError(f"unhandled eepose frame: {resolved}")

    pos_key = f"ee_pos{suffix}"
    quat_key = f"ee_quat{suffix}"
    missing = [key for key in (pos_key, quat_key) if key not in robot_state]
    if missing:
        raise KeyError(
            f"cannot select eepose frame {resolved!r}; missing robot-state fields: "
            + ", ".join(missing)
        )
    selected["ee_pos"] = robot_state[pos_key]
    selected["ee_quat"] = robot_state[quat_key]
    pose_key = f"ee_pose{suffix}"
    if pose_key in robot_state:
        selected["ee_pose"] = robot_state[pose_key]
    return selected
