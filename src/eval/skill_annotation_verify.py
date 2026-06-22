"""
Skill annotation consistency verifier.

Checks that the guidance point stays at a fixed relative offset from its
reference part within the same skill phase.  A drift larger than the
configured tolerance is flagged as ``offset_detected``.

Reference-part mapping
----------------------
======= ========= ====================================================
skill   reference rationale
======= ========= ====================================================
place   anchor    target is the hole on the stationary anchor part
insert  anchor    same as place
pick    operated  target is near the part being picked up
screw   anchor    the part is already in the hole; target is relative
                  to the anchor (e.g. table)
push    absolute  target is a fixed world-space push destination
======= ========= ====================================================

where *anchor* = the non-operated part in the assembly pair (part1), and
*operated* = the part being actively moved (the "active_part").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.eval.skill_annotation_util import _to_numpy

# ---------------------------------------------------------------------------
# Reference-part mode per skill
# ---------------------------------------------------------------------------

_REFERENCE_MODE: Dict[str, str] = {
    "place": "anchor",
    "insert": "anchor",
    "pick": "operated",
    "screw": "anchor",
    "push": "absolute",
}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class SkillAnnotationVerifier:
    """Per-annotator consistency verifier.

    One instance is attached to a single ``SkillAnnotator`` and tracks state
    across ``step()`` calls.

    Parameters
    ----------
    tolerance_m:
        Max allowed drift of (guidance_point - ref_part_pos) from the baseline
        established at the start of each skill phase [meters].
    """

    def __init__(self, tolerance_m: float = 0.005):
        self.tolerance_m = tolerance_m
        self._ref_skill_state: Optional[str] = None
        self._ref_relative_pos: Optional[np.ndarray] = None

    # -- public API -----------------------------------------------------------

    def check(
        self,
        skill: Optional[str],
        skill_state_label: Optional[str],
        guidance_point: Optional[np.ndarray],
        part1_name: str,
        part2_name: str,
        active_part_name: Optional[str],
        part_idxs: Dict[str, List[int]],
        rb_states,  # torch.Tensor or np.ndarray  (N, 13) in sim-local frame
        env_offset,  # torch.Tensor or np.ndarray  (3,)
    ) -> Optional[Dict[str, Any]]:
        """Run one consistency check.

        Returns a result dict, or ``None`` when the check is not applicable
        (e.g. unknown skill or missing guidance point).
        """
        if skill is None or guidance_point is None:
            return None

        ref_mode = _REFERENCE_MODE.get(skill)
        if ref_mode is None:
            return None

        # --- resolve reference part position in sim frame ---
        ref_pos_sim = self._ref_part_position(
            ref_mode, part1_name, part2_name, active_part_name,
            part_idxs, rb_states, env_offset,
        )
        if ref_pos_sim is None:
            return None

        relative_pos = guidance_point.astype(np.float32) - ref_pos_sim

        # --- phase-change detection ---
        current_key = skill_state_label if skill_state_label is not None else skill
        if current_key != self._ref_skill_state:
            self._ref_skill_state = current_key
            self._ref_relative_pos = relative_pos.copy()
            return {
                "status": "reference_set",
                "ref_mode": ref_mode,
                "relative_pos": relative_pos,
            }

        # --- same phase: measure drift ---
        drift = float(np.linalg.norm(relative_pos - self._ref_relative_pos))
        is_consistent = drift <= self.tolerance_m

        return {
            "status": "consistent" if is_consistent else "offset_detected",
            "drift_m": drift,
            "tolerance_m": self.tolerance_m,
            "relative_pos": relative_pos,
            "ref_relative_pos": self._ref_relative_pos.copy(),
            "ref_mode": ref_mode,
        }

    def reset(self) -> None:
        """Clear per-phase tracking state (called on annotator reset)."""
        self._ref_skill_state = None
        self._ref_relative_pos = None

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _ref_part_position(
        ref_mode: str,
        part1_name: str,
        part2_name: str,
        active_part_name: Optional[str],
        part_idxs: Dict[str, List[int]],
        rb_states,
        env_offset,
    ) -> Optional[np.ndarray]:
        """Return the reference part position in sim frame (3,)."""
        if ref_mode == "absolute":
            return np.zeros(3, dtype=np.float32)

        if ref_mode == "anchor":
            ref_name = part1_name
        elif ref_mode == "operated":
            # The operated part is whichever part is currently "active".
            if active_part_name is not None and active_part_name == part2_name:
                ref_name = part2_name
            else:
                ref_name = part1_name
        else:
            return None

        if ref_name not in part_idxs:
            return None

        idx = part_idxs[ref_name][0]
        ref_pos_sim_local = _to_numpy(rb_states[idx][:3]).astype(np.float32)
        off = _to_numpy(env_offset).astype(np.float32)
        return ref_pos_sim_local + off


# ---------------------------------------------------------------------------
# Rollout-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class OffsetEvent:
    """Details of a single consistency violation."""
    step_idx: int
    assembly_step: str
    skill: str
    drift_m: float
    tolerance_m: float
    ref_mode: str
    relative_pos: Any  # np.ndarray (3,) — current offset from ref part
    ref_relative_pos: Any  # np.ndarray (3,) — baseline offset


@dataclass
class VerifyHistory:
    """Accumulates verification results across a rollout with per-event detail."""

    furniture_name: str
    steps: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregated
    max_drift_m: float = 0.0
    verified_step_count: int = 0

    # Per-offset events
    offset_events: List[OffsetEvent] = field(default_factory=list)

    # Phase transition tracking
    phase_transitions: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        bundle: Dict[str, Any],
        *,
        step_idx: int = -1,
        assembly_step: str = "",
        skill: str = "",
    ) -> None:
        verify = bundle.get("verify")
        self.steps.append(verify)
        if verify is None:
            return

        self.verified_step_count += 1
        status = verify.get("status")

        # Track phase transitions
        if status == "reference_set":
            self.phase_transitions.append({
                "step": step_idx,
                "assembly_step": assembly_step,
                "skill": skill,
                "ref_mode": verify.get("ref_mode"),
            })

        # Track drift
        drift = verify.get("drift_m")
        if drift is not None and drift > self.max_drift_m:
            self.max_drift_m = drift

        if status == "offset_detected":
            self.offset_events.append(OffsetEvent(
                step_idx=step_idx,
                assembly_step=assembly_step,
                skill=skill,
                drift_m=drift,
                tolerance_m=verify.get("tolerance_m", 0),
                ref_mode=verify.get("ref_mode", "?"),
                relative_pos=verify.get("relative_pos"),
                ref_relative_pos=verify.get("ref_relative_pos"),
            ))

    def summary(self) -> str:
        lines = [
            f"[VerifyHistory] furniture={self.furniture_name} "
            f"verified_steps={self.verified_step_count} "
            f"max_drift={self.max_drift_m*1000:.1f}mm "
            f"offset_events={len(self.offset_events)}",
        ]

        if not self.offset_events:
            lines.append("  No consistency violations detected.")
            return "\n".join(lines)

        lines.append("")
        lines.append("  Offset events (chronological):")
        lines.append("  " + "-" * 90)
        for i, ev in enumerate(self.offset_events):
            rel = np.asarray(ev.relative_pos)
            ref = np.asarray(ev.ref_relative_pos)
            delta = rel - ref
            lines.append(
                f"  [{i+1}] step={ev.step_idx:>5}  pair={ev.assembly_step:<20} "
                f"skill={ev.skill:<8}  ref_mode={ev.ref_mode:<10}"
            )
            lines.append(
                f"       drift={ev.drift_m*1000:>6.1f}mm  "
                f"tolerance={ev.tolerance_m*1000:.0f}mm  "
                f"Δ=(dx={delta[0]:+.4f} dy={delta[1]:+.4f} dz={delta[2]:+.4f})m"
            )
            lines.append(
                f"       rel_now =[{rel[0]:.5f} {rel[1]:.5f} {rel[2]:.5f}]"
            )
            lines.append(
                f"       rel_ref =[{ref[0]:.5f} {ref[1]:.5f} {ref[2]:.5f}]"
            )
        lines.append("  " + "-" * 90)

        # Phase timeline for context
        if self.phase_transitions:
            lines.append("")
            lines.append("  Skill phase timeline (for cross-reference):")
            for pt in self.phase_transitions:
                lines.append(
                    f"    step={pt['step']:>5}  pair={pt['assembly_step']:<20} "
                    f"skill={pt['skill']:<8}  ref={pt['ref_mode']}"
                )

        return "\n".join(lines)


def verify_and_record(
    bundle: Dict[str, Any],
    history: VerifyHistory,
    *,
    step_idx: int = -1,
    assembly_step: str = "",
    skill: str = "",
) -> None:
    history.record(bundle, step_idx=step_idx, assembly_step=assembly_step, skill=skill)
