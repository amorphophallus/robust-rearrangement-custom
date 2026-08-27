"""
Skill annotation consistency verifier.

Computes the guidance point's offset from its reference part (anchor /
operated / absolute), then checks that this relative position does not
jump too much between consecutive frames within the same skill phase.
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

    For each frame, computes ``relative_pos = guidance_point - ref_part_pos``.
    Within the same skill phase, compares *relative_pos* against the previous
    frame — a jump larger than *tolerance_m* is flagged.

    Parameters
    ----------
    tolerance_m:
        Max allowed step-to-step displacement of relative_pos [meters].
        Default 0.020 (2 cm).
    """

    def __init__(self, tolerance_m: float = 0.020):
        self.tolerance_m = tolerance_m
        self._prev_skill_state: Optional[str] = None
        self._prev_relative_pos: Optional[np.ndarray] = None

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
        rb_states,
        base_pos,
    ) -> Optional[Dict[str, Any]]:
        """Run one consistency check."""
        if skill is None or guidance_point is None:
            self._prev_skill_state = None
            self._prev_relative_pos = None
            return None

        ref_mode = _REFERENCE_MODE.get(skill)
        if ref_mode is None:
            return None

        # --- resolve reference part position in robot-base frame ---
        ref_pos_robot = self._ref_part_position(
            ref_mode, part1_name, part2_name, active_part_name,
            part_idxs, rb_states, base_pos,
        )
        if ref_pos_robot is None:
            return None

        relative_pos = guidance_point.astype(np.float32) - ref_pos_robot

        # --- phase-change detection ---
        current_key = skill_state_label if skill_state_label is not None else skill
        if current_key != self._prev_skill_state:
            self._prev_skill_state = current_key
            self._prev_relative_pos = relative_pos.copy()
            return {"status": "reference_set", "ref_mode": ref_mode}

        # --- same phase: compare with previous frame ---
        result: Dict[str, Any] = {"status": "consistent", "ref_mode": ref_mode}
        if self._prev_relative_pos is not None:
            jump = float(np.linalg.norm(relative_pos - self._prev_relative_pos))
            result["jump_m"] = jump
            result["tolerance_m"] = self.tolerance_m
            if jump > self.tolerance_m:
                result["status"] = "offset_detected"

        self._prev_relative_pos = relative_pos.copy()
        return result

    def reset(self) -> None:
        self._prev_skill_state = None
        self._prev_relative_pos = None

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _ref_part_position(
        ref_mode: str,
        part1_name: str,
        part2_name: str,
        active_part_name: Optional[str],
        part_idxs: Dict[str, List[int]],
        rb_states,
        base_pos,
    ) -> Optional[np.ndarray]:
        """Return the reference part position in robot-base frame (3,)."""
        if ref_mode == "absolute":
            return np.zeros(3, dtype=np.float32)

        if ref_mode == "anchor":
            ref_name = part1_name
        elif ref_mode == "operated":
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
        robot_origin_sim_local = _to_numpy(base_pos).astype(np.float32)
        return ref_pos_sim_local - robot_origin_sim_local


# ---------------------------------------------------------------------------
# Rollout-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class JumpEvent:
    """Details of a single step-to-step relative-pos jump."""
    step_idx: int
    assembly_step: str
    skill: str
    jump_m: float
    tolerance_m: float
    ref_mode: str


@dataclass
class VerifyHistory:
    """Accumulates verification results across a rollout."""

    furniture_name: str
    steps: List[Dict[str, Any]] = field(default_factory=list)

    max_jump_m: float = 0.0
    verified_step_count: int = 0
    jump_events: List[JumpEvent] = field(default_factory=list)
    phase_starts: List[Dict[str, Any]] = field(default_factory=list)

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

        if status == "reference_set":
            self.phase_starts.append({
                "step": step_idx,
                "assembly_step": assembly_step,
                "skill": skill,
                "ref_mode": verify.get("ref_mode"),
            })

        jump = verify.get("jump_m")
        if jump is not None and jump > self.max_jump_m:
            self.max_jump_m = jump

        if status == "offset_detected":
            self.jump_events.append(JumpEvent(
                step_idx=step_idx,
                assembly_step=assembly_step,
                skill=skill,
                jump_m=jump,
                tolerance_m=verify.get("tolerance_m", 0),
                ref_mode=verify.get("ref_mode", "?"),
            ))

    def summary(self) -> str:
        lines = [
            f"[VerifyHistory] furniture={self.furniture_name} "
            f"verified_steps={self.verified_step_count} "
            f"max_jump={self.max_jump_m*1000:.1f}mm "
            f"jump_events={len(self.jump_events)}",
        ]

        if not self.jump_events:
            lines.append("  No step-to-step jumps exceeding threshold.")
            if self.phase_starts:
                lines.append("")
                lines.append("  Skill phase timeline:")
                for pt in self.phase_starts:
                    lines.append(
                        f"    step={pt['step']:>5}  pair={pt['assembly_step']:<20} "
                        f"skill={pt['skill']:<8}  ref={pt['ref_mode']}"
                    )
            return "\n".join(lines)

        lines.append("")
        lines.append("  Step-to-step jump events:")
        lines.append("  " + "-" * 80)
        for i, ev in enumerate(self.jump_events):
            lines.append(
                f"  [{i+1}] step={ev.step_idx:>5}  pair={ev.assembly_step:<20} "
                f"skill={ev.skill:<8}  ref={ev.ref_mode:<10}  "
                f"jump={ev.jump_m*1000:>6.1f}mm  thresh={ev.tolerance_m*1000:.0f}mm"
            )
        lines.append("  " + "-" * 80)

        if self.phase_starts:
            lines.append("")
            lines.append("  Skill phase timeline (for cross-reference):")
            for pt in self.phase_starts:
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
