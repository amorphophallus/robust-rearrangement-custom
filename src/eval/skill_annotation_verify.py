"""
Thin wrapper that delegates to the built-in verification in skill_annotation_util.

The primary API is now in skill_annotation_util itself — just pass
``enable_verify=True`` to get_annotation_bundle_for_env() and the returned
bundle will contain a ``"verify"`` key.

This module remains for convenience (e.g. VerifyHistory aggregation across a
rollout) and backwards compatibility.

Usage (new preferred style):
    from src.eval.skill_annotation_util import get_annotation_bundle_for_env

    bundle = get_annotation_bundle_for_env(env, env_idx, enable_verify=True)
    if bundle["verify"] and bundle["verify"]["status"] == "offset_detected":
        print(f"WARNING: guidance point drifted {bundle['verify']['drift_m']:.3f}m")

Usage with VerifyHistory (this module):
    from src.eval.skill_annotation_verify import VerifyHistory, verify_and_record

    history = VerifyHistory("one_leg")
    for step in range(max_steps):
        bundle = get_annotation_bundle_for_env(env, env_idx, enable_verify=True)
        verify_and_record(bundle, history)
    print(history.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VerifyHistory:
    """Accumulates verification results across rollout steps."""

    furniture_name: str
    steps: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregated stats
    max_drift_m: float = 0.0
    offset_detected_count: int = 0
    verified_step_count: int = 0

    def record(self, bundle: Dict[str, Any]) -> None:
        """Record verification result from one annotation bundle."""
        verify = bundle.get("verify")
        self.steps.append(verify)

        if verify is None:
            return

        self.verified_step_count += 1

        drift = verify.get("drift_m")
        if drift is not None and drift > self.max_drift_m:
            self.max_drift_m = drift

        if verify.get("status") == "offset_detected":
            self.offset_detected_count += 1

    def summary(self) -> str:
        lines = [
            f"[VerifyHistory] furniture={self.furniture_name} "
            f"steps={len(self.steps)} verified={self.verified_step_count}",
            f"  max_drift={self.max_drift_m*1000:.1f}mm "
            f"offset_detected={self.offset_detected_count}",
        ]
        skills = {}
        for s in self.steps:
            if s and "ref_part" in s:
                k = s.get("status", "?")
                skills[k] = skills.get(k, 0) + 1
        if skills:
            lines.append(f"  status_distribution={skills}")
        return "\n".join(lines)


def verify_and_record(
    bundle: Dict[str, Any],
    history: VerifyHistory,
) -> None:
    """Convenience: extract verify result from bundle and record in history."""
    history.record(bundle)
