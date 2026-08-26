"""Real-robot part-pose providers used by offline annotation.

Providers are deliberately independent of the simulator annotator.  They
return an optional pose overlay for annotation without modifying the raw pose
or AprilTag-found fields saved in a demonstration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class PartPoseEstimate:
    """One external part-pose estimate in the trajectory's April frame."""

    part_name: str
    pose_april: np.ndarray
    source: str
    confidence: float
    details: Mapping[str, Any] = field(default_factory=dict)


class PartPoseProvider:
    """Interface for real-only pose recovery sources."""

    def estimate(
        self, frame_idx: int, observation: Mapping[str, Any]
    ) -> Optional[PartPoseEstimate]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {"type": type(self).__name__}


@dataclass(frozen=True)
class RecoveredTabletopPoseProvider(PartPoseProvider):
    """Use one recovered tabletop pose from a selected frame onward."""

    pose_april: np.ndarray
    start_frame: int
    keyframe: int
    confidence: float
    source: str = "sam2_rgbd_full_tabletop_cad_chamfer"
    part_name: str = "square_table_top"
    recovery_path: Optional[Path] = None
    fit_details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pose = np.asarray(self.pose_april, dtype=np.float32).reshape(-1)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Recovered pose_april must be a finite 7-vector")
        if float(np.linalg.norm(pose[3:])) < 1e-6:
            raise ValueError("Recovered pose quaternion has zero norm")
        pose = pose.copy()
        pose[3:] /= np.linalg.norm(pose[3:])
        object.__setattr__(self, "pose_april", pose)
        if self.start_frame < 0 or self.keyframe < 0:
            raise ValueError("Recovery frame indices must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Recovery confidence must be in [0, 1]")

    @classmethod
    def from_recovery_json(cls, path: Path) -> "RecoveredTabletopPoseProvider":
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls(
            pose_april=np.asarray(payload["pose_april"], dtype=np.float32),
            start_frame=int(payload["start_frame"]),
            keyframe=int(payload["keyframe"]),
            confidence=float(payload["confidence"]),
            source=str(payload.get("source", "recovered_tabletop_pose")),
            part_name=str(payload.get("part_name", "square_table_top")),
            recovery_path=path,
            fit_details=dict(payload.get("fit_details", {})),
        )

    def estimate(
        self, frame_idx: int, observation: Mapping[str, Any]
    ) -> Optional[PartPoseEstimate]:
        del observation
        if frame_idx < self.start_frame:
            return None
        return PartPoseEstimate(
            part_name=self.part_name,
            pose_april=self.pose_april.copy(),
            source=self.source,
            confidence=self.confidence,
            details={
                "start_frame": self.start_frame,
                "keyframe": self.keyframe,
                "fit_details": dict(self.fit_details),
            },
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "part_name": self.part_name,
            "source": self.source,
            "confidence": self.confidence,
            "start_frame": self.start_frame,
            "keyframe": self.keyframe,
            "recovery_path": (
                None if self.recovery_path is None else str(self.recovery_path)
            ),
            "fit_details": dict(self.fit_details),
        }
