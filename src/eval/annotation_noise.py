from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.common.eepose import ROBOT_BASE
from src.common.guidance import GUIDANCE_SCHEMA_VERSION, normalize_guidance_frame


@dataclass(frozen=True)
class AnnotationNoiseConfig:
    pos_std_m: float = 0.0
    ori_std_deg: float = 0.0
    seed: int = 0
    mode: str = "gaussian_clip_2sigma"
    apply_to: str = "all"
    shuffle_seed: int = 0
    shuffle_bank_path: Optional[str] = None
    shuffle_records: tuple[dict[str, Any], ...] = ()

    @property
    def enabled(self) -> bool:
        return (
            self.mode == "shuffle"
            or self.pos_std_m > 0.0
            or self.ori_std_deg > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos_std_m": float(self.pos_std_m),
            "ori_std_deg": float(self.ori_std_deg),
            "seed": int(self.seed),
            "mode": self.mode,
            "apply_to": self.apply_to,
            "shuffle_seed": int(self.shuffle_seed),
            "shuffle_bank_path": self.shuffle_bank_path,
            "shuffle_record_count": len(self.shuffle_records),
            "enabled": self.enabled,
        }


def make_annotation_noise_config(
    *,
    pos_std_m: float = 0.0,
    ori_std_deg: float = 0.0,
    seed: int = 0,
    mode: str = "gaussian_clip_2sigma",
    apply_to: str = "all",
    shuffle_seed: int = 0,
    shuffle_bank_path: Optional[str | Path] = None,
    shuffle_records: Optional[list[dict[str, Any]]] = None,
) -> AnnotationNoiseConfig:
    if mode not in {"gaussian_clip_2sigma", "uniform", "shuffle"}:
        raise ValueError(f"Unsupported annotation noise mode: {mode}")
    if apply_to not in {"point", "grasp", "all"}:
        raise ValueError(f"Unsupported annotation noise apply_to: {apply_to}")
    if pos_std_m < 0.0:
        raise ValueError("--annotation-noise-pos-std-m must be non-negative")
    if ori_std_deg < 0.0:
        raise ValueError("--annotation-noise-ori-std-deg must be non-negative")
    bank_path = None if shuffle_bank_path is None else str(shuffle_bank_path)
    records = list(shuffle_records or [])
    if mode == "shuffle" and not records:
        if bank_path is None:
            raise ValueError("shuffle mode requires a guidance bank")
        records = load_guidance_shuffle_bank(Path(bank_path))
    if mode == "shuffle" and not records:
        raise ValueError("shuffle guidance bank is empty")
    return AnnotationNoiseConfig(
        pos_std_m=float(pos_std_m),
        ori_std_deg=float(ori_std_deg),
        seed=int(seed),
        mode=mode,
        apply_to=apply_to,
        shuffle_seed=int(shuffle_seed),
        shuffle_bank_path=bank_path,
        shuffle_records=tuple(records),
    )


def load_guidance_shuffle_bank(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        records: list[dict[str, Any]] = []
        for child in sorted(path.glob("*.json")):
            records.extend(load_guidance_shuffle_bank(child))
        return records
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "guidance_frame" not in payload:
        raise ValueError(
            f"Guidance bank {path} has no explicit guidance_frame; "
            "migrate or regenerate the legacy bank before use"
        )
    frame = normalize_guidance_frame(payload["guidance_frame"])
    if frame != ROBOT_BASE:
        raise ValueError(
            f"Guidance bank {path} uses {frame!r}; expected canonical {ROBOT_BASE!r}"
        )
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Invalid guidance bank payload: {path}")
    output = [dict(record) for record in records if isinstance(record, dict)]
    for record in output:
        record_frame = normalize_guidance_frame(
            record.get("guidance_frame", frame)
        )
        if record_frame != ROBOT_BASE:
            raise ValueError(
                f"Guidance bank {path} contains a non-canonical record frame: "
                f"{record_frame!r}"
            )
    return output


def write_guidance_shuffle_bank(
    path: Path,
    *,
    task: str,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": GUIDANCE_SCHEMA_VERSION,
                "task": task,
                "guidance_frame": ROBOT_BASE,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-10:
        return np.eye(3, dtype=np.float32)

    axis = axis_angle / angle
    x, y, z = axis.astype(np.float64)
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _sample_vector(rng: np.random.Generator, std: float, mode: str) -> np.ndarray:
    if std <= 0.0:
        return np.zeros(3, dtype=np.float32)
    if mode == "uniform":
        return rng.uniform(-std, std, size=3).astype(np.float32)
    sample = rng.normal(0.0, std, size=3)
    return np.clip(sample, -2.0 * std, 2.0 * std).astype(np.float32)


@dataclass
class AnnotationNoisePhaseState:
    env_idx: int
    seed_offset: int = 0
    phase_idx: int = 0
    phase_key: Optional[tuple[Any, ...]] = None
    pos_noise: Optional[np.ndarray] = None
    rot_noise: Optional[np.ndarray] = None
    shuffled_point: Optional[np.ndarray] = None
    shuffled_pose: Optional[np.ndarray] = None
    shuffle_info: Optional[dict[str, Any]] = None

    def _phase_rng(
        self,
        config: AnnotationNoiseConfig,
        phase_key: tuple[Any, ...],
        *,
        seed: int,
    ) -> tuple[bool, np.random.Generator]:
        changed = phase_key != self.phase_key
        if changed:
            self.phase_idx += 1
            self.phase_key = phase_key
        phase_seed = (
            int(seed)
            + self.seed_offset * 1_009_003
            + self.env_idx * 1_000_003
            + self.phase_idx * 9_176
        )
        return changed, np.random.default_rng(phase_seed)

    def current_noise(
        self,
        config: Optional[AnnotationNoiseConfig],
        phase_key: tuple[Any, ...],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if config is None or not config.enabled:
            self.phase_key = phase_key
            return (
                np.zeros(3, dtype=np.float32),
                np.eye(3, dtype=np.float32),
                self.phase_idx,
            )

        changed, rng = self._phase_rng(
            config,
            phase_key,
            seed=int(config.seed),
        )
        if changed:
            self.pos_noise = _sample_vector(rng, float(config.pos_std_m), config.mode)
            ori_std_rad = np.deg2rad(float(config.ori_std_deg))
            self.rot_noise = _axis_angle_to_matrix(
                _sample_vector(rng, ori_std_rad, config.mode)
            )

        return (
            np.asarray(self.pos_noise, dtype=np.float32),
            np.asarray(self.rot_noise, dtype=np.float32),
            self.phase_idx,
        )

    def current_shuffle(
        self,
        config: AnnotationNoiseConfig,
        phase_key: tuple[Any, ...],
        *,
        task: str,
        skill: Optional[str],
        skill_state: Optional[str],
        guidance_point: Optional[np.ndarray],
        guidance_pose: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
        changed, rng = self._phase_rng(
            config,
            phase_key,
            seed=int(config.shuffle_seed),
        )
        if not changed and self.shuffle_info is not None:
            return self.shuffled_point, self.shuffled_pose, self.shuffle_info

        current_state = str(skill_state or "")
        current_type = str(skill or current_state.rsplit("-", 1)[-1])
        candidates = [
            record
            for record in config.shuffle_records
            if str(record.get("task")) == str(task)
        ]
        preferred = [
            record
            for record in candidates
            if str(record.get("skill_type")) == current_type
            and str(record.get("skill_state")) != current_state
        ]
        selection_policy = "same_skill_type_different_state"
        if not preferred:
            preferred = [
                record
                for record in candidates
                if str(record.get("skill_state")) != current_state
            ]
            selection_policy = "any_skill_different_state"
        if not preferred:
            raise ValueError(
                "No different-state shuffled guidance donor for "
                f"task={task} skill_state={current_state}"
            )

        donor = preferred[int(rng.integers(0, len(preferred)))]
        donor_point_raw = donor.get("guidance_point")
        donor_pose_raw = donor.get("guidance_pose")
        donor_point = (
            None
            if donor_point_raw is None
            else np.asarray(donor_point_raw, dtype=np.float32).reshape(3)
        )
        donor_pose = (
            None
            if donor_pose_raw is None
            else np.asarray(donor_pose_raw, dtype=np.float32).reshape(4, 4)
        )
        if donor_point is None and donor_pose is not None:
            donor_point = donor_pose[:3, 3].copy()

        if config.apply_to == "point":
            if donor_point is None:
                raise ValueError("Point shuffle donor has no guidance point")
            self.shuffled_point = donor_point.copy()
            self.shuffled_pose = None if guidance_pose is None else guidance_pose.copy()
            if self.shuffled_pose is not None:
                self.shuffled_pose[:3, 3] = donor_point
        else:
            if donor_pose is None:
                raise ValueError("Pose shuffle donor has no guidance pose")
            self.shuffled_pose = donor_pose.copy()
            self.shuffled_point = (
                donor_pose[:3, 3].copy()
                if guidance_point is not None
                else None
            )

        clean_position = None
        if guidance_point is not None:
            clean_position = np.asarray(guidance_point, dtype=np.float32)
        elif guidance_pose is not None:
            clean_position = np.asarray(guidance_pose, dtype=np.float32)[:3, 3]
        shuffled_position = (
            self.shuffled_point
            if self.shuffled_point is not None
            else self.shuffled_pose[:3, 3]
        )
        displacement_m = (
            None
            if clean_position is None
            else float(np.linalg.norm(shuffled_position - clean_position))
        )
        orientation_displacement_deg = None
        if guidance_pose is not None and self.shuffled_pose is not None:
            clean_rotation = np.asarray(guidance_pose, dtype=np.float32)[:3, :3]
            shuffled_rotation = self.shuffled_pose[:3, :3]
            rotation_delta = shuffled_rotation @ clean_rotation.T
            cosine = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
            orientation_displacement_deg = float(np.degrees(np.arccos(cosine)))
        self.shuffle_info = {
            "enabled": True,
            "mode": "shuffle",
            "phase_idx": int(self.phase_idx),
            "phase_key": [None if item is None else str(item) for item in phase_key],
            "donor_task": donor.get("task"),
            "donor_skill_state": donor.get("skill_state"),
            "donor_skill_type": donor.get("skill_type"),
            "donor_source_episode": donor.get("source_episode"),
            "donor_visit_idx": donor.get("visit_idx"),
            "selection_policy": selection_policy,
            "realized_pos_displacement_m": displacement_m,
            "realized_ori_displacement_deg": orientation_displacement_deg,
            "apply_to": config.apply_to,
        }
        return self.shuffled_point, self.shuffled_pose, self.shuffle_info


def apply_annotation_noise(
    *,
    guidance_point: Optional[np.ndarray],
    guidance_pose: Optional[np.ndarray],
    task: str = "",
    skill_state: Optional[str] = None,
    skill: Optional[str],
    phase_key: tuple[Any, ...],
    state: AnnotationNoisePhaseState,
    config: Optional[AnnotationNoiseConfig],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
    if config is None or not config.enabled:
        return guidance_point, guidance_pose, {"enabled": False}

    if config.mode == "shuffle":
        return state.current_shuffle(
            config,
            phase_key,
            task=task,
            skill=skill,
            skill_state=skill_state,
            guidance_point=guidance_point,
            guidance_pose=guidance_pose,
        )

    pos_noise, rot_noise, phase_idx = state.current_noise(config, phase_key)
    noisy_point = None if guidance_point is None else guidance_point.copy()
    noisy_pose = None if guidance_pose is None else guidance_pose.copy()

    apply_point_pos = noisy_point is not None and config.apply_to in {"all", "point"}
    # The pose is annotation-only at this point. Keep its position aligned with
    # the noisy point so tracking evaluates the target actually drawn to the policy.
    apply_pose_pos = noisy_pose is not None and config.apply_to in {
        "all",
        "point",
        "grasp",
    }
    apply_ori = noisy_pose is not None and config.apply_to in {"all", "grasp"}

    if apply_point_pos:
        noisy_point = noisy_point + pos_noise
    if apply_pose_pos:
        noisy_pose[:3, 3] = noisy_pose[:3, 3] + pos_noise

    if apply_ori:
        noisy_pose[:3, :3] = rot_noise @ noisy_pose[:3, :3]

    return noisy_point, noisy_pose, {
        "enabled": True,
        "phase_idx": int(phase_idx),
        "seed_offset": int(state.seed_offset),
        "phase_key": [None if item is None else str(item) for item in phase_key],
        "pos_noise": (
            pos_noise.astype(float).tolist()
            if apply_point_pos or apply_pose_pos
            else [0.0, 0.0, 0.0]
        ),
        "ori_noise_matrix": (
            rot_noise.astype(float).tolist()
            if apply_ori
            else np.eye(3, dtype=float).tolist()
        ),
        "apply_point_pos": bool(apply_point_pos),
        "apply_pose_pos": bool(apply_pose_pos),
        "apply_ori": bool(apply_ori),
    }
