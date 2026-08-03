from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class AnnotationNoiseConfig:
    pos_std_m: float = 0.0
    ori_std_deg: float = 0.0
    seed: int = 0
    mode: str = "gaussian_clip_2sigma"
    apply_to: str = "all"

    @property
    def enabled(self) -> bool:
        return self.pos_std_m > 0.0 or self.ori_std_deg > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos_std_m": float(self.pos_std_m),
            "ori_std_deg": float(self.ori_std_deg),
            "seed": int(self.seed),
            "mode": self.mode,
            "apply_to": self.apply_to,
            "enabled": self.enabled,
        }


def make_annotation_noise_config(
    *,
    pos_std_m: float = 0.0,
    ori_std_deg: float = 0.0,
    seed: int = 0,
    mode: str = "gaussian_clip_2sigma",
    apply_to: str = "all",
) -> AnnotationNoiseConfig:
    if mode not in {"gaussian_clip_2sigma", "uniform"}:
        raise ValueError(f"Unsupported annotation noise mode: {mode}")
    if apply_to not in {"point", "grasp", "all"}:
        raise ValueError(f"Unsupported annotation noise apply_to: {apply_to}")
    if pos_std_m < 0.0:
        raise ValueError("--annotation-noise-pos-std-m must be non-negative")
    if ori_std_deg < 0.0:
        raise ValueError("--annotation-noise-ori-std-deg must be non-negative")
    return AnnotationNoiseConfig(
        pos_std_m=float(pos_std_m),
        ori_std_deg=float(ori_std_deg),
        seed=int(seed),
        mode=mode,
        apply_to=apply_to,
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

        if phase_key != self.phase_key:
            self.phase_idx += 1
            self.phase_key = phase_key
            seed = (
                int(config.seed)
                + self.seed_offset * 1_009_003
                + self.env_idx * 1_000_003
                + self.phase_idx * 9_176
            )
            rng = np.random.default_rng(seed)
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


def apply_annotation_noise(
    *,
    guidance_point: Optional[np.ndarray],
    guidance_pose: Optional[np.ndarray],
    skill: Optional[str],
    phase_key: tuple[Any, ...],
    state: AnnotationNoisePhaseState,
    config: Optional[AnnotationNoiseConfig],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
    if config is None or not config.enabled:
        return guidance_point, guidance_pose, {"enabled": False}

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
