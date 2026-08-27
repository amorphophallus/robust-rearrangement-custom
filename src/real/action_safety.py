"""Validation and conversion for absolute-pose real-robot policy actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_6d_to_matrix(rotation_6d: Sequence[float]) -> np.ndarray:
    value = np.asarray(rotation_6d, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(value)):
        raise ValueError("rotation 6D contains non-finite values")
    first = value[:3]
    second = value[3:]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        raise ValueError("rotation 6D first axis is degenerate")
    first = first / first_norm
    second = second - first * np.dot(first, second)
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        raise ValueError("rotation 6D axes are collinear")
    second = second / second_norm
    third = np.cross(first, second)
    # RR's matrix_to_rotation_6d flattens the first two matrix rows, and its
    # rotation_6d_to_matrix stacks the recovered vectors as rows.
    return np.stack((first, second, third), axis=0)


def rotation_distance_rad(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left).reshape(3, 3).T @ np.asarray(right).reshape(3, 3)
    return float(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec()))


@dataclass(frozen=True)
class ActionSafetyLimits:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    min_ee_z: float
    max_translation_step_m: float = 0.025
    max_rotation_step_rad: float = 0.35
    max_translation_speed_m_s: float = 0.25
    max_rotation_speed_rad_s: float = 1.5

    def __post_init__(self):
        minimum = np.asarray(self.workspace_min, dtype=np.float64).reshape(3)
        maximum = np.asarray(self.workspace_max, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("workspace bounds must be finite")
        if np.any(minimum >= maximum):
            raise ValueError("workspace_min must be strictly below workspace_max")
        if not minimum[2] <= self.min_ee_z <= maximum[2]:
            raise ValueError("min_ee_z must lie within workspace z bounds")
        object.__setattr__(self, "workspace_min", minimum)
        object.__setattr__(self, "workspace_max", maximum)
        for name in (
            "max_translation_step_m",
            "max_rotation_step_rad",
            "max_translation_speed_m_s",
            "max_rotation_speed_rad_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ValidatedAbsoluteAction:
    policy_action: np.ndarray
    position: np.ndarray
    rotation_matrix: np.ndarray
    axis_angle: np.ndarray
    gripper_sign: float

    def deoxys_action(self) -> np.ndarray:
        return np.concatenate(
            [self.position, self.axis_angle, [self.gripper_sign]]
        ).astype(np.float64)


def validate_absolute_action(
    action: Sequence[float],
    *,
    reference_pose: np.ndarray,
    period_s: float,
    limits: ActionSafetyLimits,
) -> ValidatedAbsoluteAction:
    """Validate an RR 10D absolute action and convert it to Deoxys OSC format."""

    value = np.asarray(action, dtype=np.float64).reshape(-1)
    if value.shape != (10,):
        raise ValueError(f"expected 10D pos/rot6d/gripper action, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("policy action contains non-finite values")
    if period_s <= 0:
        raise ValueError("period_s must be positive")
    reference = np.asarray(reference_pose, dtype=np.float64)
    if reference.shape != (4, 4) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_pose must be a finite 4x4 matrix")

    position = value[:3]
    rotation = rotation_6d_to_matrix(value[3:9])
    if np.any(position < limits.workspace_min) or np.any(position > limits.workspace_max):
        raise ValueError(
            f"target position {position.tolist()} is outside workspace "
            f"[{limits.workspace_min.tolist()}, {limits.workspace_max.tolist()}]"
        )
    if position[2] < limits.min_ee_z:
        raise ValueError(
            f"target EE z={position[2]:.4f} is below min_ee_z={limits.min_ee_z:.4f}"
        )
    translation_step = float(np.linalg.norm(position - reference[:3, 3]))
    rotation_step = rotation_distance_rad(reference[:3, :3], rotation)
    if translation_step > limits.max_translation_step_m:
        raise ValueError(
            f"translation step {translation_step:.4f} m exceeds "
            f"{limits.max_translation_step_m:.4f} m"
        )
    if rotation_step > limits.max_rotation_step_rad:
        raise ValueError(
            f"rotation step {rotation_step:.4f} rad exceeds "
            f"{limits.max_rotation_step_rad:.4f} rad"
        )
    if translation_step / period_s > limits.max_translation_speed_m_s:
        raise ValueError("translation speed exceeds configured safety limit")
    if rotation_step / period_s > limits.max_rotation_speed_rad_s:
        raise ValueError("rotation speed exceeds configured safety limit")

    return ValidatedAbsoluteAction(
        policy_action=value.copy(),
        position=position.copy(),
        rotation_matrix=rotation,
        axis_angle=Rotation.from_matrix(rotation).as_rotvec(),
        gripper_sign=float(np.sign(value[-1]) or -1.0),
    )
