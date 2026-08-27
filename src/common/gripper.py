"""Shared gripper-width representation helpers for policy inputs."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


GRIPPER_OPEN_THRESHOLD_METERS = 0.05
GRIPPER_OPEN_VALUE = -1.0
GRIPPER_CLOSED_VALUE = 1.0
GRIPPER_WIDTH_ENCODING = "binary_open_negative"


def binarize_gripper_width(width):
    """Map metric gripper width to the policy's open/closed convention.

    Widths greater than or equal to 5 cm are open (``-1``); smaller widths
    are closed (``+1``). The return value keeps the input tensor/array shape,
    device, and floating-point dtype.
    """
    if isinstance(width, torch.Tensor):
        if not width.is_floating_point():
            width = width.to(dtype=torch.float32)
        return torch.where(
            width >= GRIPPER_OPEN_THRESHOLD_METERS,
            torch.full_like(width, GRIPPER_OPEN_VALUE),
            torch.full_like(width, GRIPPER_CLOSED_VALUE),
        )

    array = np.asarray(width)
    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32
    return np.where(
        array >= GRIPPER_OPEN_THRESHOLD_METERS,
        GRIPPER_OPEN_VALUE,
        GRIPPER_CLOSED_VALUE,
    ).astype(dtype, copy=False)


def binarize_robot_state_gripper_width(robot_state: Mapping):
    """Return a shallow copy with only ``gripper_width`` binarized."""
    if "gripper_width" not in robot_state:
        raise KeyError("robot_state is missing gripper_width")
    encoded = dict(robot_state)
    encoded["gripper_width"] = binarize_gripper_width(robot_state["gripper_width"])
    return encoded


def normalizer_expects_binary_gripper_width(normalizer) -> bool:
    """Infer old metric vs new binary proprioception from checkpoint stats."""
    if "robot_state" not in normalizer.stats:
        raise KeyError("normalizer is missing robot_state statistics")
    stats = normalizer.stats["robot_state"]
    minimum = float(stats["min"][-1].detach().cpu())
    maximum = float(stats["max"][-1].detach().cpu())

    # A variable metric-width column stays close to [0, 0.08]. Constant
    # columns need special handling because LinearNormalizer expands any
    # constant value by +/-1: constant metric 0.08 becomes [-0.92, 1.08],
    # while binary open/closed become [-2, 0] or [0, 2].
    binary_bounds = ((-1.0, 1.0), (-2.0, 0.0), (0.0, 2.0))
    return any(
        np.isclose(minimum, expected_min)
        and np.isclose(maximum, expected_max)
        for expected_min, expected_max in binary_bounds
    )
