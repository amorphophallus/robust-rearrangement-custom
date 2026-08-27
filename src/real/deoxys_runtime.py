"""Small Deoxys adapters with explicit timestamp semantics.

The module itself does not import Deoxys, so its interpolation logic remains
unit-testable on machines without the robot stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from src.real.time_alignment import interpolate_quaternion_xyzw, interpolate_vector


@dataclass(frozen=True)
class RobotStateSample:
    receive_wall_time_ns: int
    source_time: float
    frame: int
    wrist_pose: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray


@dataclass(frozen=True)
class GripperStateSample:
    receive_wall_time_ns: int
    source_time: float
    width: float


@dataclass(frozen=True)
class InterpolatedRobotState:
    query_time_ns: int
    wrist_pose: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    left_receive_wall_time_ns: int
    right_receive_wall_time_ns: int


def _message_field(message: Any, name: str, default):
    return getattr(message, name, default)


def _source_time_seconds(message: Any) -> float:
    value = _message_field(message, "time", None)
    if value is None:
        return float("nan")
    if hasattr(value, "toSec"):
        return float(value.toSec)
    return float(value)


def robot_sample_from_record(record: Mapping[str, Any]) -> RobotStateSample:
    message = record["message"]
    pose = np.asarray(message.O_T_EE, dtype=np.float64)
    if pose.size != 16:
        raise ValueError("Deoxys robot state O_T_EE must contain 16 values")
    return RobotStateSample(
        receive_wall_time_ns=int(record["receive_wall_time_ns"]),
        source_time=_source_time_seconds(message),
        frame=int(_message_field(message, "frame", -1)),
        wrist_pose=pose.reshape(4, 4).transpose(),
        joint_positions=np.asarray(message.q, dtype=np.float64).reshape(7),
        joint_velocities=np.asarray(message.dq, dtype=np.float64).reshape(7),
        joint_torques=np.asarray(
            _message_field(message, "tau_J", _message_field(message, "tau_J_d", [np.nan] * 7)),
            dtype=np.float64,
        ).reshape(7),
    )


def gripper_sample_from_record(record: Mapping[str, Any]) -> GripperStateSample:
    message = record["message"]
    return GripperStateSample(
        receive_wall_time_ns=int(record["receive_wall_time_ns"]),
        source_time=_source_time_seconds(message),
        width=float(np.asarray(message.width).reshape(-1)[0]),
    )


def interpolate_robot_state(
    samples: Sequence[RobotStateSample],
    query_time_ns: int,
    *,
    observation_latency_ms: float,
) -> InterpolatedRobotState:
    """Interpolate robot state to a sensor-source wall time.

    Deoxys message ``time`` is retained for diagnostics but is not assumed to
    share a clock with RealSense.  Calibrated transport/observation latency is
    subtracted from local receive wall time before bracketing.
    """

    if len(samples) < 2:
        raise ValueError("at least two robot samples are required")
    latency_ns = int(round(float(observation_latency_ms) * 1e6))
    source_times = np.asarray(
        [sample.receive_wall_time_ns - latency_ns for sample in samples],
        dtype=np.int64,
    )
    query = int(query_time_ns)
    right = int(np.searchsorted(source_times, query, side="left"))
    if right == 0 or right >= len(samples):
        raise ValueError("robot state history does not bracket camera source time")
    left = right - 1
    pair_times = source_times[[left, right]]
    pair = [samples[left], samples[right]]

    positions = [sample.wrist_pose[:3, 3] for sample in pair]
    quaternions = [Rotation.from_matrix(sample.wrist_pose[:3, :3]).as_quat() for sample in pair]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = interpolate_vector(pair_times, positions, query)
    pose[:3, :3] = Rotation.from_quat(
        interpolate_quaternion_xyzw(pair_times, quaternions, query)
    ).as_matrix()
    return InterpolatedRobotState(
        query_time_ns=query,
        wrist_pose=pose,
        joint_positions=interpolate_vector(
            pair_times, [sample.joint_positions for sample in pair], query
        ),
        joint_velocities=interpolate_vector(
            pair_times, [sample.joint_velocities for sample in pair], query
        ),
        joint_torques=interpolate_vector(
            pair_times, [sample.joint_torques for sample in pair], query
        ),
        left_receive_wall_time_ns=pair[0].receive_wall_time_ns,
        right_receive_wall_time_ns=pair[1].receive_wall_time_ns,
    )


def interpolate_gripper_width(
    samples: Sequence[GripperStateSample],
    query_time_ns: int,
    *,
    observation_latency_ms: float,
) -> float:
    if len(samples) < 2:
        raise ValueError("at least two gripper samples are required")
    latency_ns = int(round(float(observation_latency_ms) * 1e6))
    times = np.asarray(
        [sample.receive_wall_time_ns - latency_ns for sample in samples],
        dtype=np.int64,
    )
    return float(
        interpolate_vector(
            times,
            np.asarray([[sample.width] for sample in samples]),
            int(query_time_ns),
        )[0]
    )
