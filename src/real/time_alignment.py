"""Timestamp alignment primitives shared by real-data conversion and evaluation.

All public timestamps are integer nanoseconds in the Unix wall-clock domain unless
the function name explicitly says otherwise.  Wall time is used for matching
sensor streams; monotonic time should still be used for local sleeping/deadlines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class TimestampMatch:
    """A one-to-one match from a target sample to a source sample."""

    target_index: int
    source_index: int
    target_time_ns: int
    source_time_ns: int

    @property
    def residual_ms(self) -> float:
        return (self.source_time_ns - self.target_time_ns) / NS_PER_MS


def monotonic_nearest_unique_match(
    target_times_ns: Sequence[int],
    source_times_ns: Sequence[int],
    *,
    max_residual_ms: float,
) -> Dict[int, TimestampMatch]:
    """Greedily match ordered targets to the nearest unused ordered source.

    The source index must increase strictly between matches.  This prevents a
    slowly produced PromptDA frame from being silently repeated for several
    policy timesteps.  Inputs must be non-decreasing.  Ties choose the earlier
    source so the result is deterministic.
    """

    targets = np.asarray(target_times_ns, dtype=np.int64).reshape(-1)
    sources = np.asarray(source_times_ns, dtype=np.int64).reshape(-1)
    if targets.size and np.any(np.diff(targets) < 0):
        raise ValueError("target_times_ns must be non-decreasing")
    if sources.size and np.any(np.diff(sources) < 0):
        raise ValueError("source_times_ns must be non-decreasing")
    if max_residual_ms < 0:
        raise ValueError("max_residual_ms must be non-negative")

    threshold_ns = int(round(max_residual_ms * NS_PER_MS))
    matches: Dict[int, TimestampMatch] = {}
    last_source = -1
    for target_index, target_time in enumerate(targets):
        insertion = int(np.searchsorted(sources, target_time, side="left"))
        candidates = {
            source_index
            for source_index in (insertion - 1, insertion)
            if last_source < source_index < sources.size
        }
        if not candidates:
            # searchsorted can point behind last_source after a source was used.
            next_source = last_source + 1
            if next_source < sources.size:
                candidates.add(next_source)
        if not candidates:
            break
        source_index = min(
            candidates,
            key=lambda index: (abs(int(sources[index]) - int(target_time)), index),
        )
        residual_ns = int(sources[source_index]) - int(target_time)
        if abs(residual_ns) > threshold_ns:
            # A source in the future may become useful for a later target.  Do
            # not consume it when the current target has no valid match.
            continue
        match = TimestampMatch(
            target_index=target_index,
            source_index=source_index,
            target_time_ns=int(target_time),
            source_time_ns=int(sources[source_index]),
        )
        matches[target_index] = match
        last_source = source_index
    return matches


def contiguous_segments(
    valid_indices: Iterable[int],
    target_times_ns: Sequence[int],
    *,
    max_gap_ms: float,
    min_steps: int,
) -> List[List[int]]:
    """Split valid action indices on missing steps or excessive time gaps."""

    if min_steps <= 0:
        raise ValueError("min_steps must be positive")
    if max_gap_ms < 0:
        raise ValueError("max_gap_ms must be non-negative")
    times = np.asarray(target_times_ns, dtype=np.int64).reshape(-1)
    indices = sorted(set(int(index) for index in valid_indices))
    if any(index < 0 or index >= times.size for index in indices):
        raise IndexError("valid index is outside target_times_ns")
    max_gap_ns = int(round(max_gap_ms * NS_PER_MS))
    result: List[List[int]] = []
    current: List[int] = []
    for index in indices:
        if current:
            previous = current[-1]
            if index != previous + 1 or int(times[index] - times[previous]) > max_gap_ns:
                if len(current) >= min_steps:
                    result.append(current)
                current = []
        current.append(index)
    if len(current) >= min_steps:
        result.append(current)
    return result


def interpolate_vector(
    sample_times_ns: Sequence[int],
    values: Sequence[Sequence[float]],
    query_time_ns: int,
) -> np.ndarray:
    """Linearly interpolate a vector while refusing extrapolation."""

    times = np.asarray(sample_times_ns, dtype=np.int64).reshape(-1)
    vectors = np.asarray(values, dtype=np.float64)
    if times.size != vectors.shape[0] or times.size == 0:
        raise ValueError("sample times and values must have the same non-zero length")
    if np.any(np.diff(times) <= 0):
        raise ValueError("sample_times_ns must increase strictly")
    query = int(query_time_ns)
    if query < int(times[0]) or query > int(times[-1]):
        raise ValueError("query_time_ns is outside the interpolation interval")
    right = int(np.searchsorted(times, query, side="left"))
    if right < times.size and int(times[right]) == query:
        return vectors[right].copy()
    left = right - 1
    alpha = (query - int(times[left])) / float(int(times[right] - times[left]))
    return (vectors[left] * (1.0 - alpha) + vectors[right] * alpha).copy()


def interpolate_quaternion_xyzw(
    sample_times_ns: Sequence[int],
    quaternions_xyzw: Sequence[Sequence[float]],
    query_time_ns: int,
) -> np.ndarray:
    """SLERP an xyzw quaternion while refusing extrapolation."""

    times = np.asarray(sample_times_ns, dtype=np.int64).reshape(-1)
    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64)
    if times.size != quaternions.shape[0] or quaternions.shape[1:] != (4,):
        raise ValueError("quaternions_xyzw must have shape (N, 4)")
    if times.size == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("sample_times_ns must increase strictly")
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.all(np.isfinite(quaternions)) or np.any(norms < 1e-8):
        raise ValueError("quaternions must be finite and non-zero")
    query = int(query_time_ns)
    if query < int(times[0]) or query > int(times[-1]):
        raise ValueError("query_time_ns is outside the interpolation interval")
    seconds = (times - times[0]).astype(np.float64) / 1e9
    query_seconds = (query - int(times[0])) / 1e9
    rotation = Slerp(seconds, Rotation.from_quat(quaternions / norms[:, None]))(
        [query_seconds]
    )
    return rotation.as_quat()[0]


@dataclass(frozen=True)
class LatencyProfile:
    """Measured delays used to timestamp observations and send commands early."""

    front_observation_ms: float
    wrist_observation_ms: float
    robot_observation_ms: float
    gripper_observation_ms: float
    robot_action_ms: float
    gripper_action_ms: float
    measured_at: str
    schema_version: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LatencyProfile":
        profile = cls(
            front_observation_ms=float(value["front_observation_ms"]),
            wrist_observation_ms=float(value["wrist_observation_ms"]),
            robot_observation_ms=float(value["robot_observation_ms"]),
            gripper_observation_ms=float(value["gripper_observation_ms"]),
            robot_action_ms=float(value["robot_action_ms"]),
            gripper_action_ms=float(value["gripper_action_ms"]),
            measured_at=str(value["measured_at"]),
            schema_version=int(value.get("schema_version", 1)),
        )
        profile.validate()
        return profile

    @classmethod
    def load(cls, path: Path) -> "LatencyProfile":
        value = json.loads(Path(path).expanduser().read_text())
        if not isinstance(value, Mapping):
            raise ValueError("latency profile must contain a JSON object")
        return cls.from_mapping(value)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported latency profile schema {self.schema_version}")
        for name in (
            "front_observation_ms",
            "wrist_observation_ms",
            "robot_observation_ms",
            "gripper_observation_ms",
            "robot_action_ms",
            "gripper_action_ms",
        ):
            latency = getattr(self, name)
            if not np.isfinite(latency) or latency < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.measured_at.strip():
            raise ValueError("latency profile must record measured_at")


@dataclass(frozen=True)
class ScheduledAction:
    target_time_ns: int
    action: np.ndarray
    query_id: int
    chunk_index: int


class TimestampedActionBuffer:
    """Future action slots with UMI-style overwrite semantics."""

    def __init__(self, period_ns: int):
        if period_ns <= 0:
            raise ValueError("period_ns must be positive")
        self.period_ns = int(period_ns)
        self._slots: Dict[int, ScheduledAction] = {}

    def update(
        self,
        actions: Sequence[Sequence[float]],
        target_times_ns: Sequence[int],
        *,
        query_id: int,
        now_ns: int,
    ) -> Tuple[int, int]:
        """Overwrite future slots and discard an already stale chunk prefix.

        Returns ``(accepted, stale)``.  Times need not be exact multiples of the
        period; existing future targets within half a period of the new chunk's
        first live target are treated as the same logical grid and replaced.
        """

        action_array = np.asarray(actions)
        times = np.asarray(target_times_ns, dtype=np.int64).reshape(-1)
        if action_array.shape[0] != times.size:
            raise ValueError("actions and target_times_ns must have equal length")
        if times.size and np.any(np.diff(times) <= 0):
            raise ValueError("target_times_ns must increase strictly")
        stale = int(np.searchsorted(times, int(now_ns), side="right"))
        accepted = 0
        if stale < times.size:
            first_new_time = int(times[stale])
            overwrite_from = first_new_time - self.period_ns // 2
            for old_time in [time for time in self._slots if time >= overwrite_from]:
                del self._slots[old_time]
        for chunk_index in range(stale, times.size):
            target = int(times[chunk_index])
            self._slots[target] = ScheduledAction(
                target_time_ns=target,
                action=np.asarray(action_array[chunk_index]).copy(),
                query_id=int(query_id),
                chunk_index=chunk_index,
            )
            accepted += 1
        return accepted, stale

    def prune(self, now_ns: int) -> int:
        stale_times = [target for target in self._slots if target <= int(now_ns)]
        for target in stale_times:
            del self._slots[target]
        return len(stale_times)

    def next(self, now_ns: int) -> Optional[ScheduledAction]:
        self.prune(now_ns)
        if not self._slots:
            return None
        return self._slots[min(self._slots)]

    def pop_due(self, now_ns: int, *, tolerance_ns: int = 0) -> Optional[ScheduledAction]:
        if not self._slots:
            return None
        target = min(self._slots)
        if target > int(now_ns) + int(tolerance_ns):
            return None
        return self._slots.pop(target)

    def coverage_end_ns(self, now_ns: int) -> Optional[int]:
        self.prune(now_ns)
        return max(self._slots) if self._slots else None

    def clear(self) -> None:
        self._slots.clear()

    def __len__(self) -> int:
        return len(self._slots)
