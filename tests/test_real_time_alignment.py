import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.real.time_alignment import (
    LatencyProfile,
    TimestampedActionBuffer,
    contiguous_segments,
    interpolate_quaternion_xyzw,
    interpolate_vector,
    monotonic_nearest_unique_match,
)


class RealTimeAlignmentTest(unittest.TestCase):
    def test_monotonic_matching_does_not_repeat_a_source(self):
        ms = 1_000_000
        matches = monotonic_nearest_unique_match(
            [100 * ms, 160 * ms, 220 * ms],
            [130 * ms, 230 * ms],
            max_residual_ms=75,
        )
        self.assertEqual(
            [(key, value.source_index) for key, value in matches.items()],
            [(0, 0), (1, 1)],
        )
        self.assertEqual(matches[0].residual_ms, 30.0)

    def test_segments_split_missing_indices_and_large_time_gaps(self):
        ms = 1_000_000
        times = np.array([0, 100, 200, 500, 600, 700, 800, 900]) * ms
        self.assertEqual(
            contiguous_segments(
                [0, 1, 2, 3, 4, 6, 7],
                times,
                max_gap_ms=150,
                min_steps=2,
            ),
            [[0, 1, 2], [3, 4], [6, 7]],
        )

    def test_state_interpolation_is_linear_and_orientation_uses_slerp(self):
        times = [0, 1_000_000_000]
        np.testing.assert_allclose(
            interpolate_vector(times, [[0, 2], [2, 4]], 250_000_000),
            [0.5, 2.5],
        )
        half = interpolate_quaternion_xyzw(
            times,
            [[0, 0, 0, 1], [0, 0, 1, 0]],
            500_000_000,
        )
        np.testing.assert_allclose(np.abs(half), [0, 0, 2 ** -0.5, 2 ** -0.5])

    def test_action_chunk_overwrites_future_slots_and_drops_stale_prefix(self):
        buffer = TimestampedActionBuffer(period_ns=100)
        accepted, stale = buffer.update(
            [[1], [2], [3]], [100, 200, 300], query_id=1, now_ns=150
        )
        self.assertEqual((accepted, stale), (2, 1))
        buffer.update([[20], [30]], [200, 300], query_id=2, now_ns=150)
        first = buffer.next(150)
        self.assertEqual(first.query_id, 2)
        np.testing.assert_array_equal(first.action, [20])
        self.assertEqual(buffer.coverage_end_ns(150), 300)

    def test_nearby_new_timestamps_replace_old_grid_slots(self):
        buffer = TimestampedActionBuffer(period_ns=100)
        buffer.update([[2], [3]], [200, 300], query_id=1, now_ns=100)
        buffer.update([[20], [30]], [205, 305], query_id=2, now_ns=150)
        first = buffer.next(150)
        self.assertEqual(first.target_time_ns, 205)
        self.assertEqual(first.query_id, 2)

    def test_execute_latency_profile_is_explicit_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latency.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "measured_at": "2026-08-27T12:00:00+08:00",
                        "front_observation_ms": 10,
                        "wrist_observation_ms": 11,
                        "robot_observation_ms": 4,
                        "gripper_observation_ms": 6,
                        "robot_action_ms": 12,
                        "gripper_action_ms": 20,
                    }
                )
            )
            profile = LatencyProfile.load(path)
            self.assertEqual(profile.robot_action_ms, 12)


if __name__ == "__main__":
    unittest.main()
