import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.real.time_alignment import (
    IndependentActionQueues,
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

    def test_profile_directory_selects_latest_profile_measured_today(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            now = datetime(2026, 9, 8, 17, 0, tzinfo=timezone(timedelta(hours=8)))

            def write_profile(name, measured_at):
                path = directory / name
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "measured_at": measured_at.isoformat(),
                            "latency_source": "measured",
                            "basis": "test",
                            "front_observation_ms": 1,
                            "wrist_observation_ms": 1,
                            "robot_observation_ms": 1,
                            "gripper_observation_ms": 1,
                            "robot_action_ms": 1,
                            "gripper_action_ms": 1,
                        }
                    )
                )
                return path

            write_profile("latency_profile-yesterday.json", now - timedelta(days=1))
            write_profile("latency_profile-early.json", now - timedelta(hours=2))
            latest = write_profile(
                "latency_profile-latest.json", now - timedelta(minutes=5)
            )

            self.assertEqual(
                LatencyProfile.resolve_path(directory, now=now), latest.resolve()
            )

    def test_profile_directory_refuses_to_fall_back_to_previous_day(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            now = datetime(2026, 9, 8, 1, 0, tzinfo=timezone(timedelta(hours=8)))
            (directory / "latency_profile-old.json").write_text(
                json.dumps(
                    {
                        "measured_at": (now - timedelta(days=1)).isoformat(),
                    }
                )
            )
            with self.assertRaisesRegex(FileNotFoundError, "no latency_profile"):
                LatencyProfile.resolve_path(directory, now=now)

    def test_v2_estimated_latency_profile_records_stale_guard(self):
        profile = LatencyProfile.from_mapping(
            {
                "schema_version": 2,
                "measured_at": "2026-09-01T12:00:00+08:00",
                "latency_source": "estimated",
                "basis": "Deoxys command_latency=0.01s",
                "action_stale_guard_ms": 10,
                "front_observation_ms": 0,
                "wrist_observation_ms": 0,
                "robot_observation_ms": 4,
                "gripper_observation_ms": 6,
                "robot_action_ms": 10,
                "gripper_action_ms": 10,
            }
        )

        self.assertEqual(profile.common_action_lead_ms, 20)
        self.assertEqual(profile.latency_source, "estimated")

    def test_independent_queues_dispatch_gripper_5_6_7_before_arm_5_6_7(self):
        queues = IndependentActionQueues(
            period_ns=200,
            arm_latency_ns=120,
            gripper_latency_ns=642,
        )
        plan = queues.plan_update(
            np.arange(16).reshape(8, 2),
            np.arange(8) * 200,
            query_id=4,
            admission_cutoff_ns=900,
        )
        self.assertEqual(plan.stale, 5)
        self.assertEqual([item.chunk_index for item in plan.candidates], [5, 6, 7])
        queues.reserve(plan.candidates)

        order = []
        while len(queues):
            dispatch = queues.next_dispatch()
            order.append((dispatch.channel, dispatch.scheduled.chunk_index))
            queues.consume(dispatch.channel, dispatch.scheduled.target_time_ns)

        self.assertEqual(
            order,
            [
                ("gripper", 5),
                ("gripper", 6),
                ("gripper", 7),
                ("arm", 5),
                ("arm", 6),
                ("arm", 7),
            ],
        )

    def test_overlap_preserves_occupied_timeline_and_only_appends_tail(self):
        queues = IndependentActionQueues(
            period_ns=200,
            arm_latency_ns=120,
            gripper_latency_ns=642,
        )
        initial = queues.plan_update(
            [[1], [2], [3]],
            [1000, 1200, 1400],
            query_id=1,
            admission_cutoff_ns=0,
        )
        queues.reserve(initial.candidates)

        overlap = queues.plan_update(
            [[10], [20], [30], [40]],
            [1005, 1205, 1405, 1605],
            query_id=2,
            admission_cutoff_ns=0,
        )

        self.assertEqual(overlap.occupied, 3)
        self.assertEqual(len(overlap.candidates), 1)
        self.assertEqual(overlap.candidates[0].target_time_ns, 1605)
        queues.reserve(overlap.candidates)
        np.testing.assert_array_equal(
            queues.future_reservations(0)[0].action,
            [1],
        )
        self.assertEqual(queues.reservation_tail_ns, 1605)

    def test_one_channel_consumption_does_not_clear_later_actions(self):
        queues = IndependentActionQueues(
            period_ns=200,
            arm_latency_ns=120,
            gripper_latency_ns=642,
        )
        plan = queues.plan_update(
            [[1], [2]],
            [1000, 1200],
            query_id=1,
            admission_cutoff_ns=0,
        )
        queues.reserve(plan.candidates)

        self.assertFalse(queues.consume("gripper", 1000))
        self.assertEqual(queues.pending_count("gripper"), 1)
        self.assertEqual(queues.pending_count("arm"), 2)

    def test_sent_reservation_remains_occupied_until_target_passes(self):
        queues = IndependentActionQueues(
            period_ns=200,
            arm_latency_ns=120,
            gripper_latency_ns=642,
        )
        plan = queues.plan_update(
            [[1]], [1000], query_id=1, admission_cutoff_ns=0
        )
        queues.reserve(plan.candidates)
        queues.consume("arm", 1000)
        queues.consume("gripper", 1000)

        self.assertEqual(len(queues.future_reservations(999)), 1)
        self.assertEqual(len(queues.future_reservations(1000)), 0)
        overlap = queues.plan_update(
            [[2]], [1005], query_id=2, admission_cutoff_ns=0
        )
        self.assertEqual(overlap.occupied, 1)


if __name__ == "__main__":
    unittest.main()
