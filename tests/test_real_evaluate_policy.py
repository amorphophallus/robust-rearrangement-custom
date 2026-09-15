import contextlib
import io
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch

from src.real.evaluate_policy import (
    AsyncPolicyInference,
    PolicyInferenceRequest,
    RESET_JOINT_POSITIONS,
    _gripper_dispatch_decision,
    _move_to_reset_joint_positions,
    _parse_args,
    _prediction_shift_steps,
    _queue_warmstart,
    _start_camera_and_initialize_policy,
)
from src.real.time_alignment import IndependentActionQueues


class RealEvaluatePolicyCliTest(unittest.TestCase):
    def test_execute_requires_latency(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parse_args(["--checkpoint", "model.pt", "--execute"])

    def test_default_mode_is_dry_run(self):
        args = _parse_args(["--checkpoint", "model.pt"])
        self.assertFalse(args.execute)
        self.assertEqual(args.warmup_timeout_s, 120.0)
        self.assertEqual(args.frequency, 5.0)
        self.assertEqual(args.workspace_min, [0.30, -0.35, 0.00])
        self.assertEqual(args.workspace_max, [0.75, 0.35, 0.60])
        self.assertEqual(args.min_ee_z, 0.005)
        self.assertEqual(args.max_translation_step_m, 0.05)
        self.assertFalse(args.show_input_dashboard)
        self.assertFalse(args.save_input_video)
        self.assertEqual(args.query_interval_steps, 4)

    def test_query_interval_is_always_cli_controlled(self):
        for interval in (2, 3, 8):
            with self.subTest(interval=interval):
                args = _parse_args(
                    [
                        "--checkpoint",
                        "model.pt",
                        "--query-interval-steps",
                        str(interval),
                    ]
                )
                self.assertEqual(args.query_interval_steps, interval)

    def test_input_dashboard_is_opt_in(self):
        args = _parse_args(
            ["--checkpoint", "model.pt", "--show-input-dashboard"]
        )
        self.assertTrue(args.show_input_dashboard)

    def test_input_video_is_opt_in(self):
        args = _parse_args(
            ["--checkpoint", "model.pt", "--save-input-video"]
        )
        self.assertTrue(args.save_input_video)

    def test_execute_uses_measured_workspace_defaults(self):
        args = _parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--execute",
                "--latency-profile",
                "latency.json",
            ]
        )
        self.assertEqual(args.workspace_min, [0.30, -0.35, 0.00])
        self.assertEqual(args.workspace_max, [0.75, 0.35, 0.60])
        self.assertEqual(args.min_ee_z, 0.005)

    def test_warmup_timeout_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parse_args(
                    ["--checkpoint", "model.pt", "--warmup-timeout-s", "0"]
                )

    def test_execution_frequency_alias_controls_umi_period(self):
        args = _parse_args(
            ["--checkpoint", "model.pt", "--execution-frequency", "5"]
        )
        self.assertEqual(args.frequency, 5.0)

    def test_legacy_frequency_flag_remains_compatible(self):
        args = _parse_args(["--checkpoint", "model.pt", "--frequency", "2"])
        self.assertEqual(args.frequency, 2.0)

    def test_reset_target_matches_data_collection(self):
        self.assertEqual(
            RESET_JOINT_POSITIONS.tolist(),
            [
                0.0916502534874562,
                0.006205358472252432,
                -0.02085815329544379,
                -2.552429972459778,
                -0.010695882435351968,
                2.587622772050635,
                0.8472435743003388,
            ],
        )


class RealEvaluatePolicyStartupTest(unittest.TestCase):
    def test_realsense_starts_before_policy_initialization(self):
        calls = []

        class Camera:
            def start(self):
                calls.append("camera")

        def initialize(args):
            calls.append("policy")
            return args

        with mock.patch(
            "src.real.evaluate_policy._initialize_policy_runtime",
            side_effect=initialize,
        ):
            result = _start_camera_and_initialize_policy(Camera(), "args")

        self.assertEqual(result, "args")
        self.assertEqual(calls, ["camera", "policy"])

    def test_reset_does_not_send_when_already_at_collection_target(self):
        robot = mock.Mock()
        robot.last_q = RESET_JOINT_POSITIONS.copy()

        reached = _move_to_reset_joint_positions(
            robot,
            mock.sentinel.joint_controller,
            timeout=1.0,
            tolerance=1e-3,
            gripper_open=True,
        )

        self.assertTrue(reached)
        robot.control.assert_not_called()


class RealEvaluatePolicySchedulingTest(unittest.TestCase):
    def test_same_sign_gripper_is_noop_even_after_deadline(self):
        changed, expired = _gripper_dispatch_decision(
            desired_sign=-1,
            last_sign=-1,
            command_start_ns=1_100,
            target_time_ns=1_000,
            command_deadline_ns=400,
            max_lateness_ms=0.00001,
        )
        self.assertFalse(changed)
        self.assertFalse(expired)

    def test_stale_gripper_transition_is_retried_at_next_timestep(self):
        changed, expired = _gripper_dispatch_decision(
            desired_sign=1,
            last_sign=-1,
            command_start_ns=1_100,
            target_time_ns=1_000,
            command_deadline_ns=400,
            max_lateness_ms=0.00001,
        )
        self.assertTrue(changed)
        self.assertTrue(expired)

        changed, expired = _gripper_dispatch_decision(
            desired_sign=1,
            last_sign=-1,
            command_start_ns=1_150,
            target_time_ns=1_400,
            command_deadline_ns=1_100,
            max_lateness_ms=0.0001,
        )
        self.assertTrue(changed)
        self.assertFalse(expired)

    def test_warmstart_maps_immutable_absolute_actions_to_prediction_grid(self):
        queues = IndependentActionQueues(
            period_ns=200,
            arm_latency_ns=120,
            gripper_latency_ns=642,
        )
        plan = queues.plan_update(
            [[1, 2], [3, 4], [5, 6]],
            [1000, 1200, 1400],
            query_id=1,
            admission_cutoff_ns=0,
        )
        queues.reserve(plan.candidates)

        indices, actions = _queue_warmstart(
            queues,
            observation_time_ns=800,
            period_ns=200,
            pred_horizon=3,
            action_dim=2,
        )

        self.assertEqual(indices, (1, 2))
        np.testing.assert_array_equal(actions, [[1, 2], [3, 4]])

    def test_prediction_shift_uses_elapsed_observation_time(self):
        self.assertEqual(
            _prediction_shift_steps(
                current_observation_time_ns=1_600,
                previous_observation_time_ns=1_000,
                period_ns=200,
                default_steps=8,
            ),
            3,
        )
        self.assertEqual(
            _prediction_shift_steps(
                current_observation_time_ns=1_000,
                previous_observation_time_ns=None,
                period_ns=200,
                default_steps=8,
            ),
            8,
        )

    def test_blocking_inference_does_not_block_dispatch_caller(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingActor:
            def reset(self):
                pass

            def action_chunk(self, _observation, **_kwargs):
                started.set()
                release.wait(timeout=2.0)
                return torch.zeros((1, 2, 1))

        worker = AsyncPolicyInference(
            BlockingActor(), device="cpu", binary_gripper=False
        )
        request = PolicyInferenceRequest(
            rollout_generation=1,
            query_id=0,
            observation={},
            timing={"observation_time_ns": 1_000},
            period_ns=200,
            warmstart_indices=(),
            warmstart_actions=np.empty((0, 1), dtype=np.float32),
        )
        try:
            with mock.patch(
                "src.real.evaluate_policy._policy_observation",
                return_value={},
            ):
                self.assertTrue(worker.submit(request))
                self.assertTrue(started.wait(timeout=1.0))
                queues = IndependentActionQueues(
                    period_ns=200,
                    arm_latency_ns=120,
                    gripper_latency_ns=642,
                )
                plan = queues.plan_update(
                    [[1]], [1000], query_id=1, admission_cutoff_ns=0
                )
                queues.reserve(plan.candidates)
                self.assertEqual(queues.next_dispatch().channel, "gripper")
                self.assertIsNone(worker.poll())
                release.set()
                result = None
                deadline = time.monotonic() + 1.0
                while result is None and time.monotonic() < deadline:
                    result = worker.poll()
                    time.sleep(0.005)
                self.assertIsNotNone(result)
                self.assertIsNone(result.error)
        finally:
            release.set()
            worker.stop()


if __name__ == "__main__":
    unittest.main()
