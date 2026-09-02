import contextlib
import io
import unittest
from unittest import mock

from src.real.evaluate_policy import (
    RESET_JOINT_POSITIONS,
    _move_to_reset_joint_positions,
    _parse_args,
    _start_camera_and_initialize_policy,
)


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


if __name__ == "__main__":
    unittest.main()
