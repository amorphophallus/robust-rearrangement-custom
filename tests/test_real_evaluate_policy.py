import contextlib
import io
import unittest
from unittest import mock

from src.real.evaluate_policy import _parse_args, _start_camera_and_initialize_policy


class RealEvaluatePolicyCliTest(unittest.TestCase):
    def test_execute_requires_latency_and_explicit_workspace(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parse_args(["--checkpoint", "model.pt", "--execute"])

    def test_default_mode_is_dry_run(self):
        args = _parse_args(["--checkpoint", "model.pt"])
        self.assertFalse(args.execute)
        self.assertEqual(args.warmup_timeout_s, 120.0)

    def test_warmup_timeout_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parse_args(
                    ["--checkpoint", "model.pt", "--warmup-timeout-s", "0"]
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


if __name__ == "__main__":
    unittest.main()
