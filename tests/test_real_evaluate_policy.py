import contextlib
import io
import unittest

from src.real.evaluate_policy import _parse_args


class RealEvaluatePolicyCliTest(unittest.TestCase):
    def test_execute_requires_latency_and_explicit_workspace(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parse_args(["--checkpoint", "model.pt", "--execute"])

    def test_default_mode_is_dry_run(self):
        args = _parse_args(["--checkpoint", "model.pt"])
        self.assertFalse(args.execute)


if __name__ == "__main__":
    unittest.main()
