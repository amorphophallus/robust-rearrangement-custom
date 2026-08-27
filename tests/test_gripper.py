import unittest

import numpy as np
import torch

from src.common.gripper import (
    GRIPPER_CLOSED_VALUE,
    GRIPPER_OPEN_THRESHOLD_METERS,
    GRIPPER_OPEN_VALUE,
    binarize_gripper_width,
    binarize_robot_state_gripper_width,
    normalizer_expects_binary_gripper_width,
)
from src.dataset.normalizer import LinearNormalizer


class GripperWidthEncodingTest(unittest.TestCase):
    def test_numpy_widths_use_action_sign_convention(self):
        widths = np.array(
            [
                0.0,
                GRIPPER_OPEN_THRESHOLD_METERS - 1e-6,
                GRIPPER_OPEN_THRESHOLD_METERS,
                0.08,
            ],
            dtype=np.float32,
        )
        encoded = binarize_gripper_width(widths)
        np.testing.assert_array_equal(
            encoded,
            [
                GRIPPER_CLOSED_VALUE,
                GRIPPER_CLOSED_VALUE,
                GRIPPER_OPEN_VALUE,
                GRIPPER_OPEN_VALUE,
            ],
        )
        self.assertEqual(encoded.dtype, widths.dtype)

    def test_torch_widths_preserve_shape_dtype_and_device(self):
        widths = torch.tensor([[0.01], [0.065]], dtype=torch.float64)
        encoded = binarize_gripper_width(widths)
        torch.testing.assert_close(
            encoded,
            torch.tensor(
                [[GRIPPER_CLOSED_VALUE], [GRIPPER_OPEN_VALUE]],
                dtype=torch.float64,
            ),
        )
        self.assertEqual(encoded.device, widths.device)

    def test_robot_state_conversion_does_not_mutate_raw_state(self):
        raw_width = np.array([0.08], dtype=np.float32)
        expected_raw_width = raw_width.copy()
        raw_state = {"ee_pos": np.zeros(3), "gripper_width": raw_width}
        encoded = binarize_robot_state_gripper_width(raw_state)
        np.testing.assert_array_equal(raw_state["gripper_width"], expected_raw_width)
        np.testing.assert_array_equal(encoded["gripper_width"], [GRIPPER_OPEN_VALUE])

    def test_checkpoint_stats_preserve_legacy_metric_compatibility(self):
        metric = LinearNormalizer()
        metric.fit({"robot_state": torch.tensor([[0.0], [0.08]])})
        binary = LinearNormalizer()
        binary.fit({"robot_state": torch.tensor([[-1.0], [1.0]])})
        constant_metric = LinearNormalizer()
        constant_metric.fit({"robot_state": torch.tensor([[0.08], [0.08]])})
        constant_binary = LinearNormalizer()
        constant_binary.fit({"robot_state": torch.tensor([[-1.0], [-1.0]])})

        self.assertFalse(normalizer_expects_binary_gripper_width(metric))
        self.assertTrue(normalizer_expects_binary_gripper_width(binary))
        self.assertFalse(normalizer_expects_binary_gripper_width(constant_metric))
        self.assertTrue(normalizer_expects_binary_gripper_width(constant_binary))


if __name__ == "__main__":
    unittest.main()
