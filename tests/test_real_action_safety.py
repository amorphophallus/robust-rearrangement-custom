import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.real.action_safety import (
    ActionSafetyLimits,
    rotation_6d_to_matrix,
    validate_absolute_action,
)


class RealActionSafetyTest(unittest.TestCase):
    def setUp(self):
        self.limits = ActionSafetyLimits(
            workspace_min=np.array([0.2, -0.4, 0.02]),
            workspace_max=np.array([0.8, 0.4, 0.8]),
            min_ee_z=0.04,
            max_translation_step_m=0.03,
            max_rotation_step_rad=0.4,
            max_translation_speed_m_s=0.3,
            max_rotation_speed_rad_s=4.0,
        )

    def test_identity_rotation_6d_matches_rr_row_convention(self):
        np.testing.assert_allclose(
            rotation_6d_to_matrix([1, 0, 0, 0, 1, 0]), np.eye(3)
        )

    def test_valid_action_converts_to_absolute_axis_angle(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.5, 0.0, 0.2]
        rotation = Rotation.from_euler("z", 0.1).as_matrix()
        action = np.r_[pose[:3, 3] + [0.01, 0, 0], rotation[:2].reshape(-1), 1]
        result = validate_absolute_action(
            action, reference_pose=pose, period_s=0.1, limits=self.limits
        )
        np.testing.assert_allclose(result.axis_angle, [0, 0, 0.1], atol=1e-7)
        self.assertEqual(result.deoxys_action().shape, (7,))

    def test_large_step_is_rejected_instead_of_clipped(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.5, 0.0, 0.2]
        action = np.r_[[0.6, 0.0, 0.2], [1, 0, 0, 0, 1, 0], -1]
        with self.assertRaisesRegex(ValueError, "translation step"):
            validate_absolute_action(
                action, reference_pose=pose, period_s=0.1, limits=self.limits
            )


if __name__ == "__main__":
    unittest.main()
