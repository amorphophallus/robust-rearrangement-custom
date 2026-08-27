import unittest

import numpy as np

from src.common.eepose import (
    REAL_TIP,
    ROBOT_BASE,
    SIM_LOCAL,
    resolve_eepose_frame,
    select_policy_eepose,
)


def _state():
    return {
        "ee_pos": np.array([1.0, 2.0, 3.0]),
        "ee_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "ee_pose": np.eye(4),
        "ee_pos_sim": np.array([4.0, 5.0, 6.0]),
        "ee_quat_sim": np.array([0.0, 0.0, 1.0, 0.0]),
        "ee_pos_original": np.array([7.0, 8.0, 9.0]),
        "ee_quat_original": np.array([0.0, 1.0, 0.0, 0.0]),
    }


class EEPoseFrameTest(unittest.TestCase):
    def test_resolve_eepose_frame(self):
        cases = (
            ("robot-base", SIM_LOCAL, ROBOT_BASE),
            ("original", SIM_LOCAL, SIM_LOCAL),
            ("sim-local", SIM_LOCAL, SIM_LOCAL),
            ("original", REAL_TIP, REAL_TIP),
            ("real-tip", REAL_TIP, REAL_TIP),
            ("base", REAL_TIP, ROBOT_BASE),
        )
        for spec, original, expected in cases:
            with self.subTest(spec=spec, original=original):
                self.assertEqual(
                    resolve_eepose_frame(spec, original_frame=original), expected
                )
        self.assertEqual(
            resolve_eepose_frame("original", original_frame=SIM_LOCAL),
            resolve_eepose_frame("sim-local", original_frame=SIM_LOCAL),
        )

    def test_select_policy_eepose_keeps_interface_and_input_unchanged(self):
        state = _state()
        selected = select_policy_eepose(state, "original", original_frame=SIM_LOCAL)
        np.testing.assert_array_equal(selected["ee_pos"], state["ee_pos_sim"])
        np.testing.assert_array_equal(selected["ee_quat"], state["ee_quat_sim"])
        np.testing.assert_array_equal(state["ee_pos"], [1.0, 2.0, 3.0])
        self.assertTrue(set(state).issubset(selected))

    def test_select_real_original_uses_preserved_tip_fields(self):
        state = _state()
        selected = select_policy_eepose(state, "original", original_frame=REAL_TIP)
        np.testing.assert_array_equal(selected["ee_pos"], state["ee_pos_original"])
        np.testing.assert_array_equal(
            selected["ee_quat"], state["ee_quat_original"]
        )

    def test_invalid_or_unavailable_frame_fails_loudly(self):
        with self.assertRaises(ValueError):
            resolve_eepose_frame("camera", original_frame=SIM_LOCAL)
        with self.assertRaises(ValueError):
            resolve_eepose_frame("original=sim-local", original_frame=SIM_LOCAL)
        state = _state()
        state.pop("ee_pos_sim")
        with self.assertRaises(KeyError):
            select_policy_eepose(state, "sim-local", original_frame=SIM_LOCAL)


if __name__ == "__main__":
    unittest.main()
