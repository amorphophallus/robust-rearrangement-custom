import unittest

import numpy as np

from src.data_collection.pickle_contract import (
    flattened_poses_to_robot_base,
    legacy_sim_local_to_robot_base_matrix,
    point_to_robot_base,
    pose_to_robot_base,
    robot_state_with_base_frame_aliases,
    validate_and_align_pickle_timeseries,
)


class PickleStateContractTest(unittest.TestCase):
    def test_legacy_geometry_is_transformed_to_robot_base(self):
        state = {
            "ee_pos": np.array([0.5, 0.1, 0.2], dtype=np.float32),
            "ee_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "ee_pos_sim": np.array([0.2, 0.1, 0.615], dtype=np.float32),
            "ee_quat_sim": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        sim_local_to_base = legacy_sim_local_to_robot_base_matrix(state)
        expected_translation = np.array([0.3, 0.0, -0.415], dtype=np.float32)
        np.testing.assert_allclose(
            sim_local_to_base[:3, 3], expected_translation, atol=1.0e-7
        )

        point = np.array([0.2, -0.1, 0.5], dtype=np.float32)
        np.testing.assert_allclose(
            point_to_robot_base(point, sim_local_to_base),
            point + expected_translation,
            atol=1.0e-7,
        )

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = point
        np.testing.assert_allclose(
            pose_to_robot_base(pose, sim_local_to_base)[:3, 3],
            point + expected_translation,
            atol=1.0e-7,
        )
        flattened = np.concatenate(
            (point, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        )
        np.testing.assert_allclose(
            flattened_poses_to_robot_base(flattened, sim_local_to_base)[:3],
            point + expected_translation,
            atol=1.0e-7,
        )

    def test_legacy_sim_pose_keys_are_base_frame_aliases(self):
        original = {
            "ee_pos": np.array([0.5, 0.1, 0.2], dtype=np.float32),
            "ee_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "ee_pos_sim": np.array([0.2, 0.1, 0.615], dtype=np.float32),
            "ee_quat_sim": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "ee_pos_vel": np.zeros(3, dtype=np.float32),
        }

        normalized = robot_state_with_base_frame_aliases(original)

        self.assertIsNot(normalized, original)
        np.testing.assert_array_equal(normalized["ee_pos_sim"], original["ee_pos"])
        np.testing.assert_array_equal(normalized["ee_quat_sim"], original["ee_quat"])
        np.testing.assert_array_equal(
            original["ee_pos_sim"], np.array([0.2, 0.1, 0.615], dtype=np.float32)
        )

    def test_flat_legacy_state_is_unchanged(self):
        flat = np.zeros(14, dtype=np.float32)
        self.assertIs(robot_state_with_base_frame_aliases(flat), flat)

    def test_timeseries_contract_trims_legacy_reward_tail(self):
        observations = [{}, {}, {}]
        actions = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        aligned_actions, aligned_rewards = validate_and_align_pickle_timeseries(
            observations, actions, [0.0, 1.0, 99.0]
        )
        self.assertEqual(aligned_actions.shape, (2, 8))
        np.testing.assert_array_equal(aligned_rewards, [0.0, 1.0])

    def test_timeseries_contract_rejects_wrong_observation_count(self):
        actions = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]],
            dtype=np.float32,
        )
        with self.assertRaisesRegex(ValueError, r"T\+1 observations"):
            validate_and_align_pickle_timeseries([{}], actions, [0.0])


if __name__ == "__main__":
    unittest.main()
