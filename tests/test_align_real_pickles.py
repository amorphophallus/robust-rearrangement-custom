import unittest

import numpy as np

from src.real.align_pickles import align_trajectory


def _observation(step, camera_step):
    camera_ns = camera_step * 100_000_000
    control_ns = step * 100_000_000
    pose = np.eye(4, dtype=np.float32)
    return {
        "control_wall_time_ns": control_ns,
        "camera_capture_wall_time_ns": camera_ns + 2_000_000,
        "prompt_depth_source_wall_time_ns": camera_ns,
        "front_sensor_timestamp_ms": camera_ns / 1e6,
        "wrist_sensor_timestamp_ms": (camera_ns + 1_000_000) / 1e6,
        "front_frame_number": camera_step,
        "wrist_frame_number": camera_step,
        "color_image1": np.full((2, 2, 3), camera_step, dtype=np.uint8),
        "color_image2": np.full((2, 2, 3), camera_step + 10, dtype=np.uint8),
        "depth_image1": np.full((2, 2), camera_step, dtype=np.float16),
        "depth_image2": np.full((2, 2), camera_step, dtype=np.float16),
        "robot_state": {
            "ee_pose": pose,
            "wrist_pose": pose,
            "gripper_width": 0.08,
        },
        "parts_poses": np.zeros(7, dtype=np.float32),
        "parts_founds": np.ones(1, dtype=bool),
    }


def _action():
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0])


class AlignRealPicklesTest(unittest.TestCase):
    def test_alignment_uses_action_state_and_unique_camera_sources(self):
        # camera frame 1 is repeated in two control-loop observations.  It may
        # only label one policy timestep.
        camera_steps = [0, 1, 1, 3, 4, 5, 6, 7, 8]
        observations = [
            _observation(step, camera_step)
            for step, camera_step in enumerate(camera_steps)
        ]
        trajectory = {
            "observations": observations,
            "actions": [_action() for _ in range(8)],
            "actions_original": [_action() for _ in range(8)],
            "rewards": [0.0] * 8,
            "action_type": "delta",
            "task": "one_leg",
            "furniture": "one_leg",
            "camera_info": {},
            "metadata": {},
        }
        outputs, report = align_trajectory(
            trajectory,
            max_camera_residual_ms=49,
            max_action_gap_ms=150,
            min_segment_steps=2,
            rerun_annotations=False,
        )
        self.assertEqual([len(output["actions"]) for output in outputs], [2, 5])
        self.assertEqual(report["retained_actions"], 7)
        self.assertEqual(report["reason_counts"]["missing_both_cameras"], 1)
        first = outputs[0]["observations"][1]
        self.assertEqual(first["state_source_index"], 1)
        self.assertEqual(first["front_source_index"], 1)
        self.assertEqual(first["camera_anchor"], "front")
        self.assertEqual(
            outputs[0]["metadata"]["legacy_rotation_scale_source"],
            "unsplit_source_episode",
        )

    def test_non_delta_legacy_input_is_rejected(self):
        trajectory = {
            "observations": [_observation(0, 0)],
            "actions": [_action()],
            "action_type": "pos",
        }
        with self.assertRaisesRegex(ValueError, "expects saved delta"):
            align_trajectory(trajectory, rerun_annotations=False)

    def test_new_recorder_action_timestamps_are_the_alignment_master(self):
        observations = [_observation(index, index) for index in range(3)]
        trajectory = {
            "observations": observations,
            "actions": [_action(), _action()],
            "action_timestamps_ns": [10_000_000, 110_000_000],
            "action_type": "delta",
        }
        _, report = align_trajectory(
            trajectory,
            max_camera_residual_ms=20,
            min_segment_steps=1,
            rerun_annotations=False,
        )
        self.assertEqual(report["action_time_source"], "action_timestamps_ns")
        self.assertEqual(report["steps"][0]["action_time_ns"], 10_000_000)

    def test_v5_prefers_target_time_and_interpolates_timestamped_state(self):
        observations = [_observation(index, index) for index in range(3)]
        for index, observation in enumerate(observations):
            timestamp_ns = index * 100_000_000
            pose = np.eye(4, dtype=np.float64)
            pose[0, 3] = float(index)
            observation["robot_state_receive_wall_time_ns"] = timestamp_ns
            observation["gripper_state_receive_wall_time_ns"] = timestamp_ns
            observation["robot_state"]["ee_pose"] = pose
            observation["robot_state"]["wrist_pose"] = pose
            observation["robot_state"]["ee_pos"] = pose[:3, 3]
            observation["robot_state"]["ee_quat"] = [0.0, 0.0, 0.0, 1.0]
            observation["robot_state"]["gripper_width"] = 0.08 - index * 0.02
        trajectory = {
            "observations": observations,
            "actions": [_action(), _action()],
            "actions_absolute": [_action(), _action()],
            "action_timestamps_ns": [0, 100_000_000],
            "action_target_timestamps_ns": [100_000_000, 200_000_000],
            "action_type": "delta",
            "metadata": {
                "schema": "deoxys_furniturebench_raw_v5_target_time",
                "robot_observation_latency_ms": 0,
                "gripper_observation_latency_ms": 0,
            },
        }

        outputs, report = align_trajectory(
            trajectory,
            max_camera_residual_ms=10,
            max_camera_pair_skew_ms=10,
            min_segment_steps=1,
            rerun_annotations=False,
        )

        self.assertEqual(
            report["action_time_source"], "action_target_timestamps_ns"
        )
        self.assertEqual(report["retained_actions"], 2)
        self.assertEqual(outputs[0]["action_target_timestamps_ns"], [100_000_000, 200_000_000])
        self.assertIsNone(outputs[0]["observations"][0]["state_source_index"])
        np.testing.assert_allclose(
            outputs[0]["observations"][1]["robot_state"]["wrist_pose"][:3, 3],
            [2.0, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
