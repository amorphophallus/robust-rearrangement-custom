import unittest

import numpy as np

from src.real.v6_pickle_contract import (
    V6_BUFFERED_SCHEMA,
    V6PickleContractError,
    validate_v6_buffered_trajectory,
)


def trajectory(frame_count=2):
    start = 1_700_000_000_000_000_000
    targets = start + np.arange(frame_count, dtype=np.int64) * 100_000_000
    observations = []
    for target in targets:
        observations.append(
            {
                "observation_target_wall_time_ns": int(target),
                "color_image1": np.zeros((2, 2, 3), dtype=np.uint8),
                "color_image2": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth_image1": np.ones((2, 2), dtype=np.float32),
                "depth_image2": np.ones((2, 2), dtype=np.float32),
                "depth_image1_realsense": np.ones((2, 2), dtype=np.float32),
                "depth_image2_realsense": np.ones((2, 2), dtype=np.float32),
                "robot_state": {},
                "parts_poses": np.zeros(42, dtype=np.float32),
                "skill": "pick",
                "guidance_point_2d": {
                    "color_image1": None,
                    "color_image2": None,
                },
            }
        )
    timing = [
        {"action_target_wall_time_ns": int(target)} for target in targets
    ]
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0])
    return {
        "env": "FurnitureBench",
        "task": "one_leg",
        "annotation_source": "scripted",
        "image_annotation_mode": "none",
        "observations": observations,
        "actions": np.repeat(action[None], frame_count, axis=0),
        "actions_original": np.repeat(action[None], frame_count, axis=0),
        "actions_absolute": np.repeat(action[None], frame_count, axis=0),
        "action_timing": timing,
        "action_target_timestamps_ns": targets,
        "action_timestamps_ns": targets.copy(),
        "obs_valid": np.ones(frame_count, dtype=bool),
        "rewards": np.zeros(frame_count),
        "metadata": {
            "schema": V6_BUFFERED_SCHEMA,
            "action_period_ns": 100_000_000,
            "recording_frequency_hz": 10.0,
            "real_skill_annotation": {"mode": "offline", "complete": True},
            "annotation_provenance": {
                "source": "scripted",
                "stage": "after_target_time_selection",
                "rgb_pixels_modified": False,
            },
            "offline_buffer_alignment": {"matched": frame_count},
            "prompt_depth_anything": {
                "online": False,
                "cameras": ["wrist", "front"],
            },
        },
    }


class V6PickleContractTest(unittest.TestCase):
    def test_accepts_dense_continuous_offline_episode(self):
        summary = validate_v6_buffered_trajectory(
            trajectory(), verify_projection=False
        )
        self.assertEqual(summary["frames"], 2)

    def test_rejects_gap_and_vlm_metadata(self):
        data = trajectory()
        data["action_target_timestamps_ns"][1] += 100_000_000
        data["action_timestamps_ns"][1] += 100_000_000
        with self.assertRaisesRegex(V6PickleContractError, "continuous 10 Hz"):
            validate_v6_buffered_trajectory(data, verify_projection=False)

        data = trajectory()
        data["metadata"]["vlm_prediction"] = {}
        with self.assertRaisesRegex(V6PickleContractError, "VLM metadata"):
            validate_v6_buffered_trajectory(data, verify_projection=False)


if __name__ == "__main__":
    unittest.main()
