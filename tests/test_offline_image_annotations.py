import unittest

import numpy as np

from src.data_processing.offline_image_annotations import annotate_observation_image
from src.common.image_annotations import (
    draw_grasp_annotation_on_image,
    draw_guidance_point_on_image,
    resize_guidance_point_for_image,
)


def make_observation(skill="pick"):
    return {
        "color_image1": np.zeros((40, 40, 3), dtype=np.uint8),
        "color_image2": np.zeros((40, 40, 3), dtype=np.uint8),
        "skill": skill,
        "guidance_point_2d": {
            "color_image2": np.array([20.0, 20.0], dtype=np.float32),
        },
        "guidance_point_clean": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "guidance_point_n2": np.array([0.1, 0.0, 1.0], dtype=np.float32),
        "guidance_point_n4": np.array([0.4, 0.0, 1.0], dtype=np.float32),
        "guidance_point_2d_n2": {
            "color_image2": np.array([22.0, 20.0], dtype=np.float32),
        },
        "guidance_point_2d_n4": {
            "color_image2": np.array([28.0, 20.0], dtype=np.float32),
        },
        "grasp_annotation_2d": {
            "color_image2": {
                "style": "grasp_rect",
                "center": np.array([20.0, 20.0], dtype=np.float32),
                "corners": np.array(
                    [[12.0, 17.0], [28.0, 17.0], [28.0, 23.0], [12.0, 23.0]],
                    dtype=np.float32,
                ),
            },
        },
    }


class OfflineImageAnnotationTest(unittest.TestCase):
    def test_default_guidance_point_is_half_alpha_rgb_red(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        output = draw_guidance_point_on_image(image, np.array([20.0, 20.0]))

        self.assertEqual(output[20, 20, 0], 128)
        self.assertEqual(output[20, 20, 1], 0)
        self.assertEqual(output[20, 20, 2], 0)

    def test_guidance_resize_matches_simulator_front_center_crop(self):
        output = resize_guidance_point_for_image(
            np.array([160.0, 120.0], dtype=np.float32),
            image_key="color_image2",
            source_image_size=(320, 240),
            image_shape=(224, 224, 3),
        )

        np.testing.assert_allclose(output, [112.0, 112.0], atol=1e-6)

    def test_none_keeps_pixels_and_does_not_alias_observation_dict(self):
        observation = make_observation()
        output = annotate_observation_image(observation, "none")

        self.assertIsNot(output, observation)
        np.testing.assert_array_equal(
            output["color_image2"], observation["color_image2"]
        )

    def test_guidance_point_modes_only_change_front_camera(self):
        observation = make_observation(skill="pick")
        mono = annotate_observation_image(observation, "guidance-point")
        colored = annotate_observation_image(observation, "guidance-point-colored")

        np.testing.assert_array_equal(
            mono["color_image1"], observation["color_image1"]
        )
        self.assertGreater(np.count_nonzero(mono["color_image2"]), 0)
        self.assertGreater(np.count_nonzero(colored["color_image2"]), 0)
        self.assertFalse(
            np.array_equal(mono["color_image2"], colored["color_image2"])
        )
        np.testing.assert_array_equal(observation["color_image2"], 0)

    def test_robot_base_point_is_reprojected_instead_of_using_stale_saved_pixel(self):
        observation = make_observation(skill="push")
        observation["guidance_point_2d"]["color_image2"] = np.array([5.0, 5.0])
        observation["guidance_point"] = np.array([0.0, 0.0, 1.0])
        camera_info = {
            "front_camera": {
                "image_size": np.array([40, 40]),
                "intrinsics": np.array(
                    [[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]
                ),
                "robot_base_to_camera": np.eye(4),
            }
        }

        output = annotate_observation_image(
            observation,
            "guidance-point",
            trajectory_camera_info=camera_info,
        )

        self.assertGreater(output["color_image2"][20, 20, 0], 0)
        np.testing.assert_array_equal(output["color_image2"][5, 5], 0)

    def test_grasp_part_uses_grasp_rectangle_for_grasp_skills(self):
        for skill in ("pick", "place"):
            with self.subTest(skill=skill):
                observation = make_observation(skill=skill)
                output = annotate_observation_image(observation, "grasp-part")
                expected = draw_grasp_annotation_on_image(
                    observation["color_image2"],
                    observation["grasp_annotation_2d"]["color_image2"],
                    skill=skill,
                    use_skill_color=False,
                )
                np.testing.assert_array_equal(output["color_image2"], expected)

    def test_grasp_part_uses_guidance_point_for_other_skills(self):
        for skill in ("insert", "screw", "push"):
            with self.subTest(skill=skill):
                observation = make_observation(skill=skill)
                output = annotate_observation_image(
                    observation, "grasp-part-colored"
                )
                expected = draw_guidance_point_on_image(
                    observation["color_image2"],
                    observation["guidance_point_2d"]["color_image2"],
                    skill=skill,
                    use_skill_color=True,
                )
                np.testing.assert_array_equal(output["color_image2"], expected)

    def test_noisy_guidance_point_uses_requested_frozen_level(self):
        observation = make_observation(skill="push")
        output = annotate_observation_image(
            observation, "guidance-point", guidance_noise_level="n4"
        )
        self.assertGreater(output["color_image2"][20, 28, 0], 0)
        np.testing.assert_array_equal(output["color_image2"][20, 20], 0)

    def test_noisy_grasp_translates_pose_without_rotating_rectangle(self):
        observation = make_observation(skill="pick")
        observation["guidance_pose_clean"] = np.eye(4, dtype=np.float32)
        observation["guidance_pose_clean"][2, 3] = 1.0
        camera_info = {
            "front_camera": {
                "image_size": np.array([40, 40]),
                "intrinsics": np.array(
                    [[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]
                ),
                "robot_base_to_camera": np.eye(4),
            }
        }
        clean = annotate_observation_image(
            observation,
            "grasp-part",
            trajectory_camera_info=camera_info,
            guidance_noise_level="clean",
        )
        noisy = annotate_observation_image(
            observation,
            "grasp-part",
            trajectory_camera_info=camera_info,
            guidance_noise_level="n2",
        )
        clean_x = np.nonzero(clean["color_image2"])[1].mean()
        noisy_x = np.nonzero(noisy["color_image2"])[1].mean()
        self.assertGreater(noisy_x, clean_x)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown image annotation mode"):
            annotate_observation_image(make_observation(), "unknown")


if __name__ == "__main__":
    unittest.main()
