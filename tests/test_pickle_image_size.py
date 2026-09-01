import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from src.common.vision import FrontCameraTransform
from src.data_collection.pickle_contract import (
    camera_calibration_to_robot_base,
    center_crop_camera_calibration,
    center_crop_grasp_mapping,
    center_crop_observation_images,
    center_crop_point_mapping,
    normalize_depth_meters,
)
from src.data_processing.utils import clip_quat_xyzw_magnitude


class PickleImageSizeTest(unittest.TestCase):
    def test_center_crops_rgb_and_depth_with_identical_offsets(self):
        rows = np.arange(240, dtype=np.float32)[:, None]
        cols = np.arange(320, dtype=np.float32)[None, :]
        depth = rows * 1000 + cols
        color = np.repeat(depth[..., None], 3, axis=-1).astype(np.uint8)

        outputs = center_crop_observation_images(
            color[None], color[None], depth[None], depth[None], 224
        )

        color1, color2, depth1, depth2 = outputs
        self.assertEqual(color1.shape, (1, 224, 224, 3))
        self.assertEqual(color2.shape, (1, 224, 224, 3))
        self.assertEqual(depth1.shape, (1, 224, 224))
        self.assertEqual(depth2.shape, (1, 224, 224))
        np.testing.assert_array_equal(color1[0, ..., 0], depth1[0].astype(np.uint8))
        self.assertEqual(float(depth1[0, 0, 0]), 8_048.0)
        self.assertEqual(float(depth1[0, -1, -1]), 231_271.0)

    def test_rejects_rgb_depth_shape_mismatch(self):
        color = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        depth = np.zeros((1, 223, 224), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "matching T/H/W"):
            center_crop_observation_images(color, color, depth, depth, 224)

    def test_rejects_too_small_input(self):
        color = np.zeros((1, 200, 224, 3), dtype=np.uint8)
        depth = np.zeros((1, 200, 224), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "too small"):
            center_crop_observation_images(color, color, depth, depth, 224)

    def test_front_training_transform_does_not_pad_canonical_frames(self):
        transform = FrontCameraTransform(mode="train")
        canonical_rgbd = torch.ones((2, 4, 224, 224), dtype=torch.float32)
        transformed = transform(canonical_rgbd)
        self.assertEqual(tuple(transformed.shape), (2, 4, 224, 224))
        torch.testing.assert_close(transformed[:, 3:], canonical_rgbd[:, 3:])

    def test_crop_updates_annotations_and_front_intrinsics(self):
        source_shapes = {"color_image2": (240, 320)}
        point = center_crop_point_mapping(
            {"color_image2": np.array([200.0, 156.0], dtype=np.float32)},
            source_shapes,
            224,
        )
        np.testing.assert_array_equal(point["color_image2"], [152.0, 148.0])

        grasp = center_crop_grasp_mapping(
            {
                "color_image2": {
                    "style": "grasp_rect",
                    "center": np.array([160.0, 153.0], dtype=np.float32),
                    "corners": np.array(
                        [[150.0, 152.0], [170.0, 152.0], [170.0, 154.0]],
                        dtype=np.float32,
                    ),
                }
            },
            source_shapes,
            224,
        )
        np.testing.assert_array_equal(
            grasp["color_image2"]["center"], [112.0, 145.0]
        )
        np.testing.assert_array_equal(
            grasp["color_image2"]["corners"][0], [102.0, 144.0]
        )

        calibration = center_crop_camera_calibration(
            {
                "image_size": np.array([320, 240], dtype=np.int32),
                "intrinsics": np.array(
                    [[307.0, 0.0, 160.0], [0.0, 308.0, 120.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            },
            224,
        )
        np.testing.assert_array_equal(calibration["image_size"], [224, 224])
        self.assertEqual(float(calibration["intrinsics"][0, 2]), 112.0)
        self.assertEqual(float(calibration["intrinsics"][1, 2]), 112.0)

    def test_rotation_clip_is_independent_for_each_action(self):
        quaternions = R.from_rotvec(
            np.array([[0.0, 0.0, 0.10], [0.0, 0.0, 0.50]], dtype=np.float32)
        ).as_quat()

        clipped = clip_quat_xyzw_magnitude(quaternions, clip_mag=0.35)
        magnitudes = np.linalg.norm(R.from_quat(clipped).as_rotvec(), axis=-1)

        np.testing.assert_allclose(magnitudes, [0.10, 0.35], atol=1e-7)

    def test_depth_contract_converts_legacy_fb_axis_to_positive_meters(self):
        legacy = np.array([[-1.25, -0.0], [-0.4, -2.0]], dtype=np.float64)
        normalized = normalize_depth_meters(legacy)

        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(
            normalized, np.array([[1.25, 0.0], [0.4, 2.0]], dtype=np.float32)
        )
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            normalize_depth_meters(np.array([[np.nan]], dtype=np.float32))

    def test_camera_extrinsic_is_converted_from_world_to_robot_base(self):
        camera_to_world = np.eye(4, dtype=np.float32)
        camera_to_world[:3, 3] = [0.9, 0.0, 0.65]
        calibration = {
            "camera_to_sim_local": camera_to_world,
            "sim_local_to_camera": np.linalg.inv(camera_to_world).astype(np.float32),
        }
        state = {
            "ee_pos": np.array([0.5, 0.0, 0.2], dtype=np.float32),
            "ee_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "ee_pos_sim": np.array([0.2, 0.0, 0.615], dtype=np.float32),
            "ee_quat_sim": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

        converted = camera_calibration_to_robot_base(calibration, state)

        np.testing.assert_allclose(
            converted["camera_to_sim_local"][:3, 3],
            [1.2, 0.0, 0.235],
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            converted["camera_to_sim_local"]
            @ converted["sim_local_to_camera"],
            np.eye(4),
            atol=1.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
