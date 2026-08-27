import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.data_processing.utils import clip_quat_xyzw_magnitude


class LegacyRotationSegmentScaleTest(unittest.TestCase):
    def test_explicit_source_episode_scale_is_preserved_after_split(self):
        full_rotvec = np.tile([0.0, 0.0, 0.1], (16, 1))
        full_quat = Rotation.from_rotvec(full_rotvec).as_quat()
        expected = clip_quat_xyzw_magnitude(full_quat, clip_mag=0.35)
        source_scale = 0.35 / np.linalg.norm(full_rotvec)

        first = clip_quat_xyzw_magnitude(
            full_quat[:8], clip_mag=0.35, episode_scale_factor=source_scale
        )
        second = clip_quat_xyzw_magnitude(
            full_quat[8:], clip_mag=0.35, episode_scale_factor=source_scale
        )
        np.testing.assert_allclose(np.concatenate([first, second]), expected)


if __name__ == "__main__":
    unittest.main()
