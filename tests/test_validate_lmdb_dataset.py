import unittest

import numpy as np

from scripts.validate_lmdb_dataset import check_array_specs, compare_full_stats


class ValidateLmdbDatasetTest(unittest.TestCase):
    def setUp(self):
        self.specs = {
            "robot_state": {"dtype": "float32", "shape": [3, 16]},
            "parts_poses": {"dtype": "float32", "shape": [3, 28]},
        }
        self.stats = {}
        self.incompatible = set()
        self.observed_shapes = {}

    def check(self, robot_shape=(3, 16), parts_shape=(3, 28)):
        check_array_specs(
            "test.lmdb",
            0,
            3,
            {
                "robot_state": np.zeros(robot_shape, dtype=np.float32),
                "parts_poses": np.zeros(parts_shape, dtype=np.float32),
            },
            self.specs,
            self.stats,
            self.incompatible,
            self.observed_shapes,
        )

    def test_variable_part_count_marks_stats_incompatible(self):
        self.check(parts_shape=(3, 28))
        self.check(parts_shape=(3, 42))

        self.assertEqual(self.incompatible, {"parts_poses"})
        self.assertNotIn("parts_poses", self.stats)
        self.assertEqual(self.observed_shapes["parts_poses"], {(28,), (42,)})
        compare_full_stats(
            "test.lmdb", self.stats, {}, self.incompatible, atol=1e-6
        )

    def test_non_part_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "robot_state.*trailing shape"):
            self.check(robot_shape=(3, 15))

    def test_part_pose_width_must_be_multiple_of_seven(self):
        with self.assertRaisesRegex(ValueError, "flat sequence of 7D part poses"):
            self.check(parts_shape=(3, 29))


if __name__ == "__main__":
    unittest.main()
