import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset.depth_stats import (
    DEPTH_CAMERA_KEYS,
    DEPTH_NORMALIZER_STATS_ATTR,
    deserialize_depth_moments,
    empty_depth_moments,
    finalize_depth_moments,
    merge_depth_moments,
    update_depth_moments,
    validate_usable_depth_stats,
)
from src.dataset.lmdb import (
    EPISODE_INDEX_KEY,
    LMDB_FORMAT_VERSION,
    META_KEY,
    build_episode_manifest,
    build_frame_specs,
    combine_lmdb_episode_subset,
    compute_global_depth_stats,
    episode_data_key,
    frame_key,
    json_dumps_bytes,
    open_lmdb_env,
    pack_frame,
    pack_named_arrays,
)
from scripts.validate_lmdb_dataset import validate_path


def _write_depth_lmdb(path: Path, episodes, include_stats=True):
    env = open_lmdb_env(path, readonly=False)
    moments = empty_depth_moments()
    episode_index = []
    frame_specs = None
    frame_cursor = 0
    try:
        with env.begin(write=True) as txn:
            for episode_idx, episode in enumerate(episodes):
                wrist = np.asarray(episode["wrist"], dtype=np.float32)
                front = np.asarray(episode["front"], dtype=np.float32)
                if frame_specs is None:
                    frame_specs = build_frame_specs(
                        {
                            "depth_image1": wrist[0],
                            "depth_image2": front[0],
                        }
                    )
                txn.put(episode_data_key(episode_idx), pack_named_arrays({}))
                for local_idx in range(len(wrist)):
                    txn.put(
                        frame_key(frame_cursor + local_idx),
                        pack_frame(
                            {
                                "depth_image1": wrist[local_idx],
                                "depth_image2": front[local_idx],
                            },
                            frame_specs,
                        ),
                    )
                episode_index.append(
                    {
                        "episode_idx": episode_idx,
                        "frame_start": frame_cursor,
                        "frame_end": frame_cursor + len(wrist),
                        "task": "test_task",
                        "success": 1,
                        "env": episode["env"],
                    }
                )
                update_depth_moments(moments, "wrist", wrist)
                update_depth_moments(moments, "front", front)
                frame_cursor += len(wrist)

            attrs = {
                "domain": "sim",
                "n_episodes": len(episodes),
                "n_timesteps": frame_cursor,
            }
            if include_stats:
                attrs[DEPTH_NORMALIZER_STATS_ATTR] = finalize_depth_moments(moments)
            meta = {
                "format": "robust_rearrangement_lmdb",
                "format_version": LMDB_FORMAT_VERSION,
                "attrs": attrs,
                "frame_specs": frame_specs,
                "lowdim_specs": {},
            }
            txn.put(META_KEY, json_dumps_bytes(meta))
            txn.put(EPISODE_INDEX_KEY, json_dumps_bytes(episode_index))
        env.sync()
    finally:
        env.close()


class DepthMomentTest(unittest.TestCase):
    def test_ignores_zero_and_nonfinite_without_changing_sign(self):
        moments = empty_depth_moments()
        update_depth_moments(
            moments,
            "wrist",
            np.asarray([0.0, -1.0, -2.0, np.inf, -np.inf, np.nan]),
        )
        stats = finalize_depth_moments(moments)["wrist"]
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["mean"], -1.5)
        self.assertAlmostEqual(stats["std"], 0.5)
        self.assertAlmostEqual(stats["M2"], 0.5)

    def test_merge_matches_direct_population_statistics(self):
        left = empty_depth_moments()
        right = empty_depth_moments()
        update_depth_moments(left, "front", np.asarray([1.0, 2.0]))
        update_depth_moments(right, "front", np.asarray([4.0, 8.0, 0.0]))
        merge_depth_moments(left, right)
        stats = finalize_depth_moments(left)["front"]
        expected = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        self.assertEqual(stats["count"], len(expected))
        self.assertAlmostEqual(stats["mean"], float(expected.mean()))
        self.assertAlmostEqual(stats["std"], float(expected.std()))

    def test_rejects_inconsistent_serialized_moments(self):
        stats = finalize_depth_moments(empty_depth_moments())
        stats["wrist"] = {"count": 2, "mean": 1.0, "std": 10.0, "M2": 0.5}
        with self.assertRaisesRegex(ValueError, "Inconsistent depth statistics"):
            deserialize_depth_moments(stats)

    def test_rgbd_use_rejects_empty_or_zero_variance_stats(self):
        empty_stats = finalize_depth_moments(empty_depth_moments())
        with self.assertRaisesRegex(ValueError, "no finite non-zero pixels"):
            validate_usable_depth_stats(empty_stats)

        constant_moments = empty_depth_moments()
        update_depth_moments(constant_moments, "wrist", np.ones(4))
        update_depth_moments(constant_moments, "front", np.ones(4))
        with self.assertRaisesRegex(ValueError, "std must be finite and positive"):
            validate_usable_depth_stats(finalize_depth_moments(constant_moments))


class LMDBDepthStatsTest(unittest.TestCase):
    def test_full_shards_merge_metadata_and_subset_scans_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "a.lmdb"
            path_b = root / "b.lmdb"
            episodes_a = [
                {
                    "env": "FurnitureBench",
                    "wrist": [[[-1.0, -2.0], [0.0, np.nan]]],
                    "front": [[[1.0, 2.0], [0.0, np.inf]]],
                },
                {
                    "env": "AutoMate",
                    "wrist": [[[-3.0, -4.0], [0.0, 0.0]]],
                    "front": [[[3.0, 4.0], [0.0, 0.0]]],
                },
            ]
            episodes_b = [
                {
                    "env": "ManiSkill",
                    "wrist": [[[-5.0, -6.0], [0.0, 0.0]]],
                    "front": [[[5.0, 6.0], [0.0, 0.0]]],
                }
            ]
            _write_depth_lmdb(path_a, episodes_a)
            _write_depth_lmdb(path_b, episodes_b)
            validate_path(path_a, sample_episodes=1, full_stats=True, atol=1e-6)

            manifest = build_episode_manifest([path_a, path_b])
            self.assertEqual(
                [ref.source for ref in manifest],
                ["FurnitureBench", "AutoMate", "ManiSkill"],
            )
            combined_data, _ = combine_lmdb_episode_subset(
                [path_a, path_b], manifest, keys=[]
            )
            self.assertEqual(
                combined_data["env"],
                ["FurnitureBench", "AutoMate", "ManiSkill"],
            )
            merged = compute_global_depth_stats([path_a, path_b], manifest)
            np.testing.assert_allclose(
                [merged["wrist"]["mean"], merged["wrist"]["std"]],
                [np.mean([-1, -2, -3, -4, -5, -6]), np.std([-1, -2, -3, -4, -5, -6])],
            )
            np.testing.assert_allclose(
                [merged["front"]["mean"], merged["front"]["std"]],
                [np.mean([1, 2, 3, 4, 5, 6]), np.std([1, 2, 3, 4, 5, 6])],
            )

            first_episode_only = compute_global_depth_stats(
                [path_a, path_b], [manifest[0]]
            )
            self.assertEqual(first_episode_only["wrist"]["count"], 2)
            self.assertAlmostEqual(first_episode_only["wrist"]["mean"], -1.5)
            self.assertAlmostEqual(first_episode_only["front"]["mean"], 1.5)

    def test_old_lmdb_without_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old.lmdb"
            _write_depth_lmdb(
                path,
                [
                    {
                        "env": "FurnitureBench",
                        "wrist": [[[1.0]]],
                        "front": [[[2.0]]],
                    }
                ],
                include_stats=False,
            )
            with self.assertRaisesRegex(ValueError, "does not contain"):
                compute_global_depth_stats(path)


if __name__ == "__main__":
    unittest.main()
