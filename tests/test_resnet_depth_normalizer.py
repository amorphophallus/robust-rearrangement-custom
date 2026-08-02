import unittest

import torch

from src.behavior.base import Actor
from src.dataset.normalizer import LinearNormalizer
from src.models.vision import ResnetEncoder


def _make_encoder(camera="wrist"):
    return ResnetEncoder(
        model_name="resnet18_rgbd",
        freeze=False,
        device="cpu",
        use_groupnorm=False,
        pretrained=False,
        camera_name=camera,
    )


def _make_rgbd_actor():
    # Exercise Actor's real checkpoint compatibility logic without constructing
    # a complete policy network.
    actor = object.__new__(Actor)
    torch.nn.Module.__init__(actor)
    actor.observation_type = "rgbd"
    actor.normalizer = LinearNormalizer()
    actor.encoder1 = _make_encoder("wrist")
    actor.encoder2 = _make_encoder("front")
    return actor


class ResnetDepthNormalizerTest(unittest.TestCase):
    def test_dataset_stats_normalize_depth_and_roundtrip_in_state_dict(self):
        encoder = _make_encoder()
        encoder.set_depth_normalizer_stats(
            {"count": 123, "mean": -2.0, "std": 0.5}
        )
        encoder.model = torch.nn.Identity()
        observation = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
        observation[:, 3] = -1.0
        normalized = encoder(observation)
        torch.testing.assert_close(
            normalized[:, 3], torch.full((1, 2, 2), 2.0)
        )

        state_dict = encoder.state_dict()
        self.assertIn("depth_mean", state_dict)
        self.assertIn("depth_std", state_dict)
        self.assertIn("depth_count", state_dict)

        restored = _make_encoder()
        # Identity removes model parameters, so only load the normalizer buffers
        # into a regular encoder using a matching model state from another source.
        source = _make_encoder()
        source.set_depth_normalizer_stats(
            {"count": 123, "mean": -2.0, "std": 0.5}
        )
        restored.load_state_dict(source.state_dict())
        self.assertEqual(
            restored.get_depth_normalizer_stats(),
            {"count": 123, "mean": -2.0, "std": 0.5, "M2": 30.75},
        )

    def test_checkpoint_without_depth_buffers_is_rejected(self):
        source = _make_encoder()
        source.set_depth_normalizer_stats(
            {"count": 50, "mean": 3.0, "std": 4.0}
        )
        missing_stats_state = source.state_dict()
        for key in ("depth_mean", "depth_std", "depth_count"):
            del missing_stats_state[key]

        target = _make_encoder()
        target.set_depth_normalizer_stats(
            {"count": 50, "mean": 3.0, "std": 4.0}
        )
        with self.assertRaisesRegex(RuntimeError, "missing required dataset-backed"):
            target.load_state_dict(missing_stats_state)

    def test_uninitialized_encoder_and_partial_stats_fail(self):
        source = _make_encoder(camera="front")
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            source(torch.zeros((1, 4, 2, 2), dtype=torch.float32))

        partial_state = source.state_dict()
        del partial_state["depth_count"]
        with self.assertRaisesRegex(RuntimeError, "Partially missing RGBD depth"):
            _make_encoder(camera="front").load_state_dict(partial_state)

    def test_two_camera_actor_checkpoint_is_self_contained(self):
        source = _make_rgbd_actor()
        source.set_depth_normalizer_stats(
            {
                "wrist": {"count": 20, "mean": -1.5, "std": 0.25, "M2": 1.25},
                "front": {"count": 30, "mean": 2.5, "std": 0.5, "M2": 7.5},
            }
        )
        checkpoint = {
            "model_state_dict": source.state_dict(),
            "depth_normalizer_stats": source.get_depth_normalizer_stats(),
        }
        self.assertEqual(
            set(checkpoint["depth_normalizer_stats"]["wrist"]),
            {"count", "mean", "std", "M2"},
        )

        restored = _make_rgbd_actor()
        restored.load_state_dict(checkpoint["model_state_dict"])
        self.assertEqual(
            restored.get_depth_normalizer_stats(),
            checkpoint["depth_normalizer_stats"],
        )

        corrupted = checkpoint["model_state_dict"].copy()
        del corrupted["encoder2.depth_count"]
        with self.assertRaisesRegex(RuntimeError, "Partially missing RGBD depth"):
            _make_rgbd_actor().load_state_dict(corrupted)

        missing_stats_state = checkpoint["model_state_dict"].copy()
        for encoder_name in ("encoder1", "encoder2"):
            for buffer_name in ("depth_mean", "depth_std", "depth_count"):
                del missing_stats_state[f"{encoder_name}.{buffer_name}"]
        with self.assertRaisesRegex(RuntimeError, "missing required dataset-backed"):
            _make_rgbd_actor().load_state_dict(missing_stats_state)


if __name__ == "__main__":
    unittest.main()
