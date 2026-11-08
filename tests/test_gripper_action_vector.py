import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from src.behavior.base import Actor, model_requires_skill_input, model_skill_dim, validate_annotation_config
from src.common.skills import SKILL_ORDER, skill_onehot_to_gripper_action_vector
from src.dataset.dataset import RGBDDataset
from src.dataset.lmdb import validate_gp_skill_gripper_vector_source


class GripperActionVectorTest(unittest.TestCase):
    def test_all_skill_labels_and_batch_shape(self):
        onehot = torch.eye(len(SKILL_ORDER)).repeat(2, 1, 1)
        result = skill_onehot_to_gripper_action_vector(onehot)
        self.assertEqual(result.shape, (2, 5, 1))
        torch.testing.assert_close(result[0, :, 0], torch.tensor([1., -1., -1., 1., -1.]))

    def test_rejects_missing_ambiguous_and_nonbinary_skills(self):
        for skill in (torch.zeros(5), torch.ones(5), torch.tensor([.5, .5, 0., 0., 0.]), torch.zeros(4)):
            with self.assertRaises(ValueError):
                skill_onehot_to_gripper_action_vector(skill)

    def test_config_requires_gp_rgbd_dit_lmdb(self):
        cfg = OmegaConf.create({
            "observation_type": "rgbd",
            "actor": {"name": "diffusion", "diffusion_model": {"name": "dit"}},
            "data": {"annotate_guidance_point": True, "annotate_skill_one_hot": False,
                     "gripper_action_vector": True, "storage_format": "lmdb"},
        })
        validate_annotation_config(cfg)
        self.assertTrue(model_requires_skill_input(cfg))
        self.assertEqual(model_skill_dim(cfg), 1)
        for path, value in (("observation_type", "image"), ("actor.diffusion_model.name", "unet"),
                            ("data.annotate_guidance_point", False), ("data.annotate_skill_one_hot", True),
                            ("data.storage_format", "zarr")):
            invalid = OmegaConf.create(OmegaConf.to_container(cfg))
            OmegaConf.update(invalid, path, value)
            with self.assertRaises(ValueError, msg=path):
                validate_annotation_config(invalid)

    def test_gp_skill_metadata_gate(self):
        attrs = {"provenance": {"condition": "rgbd-gp-skill"},
                 "annotation_source": "scripted", "image_annotation_mode": "guidance-point"}
        with patch("src.dataset.lmdb.read_lmdb_attrs", return_value=attrs):
            validate_gp_skill_gripper_vector_source(Path("gp.lmdb"))
        for key, value in (("annotation_source", "vlm"), ("image_annotation_mode", "none"),
                           ("provenance", {"condition": "rgbd-skill"})):
            invalid = dict(attrs)
            invalid[key] = value
            with patch("src.dataset.lmdb.read_lmdb_attrs", return_value=invalid):
                with self.assertRaises(ValueError):
                    validate_gp_skill_gripper_vector_source(Path("wrong.lmdb"))

    def test_single_and_ddp_trainers_resolve_one_dimensional_condition(self):
        from src.train.bc import configure_observation_annotation_flags as single_configure
        from src.train.bc_ddp import (
            configure_observation_annotation_flags as ddp_configure,
            get_normalizer_stats_key_map,
        )

        base = {
            "observation_type": "rgbd", "control": {"control_mode": "pos"},
            "actor": {"name": "diffusion", "diffusion_model": {"name": "dit"}},
            "data": {"annotate_guidance_point": True, "annotate_skill_one_hot": False,
                     "gripper_action_vector": True, "storage_format": "lmdb"},
        }
        for configure in (single_configure, ddp_configure):
            cfg = OmegaConf.create(base)
            configure(cfg)
            self.assertEqual(cfg.skill_dim, 1)
            self.assertFalse(cfg.data.annotate_skill_one_hot)
            self.assertEqual(get_normalizer_stats_key_map(cfg), {
                "robot_state": "robot_state", "action": "action/pos"
            })

    def test_dataset_keeps_raw_scalar_out_of_normalizer_and_legacy_onehot(self):
        skill = np.eye(5, dtype=np.float32)
        combined = {
            "episode_ends": np.array([5]), "obs_valid": np.ones(5, dtype=np.bool_),
            "robot_state": np.arange(80, dtype=np.float32).reshape(5, 16),
            "action/pos": np.arange(50, dtype=np.float32).reshape(5, 10),
            "action/delta": np.arange(50, dtype=np.float32).reshape(5, 10),
            "skill": skill, "zarr_idx": np.zeros(5, dtype=np.int64),
            "within_zarr_idx": np.arange(5, dtype=np.int64),
        }
        class ImageStore:
            def get_frames(self, frame_indices, keys):
                count = len(frame_indices)
                return {"color_image1": np.zeros((count, 2, 2, 3), dtype=np.uint8),
                        "color_image2": np.zeros((count, 2, 2, 3), dtype=np.uint8),
                        "depth_image1": np.ones((count, 2, 2), dtype=np.float32),
                        "depth_image2": np.ones((count, 2, 2), dtype=np.float32)}
        with patch("src.dataset.dataset.load_combined_data", return_value=(combined, {})), \
             patch("src.dataset.dataset.build_lazy_image_stores", return_value=[ImageStore()]), \
             patch("src.dataset.dataset.validate_gp_skill_gripper_vector_source"):
            vector = RGBDDataset(Path("gp.lmdb"), 2, 1, 1, control_mode="pos",
                                  load_into_memory=False, gripper_action_vector=True)
            legacy = RGBDDataset(Path("gp.lmdb"), 2, 1, 1, control_mode="pos",
                                  load_into_memory=False, gripper_action_vector=False)
            metadata_only = RGBDDataset(
                Path("gp.lmdb"), 2, 1, 1, control_mode="pos",
                load_into_memory=False, include_skill_input=False,
                load_skill_metadata=True,
            )
        torch.testing.assert_close(
            vector.train_data["gripper_action_vector"][:, 0],
            torch.tensor([1., -1., -1., 1., -1.]),
        )
        self.assertEqual(vector.skill_dim, 1)
        self.assertNotIn("skill", vector.normalizer.keys())
        self.assertNotIn("gripper_action_vector", vector.normalizer.keys())
        self.assertEqual(vector[0]["gripper_action_vector"].shape, (1, 1))
        self.assertNotIn("skill", vector[0])
        self.assertEqual(legacy.train_data["skill"].shape, torch.Size([5, 5]))
        self.assertIn("skill", legacy.normalizer.keys())
        self.assertEqual(metadata_only.skill_dim, 5)
        self.assertNotIn("skill", metadata_only.train_data)

    def test_training_and_eval_condition_match(self):
        actor = SimpleNamespace(skill_dim=1, gripper_action_vector=True)
        state = torch.zeros(1, 1, 16)
        training = Actor._training_skill(
            actor, {"gripper_action_vector": torch.tensor([[[1.]]])}, state
        )
        evaluation = Actor._inference_skill(
            actor,
            deque([{"gripper_action_vector": torch.tensor([[1.]])}]),
            state,
        )
        torch.testing.assert_close(training, evaluation)
        self.assertEqual(evaluation.shape, (1, 1, 1))

    def test_legacy_actor_without_new_flag_keeps_onehot_input(self):
        actor = SimpleNamespace(skill_dim=5)
        state = torch.zeros(1, 1, 16)
        onehot = torch.tensor([[[1., 0., 0., 0., 0.]]])
        training = Actor._training_skill(actor, {"skill": onehot}, state)
        evaluation = Actor._inference_skill(
            actor, deque([{"skill": onehot[:, 0]}]), state
        )
        torch.testing.assert_close(training, evaluation)

    def test_rollout_attaches_five_way_skill_before_scalar_conversion(self):
        from src.eval.rollout import _attach_skill_tensor_to_obs

        actor = SimpleNamespace(
            skill_dim=1,
            gripper_action_vector=True,
            requires_skill_input=True,
            device="cpu",
        )
        obs = {}
        _attach_skill_tensor_to_obs(obs, actor, ["place"])
        self.assertEqual(obs["gripper_action_vector"].shape, (1, 1))
        self.assertNotIn("skill", obs)
        signal = Actor._inference_skill(
            actor, deque([obs]), torch.zeros(1, 1, 16)
        )
        torch.testing.assert_close(signal, torch.tensor([[[-1.]]]))


if __name__ == "__main__":
    unittest.main()
