import unittest
from collections import Counter

import numpy as np
import torch

from src.dataset.base import EpisodeRef
from src.dataset.dataloader import build_dataloader
from src.dataset.source_sampling import (
    SourceWeightedSampler,
    allocate_rank_source_quotas,
    balance_items_by_source_and_size,
    dataset_sample_envs,
    largest_remainder_counts,
    normalize_env_sampling_weights,
    stratified_split_items,
    validate_env_sampling_mode,
)


class DummySourceDataset(torch.utils.data.Dataset):
    def __init__(self, envs):
        self.sample_envs = np.asarray(envs, dtype=object)

    def __len__(self):
        return len(self.sample_envs)

    def __getitem__(self, idx):
        return idx


class SourceWeightValidationTest(unittest.TestCase):
    def test_requires_multi_rank_ddp_sharding(self):
        validate_env_sampling_mode(None, ddp_shard_enabled=False)
        validate_env_sampling_mode({"FurnitureBench": 1.0}, ddp_shard_enabled=True)
        with self.assertRaisesRegex(ValueError, "only supported by multi-rank"):
            validate_env_sampling_mode(
                {"FurnitureBench": 1.0}, ddp_shard_enabled=False
            )

    def test_normalizes_and_requires_an_exact_source_mapping(self):
        self.assertIsNone(normalize_env_sampling_weights(None, [None]))
        weights = normalize_env_sampling_weights(
            {"FurnitureBench": 50, "AutoMate": 35, "ManiSkill": 15},
            ["FurnitureBench", "AutoMate", "ManiSkill"],
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(weights["FurnitureBench"], 0.5)

        with self.assertRaisesRegex(ValueError, "missing from config"):
            normalize_env_sampling_weights(
                {"FurnitureBench": 1.0},
                ["FurnitureBench", "AutoMate"],
            )
        with self.assertRaisesRegex(ValueError, "non-empty pickle `env`"):
            normalize_env_sampling_weights({"FurnitureBench": 1.0}, [None])

        with self.assertRaisesRegex(ValueError, "Invalid data.env_sampling_weights"):
            normalize_env_sampling_weights({"FurnitureBench": 1.0}, [" FurnitureBench"])

        zero_weight = normalize_env_sampling_weights(
            {"FurnitureBench": 1.0, "AutoMate": 0.0},
            ["FurnitureBench", "AutoMate"],
        )
        self.assertEqual(zero_weight, {"AutoMate": 0.0, "FurnitureBench": 1.0})
        for invalid_weight in (-1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                normalize_env_sampling_weights(
                    {"FurnitureBench": invalid_weight}, ["FurnitureBench"]
                )

class SourceWeightedSamplerTest(unittest.TestCase):
    def setUp(self):
        self.weights = {"FB": 0.50, "AutoMate": 0.35, "ManiSkill": 0.15}
        self.dataset = DummySourceDataset(
            ["FB"] * 10 + ["AutoMate"] * 7 + ["ManiSkill"] * 3
        )

    def _counts(self, indices, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        envs = dataset_sample_envs(dataset)
        return Counter(envs[index] for index in indices)

    def test_exact_epoch_quota_and_deterministic_epoch_shuffle(self):
        sampler = SourceWeightedSampler(
            self.dataset,
            self.weights,
            samples_per_rank=100,
            seed=123,
        )
        epoch_zero_a = list(sampler)
        epoch_zero_b = list(sampler)
        self.assertEqual(epoch_zero_a, epoch_zero_b)
        self.assertEqual(
            self._counts(epoch_zero_a),
            Counter({"FB": 50, "AutoMate": 35, "ManiSkill": 15}),
        )

        sampler.set_epoch(1)
        epoch_one = list(sampler)
        self.assertNotEqual(epoch_zero_a, epoch_one)
        self.assertEqual(self._counts(epoch_one), self._counts(epoch_zero_a))

    def test_fixed_step_dataloader_preserves_exact_single_gpu_quota(self):
        sampler = SourceWeightedSampler(
            self.dataset,
            self.weights,
            samples_per_rank=100,
            seed=321,
        )
        loader = build_dataloader(
            dataset=self.dataset,
            batch_size=10,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            sampler=sampler,
            steps_per_epoch=10,
        )
        indices = torch.cat(list(loader)).tolist()
        self.assertEqual(len(indices), 100)
        self.assertEqual(
            self._counts(indices),
            Counter({"FB": 50, "AutoMate": 35, "ManiSkill": 15}),
        )

    def test_rank_quota_allocator_respects_source_availability(self):
        quotas = allocate_rank_source_quotas(
            self.weights,
            samples_per_rank=20,
            available_sources_by_rank=[
                {"FB", "AutoMate"},
                {"FB", "ManiSkill"},
            ],
        )
        self.assertEqual([sum(rank.values()) for rank in quotas], [20, 20])
        combined = Counter()
        for rank_quota in quotas:
            combined.update(rank_quota)
        self.assertEqual(combined, Counter(largest_remainder_counts(self.weights, 40)))
        self.assertEqual(quotas[0]["ManiSkill"], 0)
        self.assertEqual(quotas[1]["AutoMate"], 0)

        rank_datasets = [
            DummySourceDataset(["FB"] * 4 + ["AutoMate"] * 4),
            DummySourceDataset(["FB"] * 4 + ["ManiSkill"] * 4),
        ]
        for rank in range(2):
            sampler = SourceWeightedSampler(
                rank_datasets[rank],
                self.weights,
                samples_per_rank=20,
                seed=9,
                source_quotas=quotas[rank],
            )
            self.assertEqual(
                self._counts(list(sampler), dataset=rank_datasets[rank]),
                Counter({key: value for key, value in quotas[rank].items() if value}),
            )

        with self.assertRaisesRegex(ValueError, "Cannot allocate"):
            allocate_rank_source_quotas(
                {"FB": 1.0, "AutoMate": 0.0},
                samples_per_rank=10,
                available_sources_by_rank=[{"FB"}, {"AutoMate"}],
            )

    def test_episode_frame_balancing_spreads_each_source(self):
        items = [
            EpisodeRef(0, 0, 0, 10, 10, "task", 1, "sim", "FB"),
            EpisodeRef(0, 1, 10, 18, 8, "task", 1, "sim", "FB"),
            EpisodeRef(0, 2, 18, 24, 6, "task", 1, "sim", "AutoMate"),
            EpisodeRef(0, 3, 24, 28, 4, "task", 1, "sim", "AutoMate"),
        ]
        shards = balance_items_by_source_and_size(items, world_size=2)
        self.assertEqual(
            [{item.source for item in shard} for shard in shards],
            [{"FB", "AutoMate"}, {"FB", "AutoMate"}],
        )
        self.assertEqual(
            [sum(item.frame_count for item in shard) for shard in shards],
            [14, 14],
        )


class StratifiedSourceSplitTest(unittest.TestCase):
    def test_split_preserves_size_and_positive_sources(self):
        items = [
            EpisodeRef(0, idx, idx, idx + 1, 1, "task", 1, "sim", source)
            for idx, source in enumerate(
                ["FB"] * 20 + ["AutoMate"] * 20 + ["ManiSkill"] * 20
            )
        ]
        weights = {"FB": 0.5, "AutoMate": 0.35, "ManiSkill": 0.15}
        train, validation = stratified_split_items(
            items, test_split=0.2, seed=42, weights=weights
        )
        self.assertEqual(len(train), 48)
        self.assertEqual(len(validation), 12)
        self.assertEqual({item.source for item in train}, set(weights))
        self.assertEqual({item.source for item in validation}, set(weights))

        train_again, validation_again = stratified_split_items(
            items, test_split=0.2, seed=42, weights=weights
        )
        self.assertEqual(train, train_again)
        self.assertEqual(validation, validation_again)

    def test_too_small_split_fails_instead_of_dropping_a_source(self):
        items = [
            EpisodeRef(0, idx, idx, idx + 1, 1, "task", 1, "sim", source)
            for idx, source in enumerate(
                ["FB"] * 10 + ["AutoMate"] * 10 + ["ManiSkill"] * 10
            )
        ]
        weights = {"FB": 0.5, "AutoMate": 0.35, "ManiSkill": 0.15}
        with self.assertRaisesRegex(ValueError, "cannot keep every positive-weight source"):
            stratified_split_items(items, test_split=0.01, seed=0, weights=weights)


if __name__ == "__main__":
    unittest.main()
