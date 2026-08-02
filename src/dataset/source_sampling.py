"""Source-aware dataset splitting and deterministic weighted sampling."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler, Subset


def _source_label(value) -> Optional[str]:
    if value is None:
        return None
    label = str(value)
    return label if label.strip() else None


def normalize_env_sampling_weights(
    raw_weights,
    available_envs: Iterable[object],
) -> Optional[dict[str, float]]:
    if raw_weights is None:
        return None
    if not hasattr(raw_weights, "items"):
        raise TypeError("data.env_sampling_weights must be a mapping or null.")

    available_labels = [_source_label(value) for value in available_envs]
    if any(label is None for label in available_labels):
        raise ValueError(
            "Source-weighted sampling requires every episode/sample to have a "
            "non-empty pickle `env` field."
        )
    available = set(available_labels)

    weights = {}
    for source, value in raw_weights.items():
        source = str(source)
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"Sampling weight for {source!r} must be finite and non-negative, "
                f"got {value!r}."
            )
        weights[source] = weight

    configured = set(weights)
    missing = available - configured
    extra = configured - available
    if missing or extra:
        details = []
        if missing:
            details.append(f"dataset sources missing from config: {sorted(missing)}")
        if extra:
            details.append(f"configured sources absent from dataset: {sorted(extra)}")
        raise ValueError("Invalid data.env_sampling_weights: " + "; ".join(details))

    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("data.env_sampling_weights must have a positive total weight.")

    return {source: weights[source] / total for source in sorted(weights)}


def dataset_sample_envs(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent_envs = dataset_sample_envs(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return parent_envs[indices]
    if not hasattr(dataset, "sample_envs"):
        raise TypeError(
            f"Dataset {type(dataset).__name__} does not expose per-sample env metadata."
        )
    return np.asarray(dataset.sample_envs, dtype=object)


def largest_remainder_counts(
    weights: Mapping[str, float], total_count: int
) -> dict[str, int]:
    if total_count < 0:
        raise ValueError(f"total_count must be non-negative, got {total_count}.")
    raw_counts = {
        source: float(weight) * total_count for source, weight in weights.items()
    }
    counts = {source: int(math.floor(value)) for source, value in raw_counts.items()}
    remaining = total_count - sum(counts.values())
    order = sorted(
        weights,
        key=lambda source: (-(raw_counts[source] - counts[source]), source),
    )
    for source in order[:remaining]:
        counts[source] += 1
    return counts


def _bounded_split_counts(
    group_sizes: Mapping[str, int],
    test_split: float,
    positive_sources: set[str],
) -> dict[str, int]:
    if not 0.0 <= test_split <= 1.0:
        raise ValueError(f"test_split must be in [0, 1], got {test_split}.")

    total_count = sum(group_sizes.values())
    target_test_count = total_count - int(total_count * (1 - test_split))
    lower = {}
    upper = {}
    raw = {}
    counts = {}

    for source, size in group_sizes.items():
        needs_both_splits = 0.0 < test_split < 1.0 and source in positive_sources
        if needs_both_splits and size < 2:
            raise ValueError(
                f"Source {source!r} has only {size} item(s), so it cannot appear in "
                "both weighted training and validation splits."
            )
        lower[source] = 1 if needs_both_splits else 0
        upper[source] = size - 1 if needs_both_splits else size
        raw[source] = size * test_split
        counts[source] = min(
            max(int(math.floor(raw[source])), lower[source]), upper[source]
        )

    min_total = sum(lower.values())
    max_total = sum(upper.values())
    if not min_total <= target_test_count <= max_total:
        raise ValueError(
            "The requested test_split cannot keep every positive-weight source in "
            "both splits while preserving the existing split size: "
            f"target={target_test_count}, feasible=[{min_total}, {max_total}]."
        )

    while sum(counts.values()) < target_test_count:
        candidates = [source for source in counts if counts[source] < upper[source]]
        source = min(
            candidates,
            key=lambda item: (-(raw[item] - counts[item]), item),
        )
        counts[source] += 1

    while sum(counts.values()) > target_test_count:
        candidates = [source for source in counts if counts[source] > lower[source]]
        source = min(
            candidates,
            key=lambda item: (raw[item] - counts[item], item),
        )
        counts[source] -= 1

    return counts


def stratified_split_indices(
    envs: Sequence[object],
    test_split: float,
    seed: int,
    positive_sources: Iterable[str],
) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(envs):
        source = _source_label(value)
        if source is None:
            raise ValueError(
                "Source-weighted splitting encountered a sample without an `env` label."
            )
        groups[source].append(idx)

    test_counts = _bounded_split_counts(
        {source: len(indices) for source, indices in groups.items()},
        test_split,
        set(positive_sources),
    )
    rng = np.random.default_rng(int(seed))
    train_indices = []
    test_indices = []
    for source in sorted(groups):
        shuffled = rng.permutation(groups[source]).tolist()
        source_test_count = test_counts[source]
        test_indices.extend(shuffled[:source_test_count])
        train_indices.extend(shuffled[source_test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def stratified_random_split(
    dataset,
    test_split: float,
    seed: int,
    weights: Mapping[str, float],
) -> tuple[Subset, Subset]:
    train_indices, test_indices = stratified_split_indices(
        dataset_sample_envs(dataset),
        test_split,
        seed,
        positive_sources={source for source, weight in weights.items() if weight > 0},
    )
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def stratified_split_items(
    items: Sequence,
    test_split: float,
    seed: int,
    weights: Mapping[str, float],
):
    envs = [getattr(item, "source", None) for item in items]
    train_indices, test_indices = stratified_split_indices(
        envs,
        test_split,
        seed,
        positive_sources={source for source, weight in weights.items() if weight > 0},
    )
    return [items[idx] for idx in train_indices], [items[idx] for idx in test_indices]


def balance_items_by_source_and_size(
    items: Sequence,
    world_size: int,
) -> list[list]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")

    shards = [[] for _ in range(world_size)]
    total_loads = [0 for _ in range(world_size)]
    grouped = defaultdict(list)
    for item in items:
        source = _source_label(getattr(item, "source", None))
        if source is None:
            raise ValueError("Cannot source-balance an episode without an `env` label.")
        grouped[source].append(item)

    for source in sorted(grouped):
        source_loads = [0 for _ in range(world_size)]
        ordered = sorted(
            grouped[source],
            key=lambda item: (-int(item.frame_count), item.path_idx, item.episode_idx),
        )
        for item in ordered:
            rank = min(
                range(world_size),
                key=lambda idx: (source_loads[idx], total_loads[idx], idx),
            )
            shards[rank].append(item)
            item_size = int(item.frame_count)
            source_loads[rank] += item_size
            total_loads[rank] += item_size
    return shards


def allocate_rank_source_quotas(
    weights: Mapping[str, float],
    samples_per_rank: int,
    available_sources_by_rank: Sequence[Iterable[str]],
) -> list[dict[str, int]]:
    """Allocate exact global source quotas subject to per-rank availability."""

    world_size = len(available_sources_by_rank)
    total_samples = samples_per_rank * world_size
    global_quotas = largest_remainder_counts(weights, total_samples)
    sources = sorted(weights)
    available = [set(values) for values in available_sources_by_rank]

    source_node = ("root", 0)
    sink_node = ("sink", 0)
    residual: dict[object, dict[object, int]] = defaultdict(dict)
    adjacency: dict[object, list[object]] = defaultdict(list)

    def add_edge(start, end, capacity):
        residual[start][end] = int(capacity)
        residual[end][start] = 0
        adjacency[start].append(end)
        adjacency[end].append(start)

    source_rank_edges = []
    for source in sources:
        env_node = ("env", source)
        add_edge(source_node, env_node, global_quotas[source])
        for rank in range(world_size):
            if source in available[rank]:
                rank_node = ("rank", rank)
                add_edge(env_node, rank_node, total_samples)
                source_rank_edges.append((source, rank, env_node, rank_node))
    for rank in range(world_size):
        add_edge(("rank", rank), sink_node, samples_per_rank)

    max_flow = 0
    while True:
        parent = {source_node: None}
        queue = deque([source_node])
        while queue and sink_node not in parent:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in parent and residual[node][neighbor] > 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if sink_node not in parent:
            break

        path_capacity = total_samples
        node = sink_node
        while parent[node] is not None:
            previous = parent[node]
            path_capacity = min(path_capacity, residual[previous][node])
            node = previous
        node = sink_node
        while parent[node] is not None:
            previous = parent[node]
            residual[previous][node] -= path_capacity
            residual[node][previous] += path_capacity
            node = previous
        max_flow += path_capacity

    if max_flow != total_samples:
        raise ValueError(
            "Cannot allocate the requested global env sampling quotas across DDP "
            "shards. Ensure every rank shard contains a feasible combination of "
            f"sources. quotas={global_quotas}, availability="
            f"{[sorted(values) for values in available]}"
        )

    quotas = [{source: 0 for source in sources} for _ in range(world_size)]
    for source, rank, env_node, rank_node in source_rank_edges:
        quotas[rank][source] = residual[rank_node][env_node]
    return quotas


class SourceWeightedSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        weights: Mapping[str, float],
        samples_per_rank: int,
        *,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        global_schedule: bool = True,
        source_quotas: Optional[Mapping[str, int]] = None,
    ):
        if samples_per_rank < 0:
            raise ValueError("samples_per_rank must be non-negative.")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} is outside [0, {num_replicas}).")

        self.dataset = dataset
        self.weights = dict(weights)
        self.samples_per_rank = int(samples_per_rank)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.global_schedule = bool(global_schedule)
        self.source_quotas = dict(source_quotas) if source_quotas is not None else None
        self.epoch = 0

        envs = dataset_sample_envs(dataset)
        self.indices_by_source: dict[str, list[int]] = defaultdict(list)
        for idx, value in enumerate(envs):
            source = _source_label(value)
            if source is None:
                raise ValueError(
                    "Source-weighted sampling encountered a sample without an `env` label."
                )
            if source not in self.weights:
                raise ValueError(
                    f"Dataset source {source!r} is missing from env sampling weights."
                )
            self.indices_by_source[source].append(idx)

        if self.source_quotas is None:
            missing_positive = [
                source
                for source, weight in self.weights.items()
                if weight > 0 and not self.indices_by_source.get(source)
            ]
            if missing_positive:
                raise ValueError(
                    "Weighted split has no samples for positive-weight sources: "
                    f"{missing_positive}."
                )
        if self.source_quotas is not None:
            if set(self.source_quotas) != set(self.weights):
                raise ValueError("source_quotas keys must exactly match weight keys.")
            if sum(self.source_quotas.values()) != self.samples_per_rank:
                raise ValueError(
                    "source_quotas must sum to samples_per_rank, got "
                    f"{sum(self.source_quotas.values())} vs {self.samples_per_rank}."
                )
            unavailable = [
                source
                for source, count in self.source_quotas.items()
                if count > 0 and not self.indices_by_source.get(source)
            ]
            if unavailable:
                raise ValueError(
                    f"Local DDP shard cannot satisfy quotas for {unavailable}."
                )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    @staticmethod
    def _draw_indices(pool, count, generator):
        if count == 0:
            return []
        if not pool:
            raise ValueError("Cannot draw a positive quota from an empty source pool.")

        selected = []
        while len(selected) < count:
            order = torch.randperm(len(pool), generator=generator).tolist()
            take = min(count - len(selected), len(pool))
            selected.extend(pool[idx] for idx in order[:take])
        return selected

    def _build_schedule(self, quotas, generator):
        schedule = []
        for source in sorted(quotas):
            schedule.extend(
                self._draw_indices(
                    self.indices_by_source.get(source, []),
                    int(quotas[source]),
                    generator,
                )
            )
        if schedule:
            order = torch.randperm(len(schedule), generator=generator).tolist()
            schedule = [schedule[idx] for idx in order]
        return schedule

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.source_quotas is not None:
            schedule = self._build_schedule(self.source_quotas, generator)
        elif self.global_schedule:
            global_count = self.samples_per_rank * self.num_replicas
            global_quotas = largest_remainder_counts(self.weights, global_count)
            global_schedule = self._build_schedule(global_quotas, generator)
            schedule = global_schedule[self.rank :: self.num_replicas]
        else:
            local_quotas = largest_remainder_counts(
                self.weights, self.samples_per_rank
            )
            schedule = self._build_schedule(local_quotas, generator)

        if len(schedule) != self.samples_per_rank:
            raise RuntimeError(
                f"Weighted sampler generated {len(schedule)} samples, expected "
                f"{self.samples_per_rank}."
            )
        return iter(schedule)

    def __len__(self):
        return self.samples_per_rank
