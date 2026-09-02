import numpy as np
import pytest
import torch

from src.behavior.base import Actor
from src.dataset.dataset import (
    action_valid_mask,
    create_sample_indices,
    observation_filtered_normalizer_data,
)
from src.real.legacy_timeline import build_uniform_action_grid


def test_uniform_action_grid_inserts_noops_without_dropping_source_actions():
    timeline, source_slots, residual = build_uniform_action_grid(
        [1_000_000_000, 1_100_000_000, 1_400_000_000]
    )

    np.testing.assert_array_equal(source_slots, [0, 1, 4])
    np.testing.assert_array_equal(
        timeline,
        [
            1_000_000_000,
            1_100_000_000,
            1_200_000_000,
            1_300_000_000,
            1_400_000_000,
        ],
    )
    np.testing.assert_array_equal(residual, [0.0, 0.0, 0.0])


def test_uniform_action_grid_fails_closed_on_large_quantization_error():
    with pytest.raises(ValueError, match="residual exceeds"):
        build_uniform_action_grid(
            [0, 51_000_000, 100_000_000],
            max_quantization_residual_ms=75.0,
        )


def test_sample_indices_only_use_valid_observations_and_keep_episode_tail():
    valid = np.asarray([True, False, True, True, True])
    indices = create_sample_indices(
        episode_ends=np.asarray([5]),
        sequence_length=4,
        pad_after=3,
        observation_valid=valid,
        obs_horizon=1,
    )

    assert len(indices) == 4
    # The last valid observation remains a sample even though only its first
    # action is inside the episode.
    np.testing.assert_array_equal(indices[-1], [4, 5, 0, 1, 0])
    np.testing.assert_array_equal(action_valid_mask(4, 0, 0, 1), [1, 0, 0, 0])


def test_masked_action_loss_ignores_terminal_padding():
    loss = torch.tensor(
        [[[1.0, 1.0], [4.0, 4.0], [1000.0, 1000.0]]]
    )
    reduced = Actor.masked_action_loss_per_sample(
        loss, {"action_valid_mask": torch.tensor([[1.0, 1.0, 0.0]])}
    )
    torch.testing.assert_close(reduced, torch.tensor([[2.5]]))


def test_normalizer_keeps_all_actions_but_filters_placeholder_observations():
    data = {
        "robot_state": torch.tensor([[1.0], [999.0], [3.0]]),
        "skill": torch.tensor([[1.0], [999.0], [3.0]]),
        "action": torch.tensor([[10.0], [20.0], [30.0]]),
    }
    filtered = observation_filtered_normalizer_data(
        data, np.asarray([True, False, True])
    )

    torch.testing.assert_close(filtered["robot_state"], torch.tensor([[1.0], [3.0]]))
    torch.testing.assert_close(filtered["skill"], torch.tensor([[1.0], [3.0]]))
    torch.testing.assert_close(filtered["action"], data["action"])
