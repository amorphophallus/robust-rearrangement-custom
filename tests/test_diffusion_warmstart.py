import unittest

import torch

from src.behavior.diffusion import DiffusionPolicy


class DiffusionWarmstartTest(unittest.TestCase):
    def test_elapsed_shift_and_queue_actions_compose_without_mutating_previous(self):
        policy = object.__new__(DiffusionPolicy)
        torch.nn.Module.__init__(policy)
        policy.device = torch.device("cpu")
        policy.pred_horizon = 4
        policy.action_horizon = 2
        policy.action_dim = 2
        policy.prev_naction = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        previous = policy.prev_naction.clone()

        seed = policy._warmstart_seed(
            1,
            torch.float32,
            use_warmstart=True,
            warmstart_shift_steps=1,
            warmstart_nactions=torch.tensor(
                [[[90.0, 91.0], [70.0, 71.0]]]
            ),
            warmstart_indices=(1, 3),
        )

        torch.testing.assert_close(
            seed,
            torch.tensor(
                [[[2.0, 3.0], [90.0, 91.0], [6.0, 7.0], [70.0, 71.0]]]
            ),
        )
        torch.testing.assert_close(policy.prev_naction, previous)

    def test_reset_clears_previous_prediction(self):
        policy = object.__new__(DiffusionPolicy)
        torch.nn.Module.__init__(policy)
        policy.observations = []
        policy.actions = []
        policy.prev_naction = torch.ones(1, 1, 1)

        policy.reset()

        self.assertIsNone(policy.prev_naction)
        self.assertEqual(policy.observations, [])
        self.assertEqual(policy.actions, [])


if __name__ == "__main__":
    unittest.main()
