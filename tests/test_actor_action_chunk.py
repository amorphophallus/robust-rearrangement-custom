import unittest
from collections import deque

import torch

from src.behavior.base import Actor


class ActorActionChunkTest(unittest.TestCase):
    def test_action_chunk_returns_fresh_full_chunk_and_clears_step_queue(self):
        actor = object.__new__(Actor)
        actor.obs_horizon = 2
        actor.flatten_obs = True
        actor.observations = deque(maxlen=2)
        actor.actions = deque([torch.tensor([[99.0]])], maxlen=3)
        actor._normalized_obs = lambda observations, flatten: torch.tensor([[1.0]])
        actor._sample_action_pred = lambda normalized: deque(
            [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
        )

        observation = {"robot_state": torch.zeros(1, 14)}
        chunk = Actor.action_chunk(actor, observation)

        self.assertEqual(tuple(chunk.shape), (1, 2, 2))
        torch.testing.assert_close(
            chunk, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        )
        self.assertEqual(len(actor.observations), 2)
        self.assertEqual(len(actor.actions), 0)


if __name__ == "__main__":
    unittest.main()
