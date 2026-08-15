import numpy as np

from scripts.recompute_saved_rollout_tracking import (
    _episode_tracking_errors,
    _observation_segments,
)


def _observation(skill: str, ee_x: float) -> dict:
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = np.asarray([0.2, 0.0, 0.5])
    return {
        "skill": skill,
        "robot_state": {
            "ee_pos_sim": np.asarray([0.2 + ee_x, 0.0, 0.5]),
            "ee_quat_sim": np.asarray([0.0, 0.0, 0.0, 1.0]),
        },
        "guidance_pose": target,
    }


def test_segments_align_forward_progress_and_fallback_to_semantic_states():
    observations = [
        _observation("push", 0.0),
        _observation("pick", 0.0),
        _observation("place", 0.0),
        _observation("insert", 0.0),
        _observation("screw", 0.0),
        _observation("pick", 0.0),
        _observation("place", 0.0),
        _observation("pick", 0.0),
    ]

    labels = [label for label, _, _ in _observation_segments(observations, "round_table")]

    assert labels == [
        "top-leg-push",
        "leg-top-pick",
        "leg-top-place",
        "leg-top-insert",
        "leg-top-screw",
        "base-leg-pick",
        "base-leg-place",
        "base-leg-pick",
    ]


def test_reentered_semantic_state_keeps_minimum_position_error():
    trajectory = {
        "observations": [
            _observation("push", 0.5),
            _observation("pick", 0.2),
            _observation("place", 0.3),
            _observation("pick", 0.1),
        ]
    }

    errors = _episode_tracking_errors(
        trajectory,
        task="round_table",
        metric_type="position",
    )

    assert np.isclose(errors["leg-top-pick"]["pos_m"], 0.1)
