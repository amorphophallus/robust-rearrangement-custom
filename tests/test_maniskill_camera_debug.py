import importlib.util
import lzma
import math
import pickle
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "data_collection"
    / "debug_maniskill_camera.py"
)
SPEC = importlib.util.spec_from_file_location("debug_maniskill_camera", MODULE_PATH)
camera_debug = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(camera_debug)


def test_camera_proposal_uses_sapien_forward_x_axis():
    proposal = camera_debug.camera_proposal(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        math.radians(40),
        2.0,
    )
    assert proposal["eye_world"] == [1.0, 2.0, 3.0]
    assert proposal["target_world"] == [3.0, 2.0, 3.0]
    assert proposal["fov_degrees"] == pytest.approx(40)


def test_apply_camera_contract_changes_only_three_assignments(tmp_path):
    contract = tmp_path / "camera_contract.py"
    contract.write_text(
        "import math\n\n"
        "RR_FRONT_EYE_WORLD = (1, 2, 3)\n"
        "RR_FRONT_TARGET_WORLD = (4, 5, 6)\n"
        "RR_FRONT_FOV_RADIANS = math.radians(40.0)\n\n"
        "SENTINEL = 'keep'\n"
    )
    proposal = {
        "eye_world": [0.1, 0.2, 0.3],
        "target_world": [-0.4, -0.5, -0.6],
        "fov_degrees": 45.0,
    }
    original_mode = contract.stat().st_mode
    camera_debug.apply_camera_contract(contract, proposal)
    result = contract.read_text()
    assert "RR_FRONT_EYE_WORLD = (0.1, 0.2, 0.3)" in result
    assert "RR_FRONT_TARGET_WORLD = (-0.4, -0.5, -0.6)" in result
    assert "RR_FRONT_FOV_RADIANS = math.radians(45)" in result
    assert "SENTINEL = 'keep'" in result
    assert contract.stat().st_mode == original_mode


def test_load_success_trajectory_requires_scripted_success(tmp_path):
    path = tmp_path / "trajectory.pkl.xz"
    with lzma.open(path, "wb") as stream:
        pickle.dump(
            {
                "task": "LiftPegUpright-v1",
                "success": True,
                "annotation_source": "scripted",
            },
            stream,
        )
    loaded = camera_debug.load_success_trajectory(path)
    assert loaded["task"] == "LiftPegUpright-v1"

    with lzma.open(path, "wb") as stream:
        pickle.dump(
            {
                "task": "LiftPegUpright-v1",
                "success": True,
                "annotation_source": "vlm",
            },
            stream,
        )
    with pytest.raises(ValueError, match="annotation_source=scripted"):
        camera_debug.load_success_trajectory(path)


def test_recorded_state_contract():
    observation = {
        "robot_state": {
            "joint_positions": np.arange(7, dtype=np.float32),
            "gripper_finger_1_pos": np.array([0.03], dtype=np.float32),
            "gripper_finger_2_pos": np.array([0.04], dtype=np.float32),
        }
    }
    trajectory = {
        "observations": [observation, observation],
        "actions": [[0.0] * 8],
    }
    assert camera_debug.recorded_frame_count(trajectory) == 2
    np.testing.assert_array_equal(
        camera_debug.recorded_qpos(observation),
        np.array([0, 1, 2, 3, 4, 5, 6, 0.03, 0.04], dtype=np.float32),
    )

    with pytest.raises(ValueError, match=r"T actions and T\+1 observations"):
        camera_debug.recorded_frame_count(
            {"observations": [observation, observation], "actions": []}
        )


def test_camera_cli_uses_square_contract_viewer_defaults(tmp_path):
    args = camera_debug.parse_args(
        [
            "--trajectory",
            str(tmp_path / "trajectory.pkl.xz"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output",
            str(tmp_path / "camera.json"),
        ]
    )
    assert args.viewer_size == 800
    assert args.scroll_speed == pytest.approx(0.02)


def test_motion_planning_replay_does_not_require_fake_checkpoint():
    provenance = camera_debug.rollout_provenance(
        "PegInsertionSide-v1", checkpoint=None
    )
    assert provenance == {
        "rollout_source": "bundled_panda_motion_planning_solver",
        "checkpoint": None,
        "checkpoint_sha256": None,
    }
