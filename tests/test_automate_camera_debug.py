import importlib.util
import lzma
import pickle
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "data_collection"
    / "debug_automate_camera.py"
)
SPEC = importlib.util.spec_from_file_location("debug_automate_camera", MODULE_PATH)
camera_debug = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(camera_debug)


def _observation():
    return {
        "robot_state": {
            "joint_positions": np.arange(7, dtype=np.float32),
            "gripper_finger_1_pos": np.array([0.03], dtype=np.float32),
            "gripper_finger_2_pos": np.array([0.04], dtype=np.float32),
        },
        "parts_poses": np.arange(14, dtype=np.float32),
        "guidance_point": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }


def test_camera_proposal_uses_usd_opengl_negative_z_forward():
    proposal = camera_debug.camera_proposal(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        2.0,
    )
    assert proposal["eye_world"] == [1.0, 2.0, 3.0]
    assert proposal["target_world"] == [1.0, 2.0, 1.0]
    assert proposal["rot_opengl_wxyz"] == [1.0, 0.0, 0.0, 0.0]


def test_load_trajectory_requires_raw_scripted_automate_contract(tmp_path):
    observation = _observation()
    payload = {
        "task": "automate_insertion_00410",
        "env": "AutoMate",
        "success": False,
        "annotation_source": "scripted",
        "image_annotation_mode": "none",
        "observations": [observation, observation],
        "actions": [np.zeros(8, dtype=np.float32)],
    }
    path = tmp_path / "trajectory.pkl.xz"
    with lzma.open(path, "wb") as stream:
        pickle.dump(payload, stream)
    loaded = camera_debug.load_trajectory(path)
    assert loaded["_camera_debug_assembly_id"] == "00410"

    payload["annotation_source"] = "vlm"
    with lzma.open(path, "wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="annotation_source=scripted"):
        camera_debug.load_trajectory(path)


def test_load_trajectory_rejects_excluded_00755(tmp_path):
    observation = _observation()
    payload = {
        "task": "automate_insertion_00755",
        "env": "AutoMate",
        "success": False,
        "annotation_source": "scripted",
        "image_annotation_mode": "none",
        "observations": [observation, observation],
        "actions": [np.zeros(8, dtype=np.float32)],
    }
    path = tmp_path / "trajectory.pkl.xz"
    with lzma.open(path, "wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="excluded from the 99-task"):
        camera_debug.load_trajectory(path)


def test_recorded_qpos_and_camera_cfg_lines():
    np.testing.assert_array_equal(
        camera_debug.recorded_qpos(_observation()),
        np.array([0, 1, 2, 3, 4, 5, 6, 0.03, 0.04], dtype=np.float32),
    )
    lines = camera_debug.camera_cfg_lines(
        {"pos": [1, 2, 3], "rot_opengl_wxyz": [1, 0, 0, 0]}
    )
    assert lines == [
        "pos=(1, 2, 3),",
        "rot=(1, 0, 0, 0),",
        'convention="opengl",',
    ]
