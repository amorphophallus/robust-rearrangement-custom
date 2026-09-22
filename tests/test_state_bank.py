import gzip
import pickle

import numpy as np
import pytest

from src.eval.state_bank import (
    STATE_BANK_SCHEMA,
    _EnvSkillClock,
    _active_part_stage,
    _randomness_name,
    load_state_record,
    save_state_record,
    state_arrays_for_restore,
    translate_root_state_origin,
    validate_state_record,
)


def make_record():
    return {
        "schema": STATE_BANK_SCHEMA,
        "metadata": {
            "task": "one_leg",
            "episode_index": 2,
            "frame_index": 19,
            "skill_state": "leg-top-place",
            "skill_frame_offset": 8,
            "annotation_source": "scripted",
        },
        "physics": {
            "root_state_refreshed": True,
            "root_state": np.arange(39, dtype=np.float32).reshape(3, 13),
            "dof_state": np.arange(18, dtype=np.float32).reshape(9, 2),
        },
    }


def test_zero_velocity_restore_preserves_positions_only():
    record = make_record()
    root, dof = state_arrays_for_restore(
        record["physics"], restore_velocity=False
    )
    np.testing.assert_array_equal(root[:, :7], record["physics"]["root_state"][:, :7])
    np.testing.assert_array_equal(dof[:, 0], record["physics"]["dof_state"][:, 0])
    np.testing.assert_array_equal(root[:, 7:], 0.0)
    np.testing.assert_array_equal(dof[:, 1], 0.0)


def test_full_velocity_restore_keeps_independent_copies():
    record = make_record()
    root, dof = state_arrays_for_restore(
        record["physics"], restore_velocity=True
    )
    np.testing.assert_array_equal(root, record["physics"]["root_state"])
    np.testing.assert_array_equal(dof, record["physics"]["dof_state"])
    root[0, 0] = -1
    dof[0, 0] = -1
    assert record["physics"]["root_state"][0, 0] != -1
    assert record["physics"]["dof_state"][0, 0] != -1


def test_root_state_translation_preserves_relative_poses():
    root = np.zeros((2, 13), dtype=np.float32)
    root[:, :3] = [[1, 2, 3], [4, 5, 6]]
    translated = translate_root_state_origin(
        root, saved_origin=[0, 0, 0], target_origin=[2, -3, 0.5]
    )
    np.testing.assert_allclose(
        translated[:, :3], [[3, -1, 3.5], [6, 2, 6.5]]
    )
    np.testing.assert_allclose(
        translated[1, :3] - translated[0, :3], root[1, :3] - root[0, :3]
    )


def test_skill_clock_tracks_offset_and_repeated_visits():
    clock = _EnvSkillClock()
    assert clock.update("pick", 10) == (0, 0)
    assert clock.update("pick", 18) == (0, 8)
    assert clock.update("place", 19) == (0, 0)
    assert clock.update("pick", 30) == (1, 0)


def test_active_part_stage_exposes_internal_scripted_fsm():
    class Part:
        name = "leg"
        skill_state = "pick"
        _state = "close_gripper"

    class Furniture:
        parts = [Part()]

    class Annotator:
        furniture = Furniture()

    class Env:
        _skill_annotators = [Annotator()]

    stage = _active_part_stage(
        Env(), 0, {"debug": {"active_part": "leg"}}
    )
    assert stage == {
        "active_part": "leg",
        "skill_stage": "pick",
        "part_skill_stage": "pick",
        "legacy_part_fsm_state": "close_gripper",
    }


def test_randomness_enum_uses_cli_name():
    class Randomness:
        name = "MEDIUM"

    assert _randomness_name(Randomness()) == "med"


def test_state_record_round_trip_is_validated(tmp_path):
    path = tmp_path / "state.pkl.gz"
    digest = save_state_record(path, make_record())
    assert len(digest) == 64
    restored = load_state_record(path)
    np.testing.assert_array_equal(
        restored["physics"]["root_state"], make_record()["physics"]["root_state"]
    )


def test_state_record_rejects_non_scripted_provenance():
    record = make_record()
    record["metadata"]["annotation_source"] = "vlm"
    with pytest.raises(ValueError, match="scripted"):
        validate_state_record(record)


def test_state_record_rejects_stale_actor_root_tensor():
    record = make_record()
    del record["physics"]["root_state_refreshed"]
    with pytest.raises(ValueError, match="refreshed actor-root"):
        validate_state_record(record)


def test_load_rejects_unvalidated_pickle(tmp_path):
    path = tmp_path / "invalid.pkl.gz"
    with gzip.open(path, "wb") as stream:
        pickle.dump({"schema": "wrong"}, stream)
    with pytest.raises(ValueError, match="schema"):
        load_state_record(path)
