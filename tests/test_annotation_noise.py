import numpy as np

from src.eval.annotation_noise import (
    AnnotationNoisePhaseState,
    apply_annotation_noise,
    make_annotation_noise_config,
    load_guidance_shuffle_bank,
    write_guidance_shuffle_bank,
)


def test_point_noise_keeps_tracking_pose_position_aligned_with_drawn_point():
    point = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = point
    config = make_annotation_noise_config(
        pos_std_m=0.05,
        seed=7,
        apply_to="point",
    )

    noisy_point, noisy_pose, info = apply_annotation_noise(
        guidance_point=point,
        guidance_pose=pose,
        skill="pick",
        phase_key=("assemble", "top-leg", "top-leg-pick"),
        state=AnnotationNoisePhaseState(env_idx=0),
        config=config,
    )

    np.testing.assert_allclose(noisy_pose[:3, 3], noisy_point)
    np.testing.assert_allclose(noisy_pose[:3, :3], np.eye(3))
    assert info["apply_point_pos"] is True
    assert info["apply_pose_pos"] is True
    assert info["apply_ori"] is False


def test_noise_is_stable_within_phase_and_resampled_when_phase_repeats():
    config = make_annotation_noise_config(pos_std_m=0.05, seed=11, apply_to="point")
    state = AnnotationNoisePhaseState(env_idx=0)
    point = np.zeros(3, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)

    first = apply_annotation_noise(
        guidance_point=point,
        guidance_pose=pose,
        skill="pick",
        phase_key=("pick", 0),
        state=state,
        config=config,
    )
    same_phase = apply_annotation_noise(
        guidance_point=point,
        guidance_pose=pose,
        skill="pick",
        phase_key=("pick", 0),
        state=state,
        config=config,
    )
    apply_annotation_noise(
        guidance_point=point,
        guidance_pose=pose,
        skill="place",
        phase_key=("place", 0),
        state=state,
        config=config,
    )
    repeated = apply_annotation_noise(
        guidance_point=point,
        guidance_pose=pose,
        skill="pick",
        phase_key=("pick", 0),
        state=state,
        config=config,
    )

    np.testing.assert_allclose(first[0], same_phase[0])
    assert not np.allclose(first[0], repeated[0])


def _shuffle_records():
    first_pose = np.eye(4, dtype=np.float32)
    first_pose[:3, 3] = [0.4, 0.1, 0.2]
    second_pose = np.eye(4, dtype=np.float32)
    second_pose[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    second_pose[:3, 3] = [-0.2, 0.3, 0.5]
    return [
        {
            "task": "one_leg",
            "skill_state": "leg-top-pick",
            "skill_type": "pick",
            "source_episode": 1,
            "visit_idx": 0,
            "guidance_point": first_pose[:3, 3].tolist(),
            "guidance_pose": first_pose.tolist(),
        },
        {
            "task": "one_leg",
            "skill_state": "base-leg-pick",
            "skill_type": "pick",
            "source_episode": 2,
            "visit_idx": 0,
            "guidance_point": second_pose[:3, 3].tolist(),
            "guidance_pose": second_pose.tolist(),
        },
    ]


def test_point_shuffle_is_phase_stable_and_preserves_clean_orientation():
    clean_point = np.zeros(3, dtype=np.float32)
    clean_pose = np.eye(4, dtype=np.float32)
    state = AnnotationNoisePhaseState(env_idx=0)
    config = make_annotation_noise_config(
        mode="shuffle",
        apply_to="point",
        shuffle_seed=3,
        shuffle_records=_shuffle_records(),
    )
    kwargs = {
        "guidance_point": clean_point,
        "guidance_pose": clean_pose,
        "task": "one_leg",
        "skill_state": "top-leg-pick",
        "skill": "pick",
        "state": state,
        "config": config,
    }
    first = apply_annotation_noise(phase_key=("top-leg-pick", 0), **kwargs)
    same = apply_annotation_noise(phase_key=("top-leg-pick", 0), **kwargs)

    np.testing.assert_allclose(first[0], same[0])
    np.testing.assert_allclose(first[1][:3, 3], first[0])
    np.testing.assert_allclose(first[1][:3, :3], clean_pose[:3, :3])
    assert first[2]["phase_idx"] == same[2]["phase_idx"]

    apply_annotation_noise(phase_key=("top-leg-place", 0), **kwargs)
    repeated = apply_annotation_noise(phase_key=("top-leg-pick", 0), **kwargs)
    assert repeated[2]["phase_idx"] > first[2]["phase_idx"]


def test_pose_shuffle_replaces_orientation_and_records_realized_error():
    clean_pose = np.eye(4, dtype=np.float32)
    config = make_annotation_noise_config(
        mode="shuffle",
        apply_to="all",
        shuffle_records=[_shuffle_records()[1]],
    )
    point, pose, info = apply_annotation_noise(
        guidance_point=np.zeros(3, dtype=np.float32),
        guidance_pose=clean_pose,
        task="one_leg",
        skill_state="top-leg-pick",
        skill="pick",
        phase_key=("top-leg-pick", 0),
        state=AnnotationNoisePhaseState(env_idx=0),
        config=config,
    )

    np.testing.assert_allclose(point, [-0.2, 0.3, 0.5])
    assert not np.allclose(pose[:3, :3], clean_pose[:3, :3])
    assert info["realized_pos_displacement_m"] > 0.0
    assert info["realized_ori_displacement_deg"] > 0.0


def test_guidance_bank_round_trip(tmp_path):
    path = tmp_path / "one_leg.json"
    records = _shuffle_records()
    write_guidance_shuffle_bank(path, task="one_leg", records=records)
    assert load_guidance_shuffle_bank(path) == records
