import numpy as np

from src.eval.annotation_noise import (
    AnnotationNoisePhaseState,
    apply_annotation_noise,
    build_annotation_noise_summary,
    make_annotation_noise_config,
    load_guidance_shuffle_bank,
    write_guidance_shuffle_bank,
    generate_fixed_guidance_point_noise,
)
import pytest


def test_fixed_n2_n4_noise_is_phase_stable_and_exactly_shared():
    points = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.2, 0.3, 0.4], dtype=np.float32),
        np.array([0.3, 0.4, 0.5], dtype=np.float32),
    ]
    variants = generate_fixed_guidance_point_noise(
        points,
        [("pick", 0), ("pick", 0), ("place", 0)],
        seed=0,
        episode_index=2,
    )

    np.testing.assert_array_equal(
        variants["standard_noise"][0], variants["standard_noise"][1]
    )
    assert not np.array_equal(
        variants["standard_noise"][1], variants["standard_noise"][2]
    )
    for clean, n2, n4 in zip(points, variants["n2"], variants["n4"]):
        np.testing.assert_allclose(n4 - clean, 4.0 * (n2 - clean), atol=1e-7)
        assert np.max(np.abs(n2 - clean)) <= 0.0120001
        assert np.max(np.abs(n4 - clean)) <= 0.0480001


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


def test_fixed_geodesic_180_is_deterministic_phase_stable_and_valid_rotation():
    clean_pose = np.eye(4, dtype=np.float32)
    config = make_annotation_noise_config(
        mode="fixed_geodesic",
        pos_std_m=0.0,
        ori_std_deg=180.0,
        seed=17,
        apply_to="all",
    )
    state = AnnotationNoisePhaseState(env_idx=1)
    kwargs = dict(
        guidance_point=np.zeros(3, dtype=np.float32),
        guidance_pose=clean_pose,
        skill="pick",
        phase_key=("pick", 0),
        state=state,
        config=config,
    )
    first = apply_annotation_noise(**kwargs)
    same = apply_annotation_noise(**kwargs)
    rotation = first[1][:3, :3]

    np.testing.assert_allclose(first[1], same[1])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)
    assert first[2]["target_geodesic_deg"] == 180.0
    assert first[2]["realized_ori_geodesic_deg"] == pytest.approx(180.0, abs=1e-3)
    assert np.linalg.norm(first[2]["sampled_axis"]) == pytest.approx(1.0, abs=1e-6)

    other_seed = apply_annotation_noise(
        **{**kwargs, "state": AnnotationNoisePhaseState(env_idx=1),
           "config": make_annotation_noise_config(
               mode="fixed_geodesic", ori_std_deg=180.0, seed=18
           )}
    )
    assert not np.allclose(first[2]["sampled_axis"], other_seed[2]["sampled_axis"])


def test_fixed_geodesic_rejects_position_noise_and_out_of_range_angle():
    with pytest.raises(ValueError, match="position std"):
        make_annotation_noise_config(
            mode="fixed_geodesic", pos_std_m=0.001, ori_std_deg=180.0
        )
    with pytest.raises(ValueError, match=r"\[0, 180\]"):
        make_annotation_noise_config(mode="fixed_geodesic", ori_std_deg=180.1)


def test_annotation_noise_summary_keeps_invalid_samples_in_denominator():
    summary = build_annotation_noise_summary(
        [
            {
                "realized_pos_norm_m": 0.03,
                "realized_ori_geodesic_deg": 20.0,
                "apply_pose_pos": True,
                "apply_ori": True,
                "target_finite": True,
                "workspace_valid": True,
                "front_projection_visible": True,
            },
            {
                "realized_pos_norm_m": 0.04,
                "realized_ori_geodesic_deg": 40.0,
                "apply_pose_pos": True,
                "apply_ori": True,
                "target_finite": False,
                "workspace_valid": False,
                "front_projection_visible": False,
            },
        ]
    )
    assert summary["phase_count"] == 2
    assert summary["workspace_valid_rate"] == 0.5
    assert summary["front_projection_visible_rate"] == 0.5
    assert summary["invalid_nonfinite_rate"] == 0.5
    assert summary["position_norm_m"]["rms"] == pytest.approx(
        np.sqrt((0.03**2 + 0.04**2) / 2)
    )


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
    assert info["realized_pos_norm_m"] == info["realized_pos_displacement_m"]
    assert info["realized_ori_geodesic_deg"] == info["realized_ori_displacement_deg"]
    assert info["apply_ori"] is True


def test_shuffle_falls_back_to_any_skill_but_never_the_current_state():
    current_state_record = _shuffle_records()[0]
    current_state_record = {
        **current_state_record,
        "skill_state": "leg-top-place",
        "skill_type": "place",
    }
    different_skill_record = _shuffle_records()[1]
    different_skill_record = {
        **different_skill_record,
        "skill_state": "leg-top-screw",
        "skill_type": "screw",
    }
    config = make_annotation_noise_config(
        mode="shuffle",
        apply_to="point",
        shuffle_records=[current_state_record, different_skill_record],
    )

    _, _, info = apply_annotation_noise(
        guidance_point=np.zeros(3, dtype=np.float32),
        guidance_pose=np.eye(4, dtype=np.float32),
        task="one_leg",
        skill_state="leg-top-place",
        skill="place",
        phase_key=("leg-top-place", 0),
        state=AnnotationNoisePhaseState(env_idx=0),
        config=config,
    )

    assert info["donor_skill_state"] == "leg-top-screw"
    assert info["donor_skill_type"] == "screw"
    assert info["selection_policy"] == "any_skill_different_state"


def test_shuffle_rejects_bank_without_a_different_semantic_state():
    record = {
        **_shuffle_records()[0],
        "skill_state": "leg-top-place",
        "skill_type": "place",
    }
    config = make_annotation_noise_config(
        mode="shuffle",
        apply_to="point",
        shuffle_records=[record],
    )

    try:
        apply_annotation_noise(
            guidance_point=np.zeros(3, dtype=np.float32),
            guidance_pose=np.eye(4, dtype=np.float32),
            task="one_leg",
            skill_state="leg-top-place",
            skill="place",
            phase_key=("leg-top-place", 0),
            state=AnnotationNoisePhaseState(env_idx=0),
            config=config,
        )
    except ValueError as exc:
        assert "No different-state shuffled guidance donor" in str(exc)
    else:
        raise AssertionError("same-state shuffle donor should have been rejected")


def test_guidance_bank_round_trip(tmp_path):
    path = tmp_path / "one_leg.json"
    records = _shuffle_records()
    write_guidance_shuffle_bank(path, task="one_leg", records=records)
    assert load_guidance_shuffle_bank(path) == records
    payload = __import__("json").loads(path.read_text())
    assert payload["version"] == 2
    assert payload["guidance_frame"] == "robot-base"
