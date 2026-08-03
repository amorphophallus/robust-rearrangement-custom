import numpy as np

from src.eval.annotation_noise import (
    AnnotationNoisePhaseState,
    apply_annotation_noise,
    make_annotation_noise_config,
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
