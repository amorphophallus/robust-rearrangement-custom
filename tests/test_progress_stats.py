import numpy as np

from src.eval.progress_schema import (
    accumulate_episode_skill_stats,
    accumulate_tracking_error_records,
    build_tracking_error_summary,
    compute_success_rates,
    compute_episode_tracking_errors,
    get_task_progress_labels,
)


def test_episode_progress_completes_states_before_last_reached_state():
    state_counts = {}
    skill_completion_counts = {}
    step_counts = {}
    step_completion_counts = {}

    accumulate_episode_skill_stats(
        state_labels=[
            "base-bulb-push",
            "base-bulb-push",
            "bulb-base-pick",
            "bulb-base-place",
        ],
        step_labels=["base-bulb", "base-bulb"],
        success=False,
        state_counts=state_counts,
        skill_completion_counts=skill_completion_counts,
        step_counts=step_counts,
        step_completion_counts=step_completion_counts,
    )

    assert state_counts == {
        "base-bulb-push": 1,
        "bulb-base-pick": 1,
        "bulb-base-place": 1,
    }
    assert skill_completion_counts == {
        "base-bulb-push": 1,
        "bulb-base-pick": 1,
    }
    assert step_counts == {"base-bulb": 1}
    assert step_completion_counts == {}


def test_episode_progress_completes_final_state_and_step_on_success():
    state_counts = {}
    skill_completion_counts = {}
    step_counts = {}
    step_completion_counts = {}

    accumulate_episode_skill_stats(
        state_labels=[
            "top-leg-pick",
            "top-leg-push",
            "leg-top-pick",
            "leg-top-place",
            "leg-top-insert",
            "leg-top-screw",
        ],
        step_labels=["top-leg"],
        success=True,
        state_counts=state_counts,
        skill_completion_counts=skill_completion_counts,
        step_counts=step_counts,
        step_completion_counts=step_completion_counts,
    )

    assert state_counts["leg-top-screw"] == 1
    assert skill_completion_counts["leg-top-screw"] == 1
    assert step_counts == {"top-leg": 1}
    assert step_completion_counts == {"top-leg": 1}
    assert compute_success_rates(step_counts, step_completion_counts) == {
        "top-leg": 1.0
    }


def test_lamp_progress_schema_includes_hood_place_state():
    assert get_task_progress_labels("lamp", "skill_states") == [
        "base-bulb-push",
        "bulb-base-pick",
        "bulb-base-place",
        "bulb-base-insert",
        "bulb-base-screw",
        "hood-base-pick",
        "hood-base-place",
    ]


def _robot_state(pos, quat=(0.0, 0.0, 0.0, 1.0)):
    return {
        "ee_pos": np.asarray(pos, dtype=np.float32),
        "ee_quat": np.asarray(quat, dtype=np.float32),
    }


def _robot_state_with_sim_pose(
    pos,
    sim_pos,
    quat=(0.0, 0.0, 0.0, 1.0),
    sim_quat=(0.0, 0.0, 0.0, 1.0),
):
    state = _robot_state(pos, quat=quat)
    state["ee_pos_sim"] = np.asarray(sim_pos, dtype=np.float32)
    state["ee_quat_sim"] = np.asarray(sim_quat, dtype=np.float32)
    return state


def _target_pose(pos):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray(pos, dtype=np.float32)
    return pose


def test_tracking_error_uses_final_frame_of_each_skill_phase():
    errors = compute_episode_tracking_errors(
        robot_states=[
            _robot_state([0.0, 0.0, 0.0]),
            _robot_state([0.1, 0.0, 0.0]),
            _robot_state([0.2, 0.0, 0.0]),
        ],
        skill_states=["top-leg-pick", "top-leg-pick", "leg-top-place"],
        target_poses=[
            _target_pose([1.0, 0.0, 0.0]),
            _target_pose([0.1, 0.0, 0.0]),
            _target_pose([0.25, 0.0, 0.0]),
        ],
    )

    assert errors["top-leg-pick"]["pos_m"] == 0.0
    assert np.isclose(errors["leg-top-place"]["pos_m"], 0.05)


def test_tracking_error_prefers_sim_local_ee_pose_when_available():
    errors = compute_episode_tracking_errors(
        robot_states=[
            _robot_state_with_sim_pose(
                pos=[0.5, 0.0, 0.4],
                sim_pos=[0.1, 0.0, 0.0],
            )
        ],
        skill_states=["top-leg-pick"],
        target_poses=[_target_pose([0.1, 0.0, 0.0])],
    )

    assert errors["top-leg-pick"]["pos_m"] == 0.0


def test_tracking_error_keeps_min_total_for_repeated_skill_state():
    errors = compute_episode_tracking_errors(
        robot_states=[
            _robot_state([0.0, 0.0, 0.0]),
            _robot_state([0.0, 0.0, 0.0]),
            _robot_state([1.0, 0.0, 0.0]),
            _robot_state([1.0, 0.0, 0.0]),
        ],
        skill_states=[
            "top-leg-pick",
            "leg-top-place",
            "top-leg-pick",
            "top-leg-pick",
        ],
        target_poses=[
            _target_pose([0.5, 0.0, 0.0]),
            _target_pose([0.5, 0.0, 0.0]),
            _target_pose([2.0, 0.0, 0.0]),
            _target_pose([1.01, 0.0, 0.0]),
        ],
    )

    assert np.isclose(errors["top-leg-pick"]["pos_m"], 0.01)


def test_tracking_error_treats_opposite_quaternion_sign_as_same_orientation():
    errors = compute_episode_tracking_errors(
        robot_states=[_robot_state([0.0, 0.0, 0.0], quat=(0.0, 0.0, 0.0, -1.0))],
        skill_states=["top-leg-pick"],
        target_poses=[_target_pose([0.0, 0.0, 0.0])],
    )

    assert np.isclose(errors["top-leg-pick"]["ori_deg"], 0.0)


def test_tracking_error_summary_includes_expected_empty_labels_and_overall():
    accumulator = {}
    accumulate_tracking_error_records(
        accumulator,
        {
            "top-leg-pick": {
                "pos_m": 0.01,
                "ori_deg": 5.0,
                "total": 2.0,
            }
        },
    )

    summary = build_tracking_error_summary(
        accumulator,
        expected_labels=["top-leg-pick", "leg-top-place"],
    )

    assert summary["by_skill"]["top-leg-pick"]["count"] == 1
    assert summary["by_skill"]["leg-top-place"]["count"] == 0
    assert summary["overall"]["mean_total"] == 2.0
