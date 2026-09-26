import pytest

from scripts import select_skill_level_states as selector
from scripts.report_furniturebench_skill_level import (
    paired_episode_cluster_bootstrap,
)

from src.eval.skill_level import (
    adjudicate_stage_step,
    assembly_pair_mask_value,
    calibrated_stage_timeout,
    classify_stage_transition,
    default_stage_timeout,
    stage_assembly_pair_index,
    stage_skill,
    summarize_skill_level_records,
    stable_annotation_noise_seed_offset,
)
from src.eval.noisy_skill_level import paired_endpoint_metrics


def test_stage_transition_requires_forward_progress():
    assert (
        classify_stage_transition("round_table", "leg-top-place", "leg-top-insert")
        == "success"
    )
    assert (
        classify_stage_transition("round_table", "leg-top-place", "leg-top-pick")
        == "wrong_stage"
    )
    assert classify_stage_transition("round_table", "leg-top-place", "leg-top-place") is None
    assert classify_stage_transition("round_table", "leg-top-place", "unknown") == "wrong_stage"


def test_stage_timeout_is_defined_by_skill_type():
    assert stage_skill("bulb-base-screw") == "screw"
    assert default_stage_timeout("bulb-base-screw") == 360
    with pytest.raises(ValueError, match="Unsupported"):
        default_stage_timeout("bulb-base-insert")


def test_selector_progress_stratum_boundaries_and_gaps():
    assert selector.progress_stratum(0.10) == "early"
    assert selector.progress_stratum(0.33) == "early"
    assert selector.progress_stratum(0.34) is None
    assert selector.progress_stratum(0.40) == "middle"
    assert selector.progress_stratum(0.65) == "middle"
    assert selector.progress_stratum(0.66) is None
    assert selector.progress_stratum(0.70) == "late"
    assert selector.progress_stratum(0.85) == "late"
    assert selector.progress_stratum(0.86) is None


def test_only_pair_finalizers_use_assembly_completion_masks():
    assert stage_assembly_pair_index("lamp", "bulb-base-screw") == 0
    assert stage_assembly_pair_index("lamp", "hood-base-place") == 1
    assert stage_assembly_pair_index("lamp", "bulb-base-place") is None
    assert assembly_pair_mask_value([True, False], 0)
    assert not assembly_pair_mask_value([True, False], 1)


def test_nonterminal_pair_finalizer_uses_fsm_transition_not_assembly_mask():
    completed, reason = adjudicate_stage_step(
        "success",
        completion_pair_index=0,
        pair_completed=False,
        task_final_stage=False,
        environment_done=False,
    )
    assert completed
    assert reason == "stage_transition"

    completed, reason = adjudicate_stage_step(
        None,
        completion_pair_index=0,
        pair_completed=True,
        task_final_stage=False,
        environment_done=False,
    )
    assert not completed
    assert reason is None


def test_non_finalizer_uses_forward_fsm_transition_for_success():
    completed, reason = adjudicate_stage_step(
        "success",
        completion_pair_index=None,
        pair_completed=False,
        task_final_stage=False,
        environment_done=False,
    )
    assert completed
    assert reason == "stage_transition"


def test_final_stage_accepts_same_step_environment_done_and_completed_pair():
    completed, reason = adjudicate_stage_step(
        None,
        completion_pair_index=1,
        pair_completed=True,
        task_final_stage=True,
        environment_done=True,
    )
    assert completed
    assert reason == "final_stage_environment_done"


def test_final_stage_fallback_requires_done_pair_and_no_wrong_transition():
    for transition, pair_completed, environment_done in (
        (None, False, True),
        (None, True, False),
        ("wrong_stage", True, True),
    ):
        completed, _ = adjudicate_stage_step(
            transition,
            completion_pair_index=1,
            pair_completed=pair_completed,
            task_final_stage=True,
            environment_done=environment_done,
        )
        assert not completed


def test_terminal_fsm_done_is_a_forward_transition():
    assert classify_stage_transition("lamp", "hood-base-place", "done") == "success"


def test_stage_timeout_covers_slowest_selected_expert_with_margin():
    manifest = [
        {"stage_length_frames": 524, "skill_frame_offset": 69},
        {"stage_length_frames": 200, "skill_frame_offset": 100},
    ]
    timeout, audit = calibrated_stage_timeout("leg-top-screw", manifest)
    assert audit["expert_remaining_max_steps"] == 454
    assert timeout == 580


def test_stage_timeout_keeps_skill_floor_for_short_expert_paths():
    manifest = [{"stage_length_frames": 50, "skill_frame_offset": 10}]
    timeout, audit = calibrated_stage_timeout("leg-top-place", manifest)
    assert timeout == 240
    assert audit["resolved_steps"] == 240


def test_summary_keeps_failures_in_tracking_metrics():
    rows = [
        {
            "state_sha256": "a",
            "selection_stratum": "early",
            "completed_current_stage": True,
            "termination_reason": "stage_transition",
            "e_gt_m": 0.01,
            "te_position_m": 0.01,
            "te_orientation_deg": 5.0,
            "te_normalized_total": 2.0,
            "completion_steps": 10,
        },
        {
            "state_sha256": "b",
            "selection_stratum": "late",
            "completed_current_stage": False,
            "termination_reason": "timeout",
            "e_gt_m": 0.03,
            "te_position_m": 0.03,
            "te_orientation_deg": 15.0,
            "te_normalized_total": 6.0,
            "completion_steps": 20,
        },
    ]

    summary = summarize_skill_level_records(rows)

    assert summary["success_rate"] == 0.5
    assert summary["e_gt_cm"]["count"] == 2
    assert summary["e_gt_cm"]["mean"] == pytest.approx(2.0)
    assert summary["e_gt_success_only_cm"]["mean"] == pytest.approx(1.0)
    assert summary["te_position_cm"]["mean"] == pytest.approx(2.0)
    assert summary["by_stratum"]["middle"]["attempted"] == 0


def test_noise_seed_offset_is_state_repeat_keyed_and_deterministic():
    first = stable_annotation_noise_seed_offset("abc", 7)
    assert first == stable_annotation_noise_seed_offset("abc", 7)
    assert first != stable_annotation_noise_seed_offset("abd", 7)
    assert first != stable_annotation_noise_seed_offset("abc", 8)


def test_paired_endpoint_metrics_execute_declared_formulae():
    clean = {"end_ee_pos_robot_base_m": [1.0, 2.0, 3.0]}
    noisy = {
        "end_ee_pos_robot_base_m": [1.02, 2.0, 3.0],
        "p0_robot_base_m": [1.0, 2.0, 3.0],
        "p_delta_robot_base_m": [1.01, 2.0, 3.0],
    }
    metrics = paired_endpoint_metrics(clean, noisy)
    assert metrics["e_gt_m"] == pytest.approx(0.02)
    assert metrics["e_input_m"] == pytest.approx(0.01)
    assert metrics["delta_x_m"] == pytest.approx(0.02)
    assert metrics["guidance_following_displacement_m"] == pytest.approx(0.02)
    assert metrics["guidance_gain"] == pytest.approx(2.0)


def test_paired_endpoint_metrics_leave_clean_displacement_metrics_undefined():
    clean = {"end_ee_pos_robot_base_m": [1.0, 2.0, 3.0]}
    noisy = {
        "end_ee_pos_robot_base_m": [1.0, 2.0, 3.0],
        "p0_robot_base_m": [0.0, 0.0, 0.0],
        "p_delta_robot_base_m": [0.0, 0.0, 0.0],
    }
    metrics = paired_endpoint_metrics(clean, noisy)
    assert metrics["delta_x_m"] is None
    assert metrics["guidance_following_displacement_m"] is None
    assert metrics["guidance_gain"] is None


def test_assembly_pair_completion_uses_requested_mask_index():
    record = {
        "physics": {"runtime": {"already_assembled": [True, False]}}
    }
    assert selector.record_assembly_pair_is_complete(record, 0)
    assert not selector.record_assembly_pair_is_complete(record, 1)
    with pytest.raises(ValueError, match="outside runtime mask"):
        selector.record_assembly_pair_is_complete(record, 2)


def test_lamp_screw_endpoint_is_first_pair_completion(monkeypatch):
    entries = [
        {"source_bank": "/bank", "path": f"state-{idx}", "frame_index": 20 + idx}
        for idx in range(4)
    ]
    masks = {
        "state-0": [False, False],
        "state-1": [False, False],
        "state-2": [True, False],
        "state-3": [True, False],
    }

    def fake_load(path):
        return {
            "physics": {
                "runtime": {"already_assembled": masks[path.name]}
            }
        }

    monkeypatch.setattr(selector, "load_state_record", fake_load)
    endpoint = selector.first_assembly_completion_entry(
        entries, task="lamp", stage="bulb-base-screw"
    )
    assert endpoint is entries[2]


def test_non_completion_stage_does_not_consume_assembly_mask(monkeypatch):
    def unexpected_load(_path):
        raise AssertionError("non-completion stages must not load runtime masks")

    monkeypatch.setattr(selector, "load_state_record", unexpected_load)
    assert (
        selector.first_assembly_completion_entry(
            [{"source_bank": "/bank", "path": "state"}],
            task="lamp",
            stage="bulb-base-place",
        )
        is None
    )


def _lamp_transition_entries(*stages):
    return [
        {"frame_index": index, "skill_state": stage}
        for index, stage in enumerate(stages)
    ]


def test_lamp_bulb_screw_episode_gate_accepts_three_frame_hood_pick():
    entries = _lamp_transition_entries(
        "bulb-base-screw",
        "bulb-base-screw",
        *("hood-base-pick",) * 3,
        "hood-base-place",
        "hood-base-place",
    )
    passed, reason, audit = selector.lamp_bulb_screw_episode_gate(entries)
    assert passed
    assert reason == "pass"
    assert audit["min_hood_pick_frames"] == 3


def test_lamp_bulb_screw_episode_gate_rejects_two_frame_hood_pick():
    entries = _lamp_transition_entries(
        "bulb-base-screw",
        *("hood-base-pick",) * 2,
        "hood-base-place",
    )
    passed, reason, _audit = selector.lamp_bulb_screw_episode_gate(entries)
    assert not passed
    assert reason == "hood_pick_too_short"


def test_lamp_bulb_screw_episode_gate_rejects_direct_place():
    entries = _lamp_transition_entries(
        "bulb-base-screw", "hood-base-place", "hood-base-place"
    )
    passed, reason, _audit = selector.lamp_bulb_screw_episode_gate(entries)
    assert not passed
    assert reason == "next_stage_not_hood_pick"


def test_lamp_bulb_screw_episode_gate_requires_place_after_pick():
    entries = _lamp_transition_entries(
        "bulb-base-screw", *("hood-base-pick",) * 3
    )
    passed, reason, _audit = selector.lamp_bulb_screw_episode_gate(entries)
    assert not passed
    assert reason == "missing_stage_after_hood_pick"


def test_paired_bootstrap_resamples_source_episode_clusters():
    left = {f"state-{index}": float(index) for index in range(48)}
    right = {key: value - 1.0 for key, value in left.items()}
    episode_by_state = {
        f"state-{index}": f"episode-{index // 2}" for index in range(48)
    }
    result = paired_episode_cluster_bootstrap(
        left,
        right,
        episode_by_state=episode_by_state,
        samples=100,
        seed=923999,
    )
    assert result["paired_state_count"] == 48
    assert result["paired_episode_count"] == 24
    assert result["max_states_per_episode"] == 2
    assert result["bootstrap_unit"] == "source_episode_cluster"
    assert result["estimate"] == pytest.approx(1.0)
    assert result["ci95_low"] == pytest.approx(1.0)
    assert result["ci95_high"] == pytest.approx(1.0)
