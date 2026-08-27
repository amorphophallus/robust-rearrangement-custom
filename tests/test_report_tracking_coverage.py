import json

from scripts.generate_annotation_noise_report import (
    _apply_saved_tracking,
    _categorical_endpoint_offsets,
    _invalidate_unfiltered_tracking,
    _result_cell,
    _shuffle_summary_rows,
    _tracking_coverage_complete,
    _tracking_workspace_exclusion_rows,
)


def test_categorical_offset_is_used_only_for_fully_overlapping_segments():
    rows = [
        {"condition": "rgbd+GP", "noise_id": "n4", "value": 0.4},
        {"condition": "rgbd+GP", "noise_id": "shuffle", "value": 0.3},
        {"condition": "rgbd+colored GP", "noise_id": "n4", "value": 0.4},
        {"condition": "rgbd+colored GP", "noise_id": "shuffle", "value": 0.2},
        {"condition": "rgbd+GP+skill", "noise_id": "n4", "value": 0.4},
        {"condition": "rgbd+GP+skill", "noise_id": "shuffle", "value": 0.3},
    ]

    offsets = _categorical_endpoint_offsets(rows, "value")

    assert offsets["rgbd+GP"] < 0.0
    assert offsets["rgbd+GP+skill"] > 0.0
    assert offsets["rgbd+colored GP"] == 0.0


def test_saved_tracking_replaces_legacy_task_values(tmp_path):
    saved_path = tmp_path / "saved.json"
    saved_path.write_text(
        json.dumps(
            {
                "rollouts_per_task_setting": 8,
                "groups": [
                    {
                        "condition_id": "gp",
                        "noise_id": "n0",
                        "task": "one_leg",
                        "rollout_count": 8,
                        "tracking_error": {
                            "overall": {"count": 5, "mean_pos_m": 0.12},
                            "by_skill": {},
                        },
                        "by_skill_type": {},
                    }
                ],
            }
        )
    )
    task_rows = [
        {
            "condition_id": "gp",
            "noise_id": "n0",
            "task": "one_leg",
            "family": "point",
            "track_pos_cm": 999.0,
            "tracking_complete": False,
        }
    ]

    _apply_saved_tracking(
        overall_rows=[],
        task_rows=task_rows,
        per_step_rows=[],
        skill_type_rows=[],
        saved_tracking_path=saved_path,
    )

    assert task_rows[0]["track_pos_cm"] == 12.0
    assert task_rows[0]["tracking_count"] == 5
    assert task_rows[0]["tracking_source"] == "saved_rollouts_8"
    assert task_rows[0]["tracking_rollouts_per_task"] == 8


def test_tracking_coverage_requires_explicit_complete_episode_counts():
    legacy = {
        "n_rollouts": 36,
        "tracking_error": {"overall": {"count": 128}},
    }

    assert not _tracking_coverage_complete(legacy)


def test_tracking_coverage_accepts_all_complete_episodes():
    complete = {
        "n_rollouts": 36,
        "tracking_error": {
            "episode_count": 36,
            "expected_episode_count": 36,
            "incomplete_episode_count": 0,
            "complete": True,
            "workspace_filter": {
                "coordinate_frame": "robot_base_m",
                "final_segment_count": 100,
                "included_segment_count": 99,
                "excluded_outside_workspace_count": 1,
                "missing_or_invalid_target_count": 0,
            },
        },
    }

    assert _tracking_coverage_complete(complete)


def test_workspace_exclusions_are_aggregated_by_condition_and_task():
    payload = {
        "groups": [
            {
                "condition": "rgbd+GP",
                "condition_id": "gp",
                "task": "one_leg",
                "workspace_filter": {
                    "included_segment_count": 9,
                    "excluded_outside_workspace_count": 1,
                    "missing_or_invalid_target_count": 2,
                },
            },
            {
                "condition": "rgbd+GP",
                "condition_id": "gp",
                "task": "one_leg",
                "workspace_filter": {
                    "included_segment_count": 10,
                    "excluded_outside_workspace_count": 0,
                    "missing_or_invalid_target_count": 0,
                },
            },
        ]
    }

    rows = _tracking_workspace_exclusion_rows(payload)

    gp = next(row for row in rows if row["condition"] == "rgbd+GP")
    assert gp["one_leg"] == "1/20 (5.00%)"
    assert gp["invalid"] == "2"


def test_result_cell_withholds_legacy_partial_tracking_values():
    cell = _result_cell(
        {
            "family": "point",
            "success_rate": 0.5,
            "n_success": 18,
            "n_rollouts": 36,
            "track_pos_cm": 123.0,
            "tracking_count": 10,
            "tracking_complete": False,
        }
    )

    assert "SR 50.0% (18/36)" in cell
    assert "legacy partial" in cell
    assert "123.0" not in cell


def test_unfiltered_tracking_is_retained_only_as_legacy_plot_data():
    row = {
        "track_pos_cm": 12.3,
        "track_ori_deg": 45.6,
        "track_total": 21.4,
        "tracking_count": 17,
        "tracking_complete": True,
    }

    _invalidate_unfiltered_tracking(row)

    assert row["track_pos_cm"] == 12.3
    assert row["track_ori_deg"] == 45.6
    assert row["track_total"] == 21.4
    assert row["tracking_count"] == 17
    assert row["tracking_complete"] is False
    assert row["tracking_source"] == "full_evaluator_36_pre_workspace_filter"


def test_shuffle_summary_publishes_only_valid_tracking():
    common = {
        "condition": "rgbd+GP",
        "family": "point",
        "track_ori_deg": None,
        "track_total": None,
    }
    rows = _shuffle_summary_rows(
        [
            {
                **common,
                "noise_id": "n0",
                "success_rate": 0.6,
                "tracking_complete": False,
                "track_pos_cm": 999.0,
                "skill_state_count": 1,
            },
            {
                **common,
                "noise_id": "shuffle",
                "success_rate": 0.4,
                "tracking_complete": True,
                "track_pos_cm": 12.34,
                "skill_state_count": 123,
            },
        ]
    )

    assert rows == [
        {
            "condition": "rgbd+GP",
            "n0_success": "60.0%",
            "shuffle_success": "40.0%",
            "success_delta": "-20.0 pp",
            "track_pos_cm": "12.34",
            "track_ori_deg": "N/A",
            "track_total": "N/A",
            "tracking_count": "123",
        }
    ]
