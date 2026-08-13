import math

from src.eval.vlm_point_metrics import (
    build_vlm_point_error_summary,
    make_point_error_record,
    merge_vlm_point_error_summaries,
)


def _record(error_target, *, skill="pick", step=0, query_step=0):
    return make_point_error_record(
        step_idx=step,
        oracle_skill=skill,
        oracle_point=error_target,
        vlm_point=[0.0, 0.0],
        query_step=query_step,
    )


def test_point_error_uses_pixel_euclidean_distance_and_gt_skill():
    record = _record([3.0, 4.0], skill="insert", step=7, query_step=0)
    summary = build_vlm_point_error_summary([[record]], [True])

    assert record["error_px"] == 5.0
    assert record["cache_age_steps"] == 7
    assert summary["all"]["overall"]["mean_error_px"] == 5.0
    assert summary["all"]["by_skill"]["insert"]["mean_error_px"] == 5.0
    assert summary["all"]["by_skill"]["pick"]["count_valid"] == 0


def test_point_error_reports_invalid_gt_as_coverage_not_zero_error():
    valid = _record([3.0, 4.0])
    invalid = _record(None, step=1)
    summary = build_vlm_point_error_summary([[valid, invalid]], [False])
    overall = summary["all"]["overall"]

    assert overall["mean_error_px"] == 5.0
    assert overall["count_valid"] == 1
    assert overall["count_invalid_gt"] == 1
    assert overall["coverage"] == 0.5
    assert summary["success_only"]["overall"]["mean_error_px"] is None


def test_point_error_splits_success_and_failure_rollouts():
    summary = build_vlm_point_error_summary(
        [[_record([3.0, 4.0])], [_record([0.0, 12.0])]],
        [True, False],
    )

    assert summary["all"]["overall"]["mean_error_px"] == 8.5
    assert summary["success_only"]["overall"]["mean_error_px"] == 5.0
    assert summary["failure_only"]["overall"]["mean_error_px"] == 12.0


def test_point_error_merge_is_weighted_by_steps_not_task_means():
    first = build_vlm_point_error_summary([[_record([3.0, 4.0])]], [True])
    second = build_vlm_point_error_summary(
        [[_record([0.0, 10.0]), _record([0.0, 10.0], step=1)]],
        [True],
    )
    merged = merge_vlm_point_error_summaries([first, second])

    assert merged["all"]["overall"]["mean_error_px"] == 25.0 / 3.0
    assert math.isclose(
        merged["all"]["overall"]["rmse_px"],
        math.sqrt(225.0 / 3.0),
    )
