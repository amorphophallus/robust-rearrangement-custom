import math

import numpy as np

from src.eval.vlm_point_metrics import (
    build_vlm_point_error_summary,
    make_point_error_record,
    merge_vlm_point_error_summaries,
)


def _record(
    error_target,
    *,
    skill="pick",
    vlm_skill=None,
    step=0,
    query_step=0,
):
    return make_point_error_record(
        step_idx=step,
        oracle_skill=skill,
        vlm_skill=skill if vlm_skill is None else vlm_skill,
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


def _camera_info():
    return {
        "intrinsics": np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        "sim_local_to_camera": np.eye(4),
        "camera_to_sim_local": np.eye(4),
        "image_size": np.array([100, 100]),
    }


def test_same_depth_unprojection_is_reported_as_lateral_mm():
    record = make_point_error_record(
        step_idx=0,
        oracle_skill="push",
        oracle_point=[50.0, 50.0],
        vlm_point=[60.0, 50.0],
        query_step=0,
        oracle_point_3d=[0.0, 0.0, 1.0],
        camera_info=_camera_info(),
        noise_seed=3,
    )

    assert math.isclose(record["same_depth_error_mm"], 100.0)
    n0_samples = record["projected_noise_residuals_px"]["n0"]
    assert len(n0_samples) == 200
    assert n0_samples[0] == [0.0, 0.0]
    assert record["is_fresh_query"] is True


def test_step_distribution_matches_zero_residual_to_projected_n0():
    records = [
        make_point_error_record(
            step_idx=step,
            oracle_skill="push",
            oracle_point=[50.0, 50.0],
            vlm_point=[50.0, 50.0],
            query_step=step,
            oracle_point_3d=[0.0, 0.0, 1.0],
            camera_info=_camera_info(),
            noise_seed=step,
        )
        for step in range(16)
    ]
    summary = build_vlm_point_error_summary([records], [True])
    distribution = summary["all"]["step_distribution"]

    assert distribution["vlm"]["count"] == 16
    assert distribution["projection_reference"]["reference_pair_count"] == 16
    assert distribution["projection_reference"]["monte_carlo_samples_per_pair"] == 200
    assert distribution["noise_levels"]["n1"]["projected"]["count"] == 3200
    assert distribution["closest_level_sliced_wasserstein"] == "n0"
    assert distribution["closest_level_radial_wasserstein"] == "n0"
    assert distribution["closest_level_rmse"] == "n0"
    assert distribution["magnitude_equivalent_bracket"] == "n0"
    assert distribution["magnitude_equivalent_std_mm"] == 0.0


def test_cached_steps_are_included_in_step_distribution():
    fresh = make_point_error_record(
        step_idx=0,
        oracle_skill="push",
        oracle_point=[50.0, 50.0],
        vlm_point=[55.0, 50.0],
        query_step=0,
        oracle_point_3d=[0.0, 0.0, 1.0],
        camera_info=_camera_info(),
    )
    cached = make_point_error_record(
        step_idx=1,
        oracle_skill="push",
        oracle_point=[50.0, 50.0],
        vlm_point=[55.0, 50.0],
        query_step=0,
        oracle_point_3d=[0.0, 0.0, 1.0],
        camera_info=_camera_info(),
    )
    summary = build_vlm_point_error_summary([[fresh, cached]], [False])

    assert summary["all"]["overall"]["count_valid"] == 2
    assert summary["all"]["step_distribution"]["vlm"]["count"] == 2
    assert (
        summary["all"]["step_distribution"]["noise_levels"]["n1"][
            "projected"
        ]["count"]
        == 400
    )
    assert summary["all"]["fresh_queries"]["overall"]["count_valid"] == 1
    assert summary["all"]["fresh_queries"]["distribution"]["vlm"]["count"] == 1


def _spatial_record(*, target, prediction, step, query_step=None, oracle="pick", predicted="pick"):
    return make_point_error_record(
        step_idx=step,
        oracle_skill=oracle,
        vlm_skill=predicted,
        oracle_point=target,
        vlm_point=prediction,
        query_step=step if query_step is None else query_step,
    )


def test_spatial_quality_metrics_cover_perfect_mean_and_negative_r2():
    targets = ([0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0])
    perfect = [
        _spatial_record(target=target, prediction=target, step=index)
        for index, target in enumerate(targets)
    ]
    perfect_stats = build_vlm_point_error_summary([perfect], [True])["all"]["overall"]
    assert perfect_stats["point_r2"] == 1.0
    assert perfect_stats["spread_ratio"] == 1.0
    assert perfect_stats["corr_u"] == 1.0
    assert perfect_stats["corr_v"] == 1.0
    assert perfect_stats["p50_error_px"] == 0.0

    mean_prediction = [1.0, 1.0]
    mean_records = [
        _spatial_record(target=target, prediction=mean_prediction, step=index)
        for index, target in enumerate(targets)
    ]
    mean_stats = build_vlm_point_error_summary([mean_records], [False])["all"]["overall"]
    assert math.isclose(mean_stats["point_r2"], 0.0, abs_tol=1e-12)
    assert mean_stats["spread_ratio"] == 0.0
    assert mean_stats["corr_u"] is None
    assert mean_stats["corr_v"] is None

    wrong_records = [
        _spatial_record(target=target, prediction=[10.0, 10.0], step=index)
        for index, target in enumerate(targets)
    ]
    wrong_stats = build_vlm_point_error_summary([wrong_records], [False])["all"]["overall"]
    assert wrong_stats["point_r2"] < 0.0


def test_spatial_metrics_handle_zero_target_variance():
    records = [
        _spatial_record(target=[2.0, 3.0], prediction=[2.0 + step, 3.0], step=step)
        for step in range(3)
    ]
    stats = build_vlm_point_error_summary([records], [True])["all"]["overall"]
    assert stats["point_r2"] is None
    assert stats["spread_ratio"] is None


def test_skill_accuracy_confusion_and_tail_counts_are_reported():
    records = [
        _spatial_record(target=[0.0, 0.0], prediction=[0.0, 0.0], step=0),
        _spatial_record(
            target=[0.0, 0.0],
            prediction=[50.0, 0.0],
            step=1,
            oracle="pick",
            predicted="place",
        ),
        _spatial_record(
            target=[0.0, 0.0],
            prediction=[80.0, 0.0],
            step=2,
            oracle="push",
            predicted="pick",
        ),
    ]
    stats = build_vlm_point_error_summary([records], [False])["all"]["overall"]
    assert stats["skill_accuracy"] == 1.0 / 3.0
    assert stats["skill_confusion"]["pick"]["pick"] == 1
    assert stats["skill_confusion"]["pick"]["place"] == 1
    assert stats["skill_confusion"]["push"]["pick"] == 1
    assert stats["tail_count_gt_40px"] == 2
    assert stats["tail_count_gt_70px"] == 1
    assert stats["p50_error_px"] == 50.0


def test_merge_preserves_r2_spread_quantiles_and_skill_metrics():
    records = [
        _spatial_record(target=[0.0, 0.0], prediction=[0.0, 0.0], step=0),
        _spatial_record(
            target=[2.0, 0.0],
            prediction=[1.0, 0.0],
            step=1,
            predicted="place",
        ),
        _spatial_record(target=[4.0, 0.0], prediction=[4.0, 0.0], step=2),
    ]
    direct = build_vlm_point_error_summary([records], [True])
    first = build_vlm_point_error_summary([[records[0]]], [True])
    second = build_vlm_point_error_summary([[records[1], records[2]]], [True])
    merged = merge_vlm_point_error_summaries([first, second])
    for key in (
        "point_r2",
        "spread_ratio",
        "p50_error_px",
        "p90_error_px",
        "skill_accuracy",
    ):
        assert math.isclose(
            merged["all"]["overall"][key],
            direct["all"]["overall"][key],
        )
    assert (
        merged["all"]["overall"]["skill_confusion"]
        == direct["all"]["overall"]["skill_confusion"]
    )


def test_merge_preserves_step_distribution_samples():
    first = build_vlm_point_error_summary(
        [[
            make_point_error_record(
                step_idx=0,
                oracle_skill="push",
                oracle_point=[50.0, 50.0],
                vlm_point=[50.0, 50.0],
                query_step=0,
                oracle_point_3d=[0.0, 0.0, 1.0],
                camera_info=_camera_info(),
            )
        ]],
        [True],
    )
    merged = merge_vlm_point_error_summaries([first, first])

    assert merged["all"]["step_distribution"]["vlm"]["count"] == 2
    assert (
        merged["all"]["step_distribution"]["noise_levels"]["n1"][
            "projected"
        ]["count"]
        == 400
    )
    assert (
        merged["all"]["step_distribution"][
            "closest_level_sliced_wasserstein"
        ]
        == "n0"
    )


def test_magnitude_equivalence_can_report_above_n4():
    records = [
        make_point_error_record(
            step_idx=step,
            oracle_skill="push",
            oracle_point=[50.0, 50.0],
            vlm_point=[90.0, 50.0],
            query_step=step,
            oracle_point_3d=[0.0, 0.0, 1.0],
            camera_info=_camera_info(),
            noise_seed=step,
        )
        for step in range(128)
    ]
    summary = build_vlm_point_error_summary([records], [False])
    distribution = summary["all"]["step_distribution"]

    assert distribution["magnitude_equivalent_bracket"] == ">n4"
    assert distribution["magnitude_equivalent_std_mm"] > 24.0
    assert "closest_level_centered_sliced_wasserstein" in distribution
