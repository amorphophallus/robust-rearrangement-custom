import json

import matplotlib.pyplot as plt

from scripts.generate_vlm_cover_108_report import (
    _add_unique_figure_legend,
    _categorical_endpoint_offsets,
    generate_vlm_cover_108_report,
)
from scripts.run_clean_train_noise_eval import CONDITIONS, _vlm_cover_schedule


def _summary(seed, *, legacy=False):
    per_task = {}
    for task in ("one_leg", "round_table", "lamp"):
        per_task[task] = {
            "n_success": 18 + seed,
            "n_rollouts": 36,
            "skill_state_counts": {"part-part-pick": 36},
            "skill_completion_counts": {"part-part-pick": 30},
            "tracking_error": {
                "complete": not legacy,
                "metric_type": "position",
                "overall": {"count": 36, "mean_pos_m": 0.01},
            },
            "annotation_noise_stats": ({
                "phase_samples": [{
                    "realized_pos_norm_m": 0.01,
                    "realized_ori_geodesic_deg": 0.0,
                    "apply_pose_pos": True,
                    "apply_ori": False,
                    "target_finite": True,
                    "workspace_valid": True,
                    "front_projection_visible": True,
                }]
            } if not legacy else None),
        }
    return {"per_task": per_task}


def test_full_report_pools_success_but_excludes_legacy_tracking(tmp_path):
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    new_rows = []
    legacy_rows = []

    for condition in CONDITIONS:
        for noise_id in ("n0", "n1", "n2", "n3", "n4", "shuffle"):
            path = summary_dir / f"legacy-{condition.condition_id}-{noise_id}.json"
            path.write_text(json.dumps(_summary(0, legacy=True)))
            legacy_rows.append({
                "condition_id": condition.condition_id,
                "noise_id": noise_id,
                "status": "ok",
                "summary_json": str(path),
            })

    for idx, (condition, noise, replicate) in enumerate(_vlm_cover_schedule(CONDITIONS)):
        path = summary_dir / f"new-{idx}.json"
        path.write_text(json.dumps(_summary(replicate.replicate_id)))
        new_rows.append({
            "condition_id": condition.condition_id,
            "noise_id": noise.noise_id,
            "replicate_id": replicate.replicate_id,
            "simulator_seed": replicate.simulator_seed,
            "annotation_seed": replicate.annotation_seed,
            "status": "ok",
            "summary_json": str(path),
        })

    new_manifest = tmp_path / "new.jsonl"
    legacy_manifest = tmp_path / "legacy.jsonl"
    new_manifest.write_text("".join(json.dumps(row) + "\n" for row in new_rows))
    legacy_manifest.write_text("".join(json.dumps(row) + "\n" for row in legacy_rows))
    report = tmp_path / "reports" / "report.md"
    data_dir = tmp_path / "reports" / "data"

    generate_vlm_cover_108_report(
        manifest_path=new_manifest,
        legacy_manifest_path=legacy_manifest,
        report_path=report,
        figures_dir=tmp_path / "reports" / "figures",
        data_dir=data_dir,
    )

    pooled = (data_dir / "success_tracking_pooled.csv").read_text()
    assert ",108," in pooled
    assert ",72," in pooled
    validation = (data_dir / "table_validation.csv").read_text()
    assert "pooled_matches_replicates" in validation
    assert ",ok," in validation
    index = json.loads((data_dir / "data_index.json").read_text())
    assert index["pipeline"] == "json -> tables -> figures"
    assert index["figures_read_only_sources"] == [
        "success_tracking_pooled.csv",
        "three_task_pooled.csv",
        "vlm_sigma_by_task.csv",
        "vlm_sigma_3task_pooled.csv",
        "vlm_orientation_tracking_equivalent.csv",
        "skill_type_replicate_and_pooled.csv",
    ]
    assert "three_task_pooled" in index["tables"]
    assert "noise_schedule" in index["tables"]
    assert "vlm_sigma_3task_pooled" in index["tables"]
    assert "vlm_orientation_tracking_equivalent" in index["tables"]
    assert "success_3task_pooled_png" in index["figures"]
    assert "tracking_position_3task_pooled_png" in index["figures"]
    assert "vlm_sigma_by_task" in index["tables"]
    assert "skill_type_replicate_and_pooled" in index["tables"]
    assert "tracking_error_png" in index["figures"]
    for figure_key in (
        "skill_success_rate_png",
        "tracking_position_png",
        "tracking_orientation_png",
        "tracking_total_png",
    ):
        assert figure_key in index["figures"]
    text = report.read_text()
    assert "tracking_n=72" in text
    assert "所有成功率都使用每 task 108 rollout" in text
    assert "成功率表中不存在 72-rollout 结果" in text
    assert "tracking_n/task" not in text
    assert text.index("## 1. 结果图与主要结论") < text.index(
        "## 2. 完整 pooled success"
    )
    assert "当前数据不支持“噪声使 grasp 变得更好”" in text
    assert "n0–n4 会集中在低噪声段，不再等距排列" in text
    assert "只读取已校验的 `success_tracking_pooled.csv`，不直接 query JSON" in text
    assert "n7→Shuffle` categorical endpoint" in text
    assert "每个 task 画两条上游 position-equivalent VLM σ 线" in text
    assert "pooled tracking error（position / orientation / total）" in text
    assert "三 task 合并的 pooled summary" in text
    assert "pooled 图只保留三条三-task 纵向线" in text
    assert "图下方的 orientation 对齐采用一个行为等效尺度" in text
    assert text.index("### 1.3 三 task 合并的 pooled summary") < text.index(
        "### 1.4 主要结论"
    )
    assert text.index("### 1.4 主要结论") < text.index(
        "### 1.7 skill-level success rate"
    )
    assert "skill-level success rate 与 tracking error（5 skills × 3 tasks）" in text
    assert "Skill-level position tracking error" in text
    assert "skill_type_replicate_and_pooled.csv" in text
    assert "每个 control step 贡献一个 VLM–GT 点对" in text
    assert "Point 使用 `3×36=108` 条 source trajectory" in text
    assert "`pooled` 表示把同一 VLM family 在同一 task" in text
    assert "不单独画 endpoint 图" in text
    assert "success_endpoints" not in text
    assert "n0–n7 在图中等距排列" not in text
    assert "竖向 whisker 是三个 replicate success 的 min–max" not in text
    assert "这些都是误差尺度参考，不是额外成功率数据点" in text
    assert "Equivalent position σ (mm/axis)" in text
    assert "n0–n7 的 position 与 orientation 扰动是绑定的" in text
    assert "行为等效 orientation scale" in text
    assert "不冒充 raw VLM orientation residual" in text
    assert "r180" in text
    assert "raw rotation error" in text

    success_pooled_svg = (
        tmp_path / "reports" / "figures" / "vlm_cover_108_success_3task_pooled.svg"
    ).read_text()
    assert "P·3task" in success_pooled_svg
    assert "G·3task" in success_pooled_svg
    assert "G-ori·3task" in success_pooled_svg
    assert "P·one_leg" not in success_pooled_svg
    assert "G·one_leg" not in success_pooled_svg
    assert "P·round_table" not in success_pooled_svg
    assert "G·round_table" not in success_pooled_svg
    assert "P·lamp" not in success_pooled_svg
    assert "G·lamp" not in success_pooled_svg


def test_figure_legend_deduplicates_subplot_labels():
    fig, axes = plt.subplots(1, 2)
    axes[0].plot([0, 1], label="shared")
    axes[1].plot([0, 1], label="shared")
    axes[1].plot([1, 0], label="unique")

    _add_unique_figure_legend(fig, axes)

    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "shared",
        "unique",
    ]
    plt.close(fig)


def test_categorical_offsets_only_complete_endpoint_overlaps():
    rows = [
        {"condition_id": "gp", "noise_id": "n7", "success_rate": 0.5},
        {"condition_id": "gp", "noise_id": "shuffle", "success_rate": 0.4},
        {"condition_id": "colored_gp", "noise_id": "n7", "success_rate": 0.5},
        {"condition_id": "colored_gp", "noise_id": "shuffle", "success_rate": 0.4},
        {"condition_id": "gp_skill", "noise_id": "n7", "success_rate": 0.5},
        {"condition_id": "gp_skill", "noise_id": "shuffle", "success_rate": 0.3},
    ]
    offsets = _categorical_endpoint_offsets(rows, "success_rate")
    assert offsets["gp"] < offsets["colored_gp"]
    assert offsets["gp_skill"] == 0.0
