import json

from scripts.audit_clean_train_noise_eval import (
    _expected_cover_new_keys,
    _limit_reported_issues,
    audit_vlm_cover_108,
)
from scripts.run_clean_train_noise_eval import (
    CONDITIONS,
    POINT_NOISE_LEVELS,
    ReplicateConfig,
    _validate_summary,
)


def test_limit_reported_issues_preserves_count_and_completion_state():
    payload = {
        "issues": ["first", "second", "third"],
        "complete": False,
    }

    limited = _limit_reported_issues(payload, 2)

    assert limited["issues"] == ["first", "second"]
    assert limited["issue_count"] == 3
    assert limited["issues_truncated"] == 1
    assert limited["complete"] is False
    assert payload["issues"] == ["first", "second", "third"]


def test_cover_expected_key_count_matches_111_invocations():
    assert len(_expected_cover_new_keys()) == 111


def test_cover_audit_fails_closed_on_duplicate_full_key(tmp_path):
    manifest = tmp_path / "new.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    row = {
        "condition_id": "gp",
        "noise_id": "n5",
        "replicate_id": 0,
        "simulator_seed": 0,
        "annotation_seed": 0,
        "status": "ok",
    }
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    legacy.write_text("")

    payload, returncode = audit_vlm_cover_108(
        manifest_path=manifest,
        legacy_manifest_path=legacy,
        require_complete=False,
    )

    assert returncode == 1
    assert any("duplicate manifest key" in issue for issue in payload["issues"])


def test_cover_audit_fails_closed_on_missing_replicates(tmp_path):
    manifest = tmp_path / "new.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    manifest.write_text("")
    legacy.write_text("")

    payload, returncode = audit_vlm_cover_108(
        manifest_path=manifest,
        legacy_manifest_path=legacy,
        require_complete=True,
    )

    assert returncode == 2
    assert payload["completed_new_invocations"] == 0
    assert len(payload["missing"]) == 141


def _valid_cover_summary(condition, noise):
    tracking = {
        "metric_type": "position",
        "overall": {"count": 1, "mean_pos_m": 0.01},
        "episode_count": 36,
        "incomplete_episode_count": 0,
        "complete": True,
        "target_source": "scripted_displayed_annotation",
    }
    noise_stats = {
        "phase_count": 1,
        "position_norm_m": {"count": 1},
        "rotation_geodesic_deg": {"count": 0},
        "workspace_valid_rate": 1.0,
        "front_projection_visible_rate": 1.0,
        "invalid_nonfinite_rate": 0.0,
        "phase_samples": [],
    }
    task_payload = {
        "n_rollouts": 36,
        "n_envs": 3,
        "n_saved_rollouts": 0,
        "eval_randomness": "low",
        "simulator_seed": 0,
        "tracking_error": tracking,
        "annotation_noise_stats": noise_stats,
    }
    return {
        "n_rollouts": 108,
        "n_envs": 3,
        "checkpoint_name": condition.checkpoint.stem,
        "task_group": "one_leg+round_table+lamp",
        "eval_randomness": "low",
        "observation_space": "image",
        "action_type": "pos",
        "annotation_source": "scripted",
        "simulator_seed": 0,
        "training_config": {
            "data": {
                "annotation_noise_pos_std_m": 0.0,
                "annotation_noise_ori_std_deg": 0.0,
            }
        },
        "annotation_noise_config": {
            "pos_std_m": noise.pos_std_m,
            "ori_std_deg": noise.ori_std_deg,
            "enabled": True,
            "apply_to": condition.apply_to,
            "mode": "gaussian_clip_2sigma",
            "seed": 0,
        },
        "eval_annotation_config": {
            "annotate_skill": True,
            "guidance_point_on_image": True,
            "guidance_point_colored": False,
            "grasp_part_annotate": False,
            "grasp_annotation_colored": False,
        },
        "per_task": {
            task: json.loads(json.dumps(task_payload))
            for task in ("one_leg", "round_table", "lamp")
        },
    }


def test_cover_summary_rejects_wrong_rollout_and_tracking_sample_counts(tmp_path):
    condition = CONDITIONS[0]
    noise = POINT_NOISE_LEVELS[5]
    summary = _valid_cover_summary(condition, noise)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    kwargs = {
        "summary_path": path,
        "condition": condition,
        "noise": noise,
        "task_group": "one_leg+round_table+lamp",
        "n_envs": 3,
        "n_rollouts": 36,
        "randomness": "low",
        "replicate": ReplicateConfig(0, 0, 0),
        "expected_saved_rollouts": 0,
        "require_noise_stats": True,
    }
    assert _validate_summary(**kwargs) == []

    summary["per_task"]["lamp"]["n_rollouts"] = 35
    summary["per_task"]["round_table"]["tracking_error"]["episode_count"] = 35
    path.write_text(json.dumps(summary))
    errors = _validate_summary(**kwargs)

    assert any("lamp.n_rollouts=35 expected=36" in error for error in errors)
    assert any(
        "round_table.tracking_error.episode_count=35 expected=36" in error
        for error in errors
    )
