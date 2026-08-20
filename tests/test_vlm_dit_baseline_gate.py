import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_vlm_dit_eval import (
    CONDITIONS,
    EXPECTED_VLM_REVISION,
    HISTORICAL_CLEAN_SUCCESS,
    _smoke_gate,
    _formal_gate_record,
    _normalized_smoke_formal_command,
    _require_formal_smoke_gate,
    build_auto_eval_command,
    validate_expanded_command,
)


def _args(tmp_path: Path):
    return SimpleNamespace(
        auto_eval=Path("/data/hy/gpu-snatcher/auto_eval.sh"),
        n_rollouts=3,
        n_envs=3,
        stage="smoke",
        gpu=0,
        vlm_base_url="http://10.71.106.240:8000",
    )


def test_builder_uses_auto_eval_and_never_direct_evaluator(tmp_path):
    args = _args(tmp_path)
    command = build_auto_eval_command(
        args=args,
        condition="rgbd_gp",
        task="one_leg",
        summary_path=tmp_path / "summary.json",
        rollout_suffix="smoke/rgbd_gp/one_leg",
        print_command=True,
    )

    assert command[0] == "/data/hy/gpu-snatcher/auto_eval.sh"
    assert "src.eval.evaluate_model" not in command
    assert "--print-command" in command
    assert "--no-annotate-skill" in command
    assert command[command.index("--max-saved-rollouts") + 1] == "10"
    assert "--compress-pickles" not in command
    assert "--output-only-video" not in command


def test_preview_builder_runs_exactly_one_env_and_adds_review_overlays(tmp_path):
    args = _args(tmp_path)
    args.stage = "preview"
    args.n_envs = 1
    args.n_rollouts = 1
    command = build_auto_eval_command(
        args=args,
        condition="rgbd_gp",
        task="lamp",
        summary_path=tmp_path / "lamp.json",
        rollout_suffix="preview/rgbd_gp/lamp",
        print_command=True,
    )

    assert command[command.index("--n-envs") + 1] == "1"
    assert command[command.index("--n-rollouts") + 1] == "1"
    assert "--guidance-point-on-image" in command
    assert "--annotate-skill" in command
    assert "--skill-on-image" in command
    assert "--no-annotate-skill" not in command


def test_task_summary_collects_stats_independent_of_skill_overlay():
    source = Path("src/eval/evaluate_model.py").read_text()
    assert 'collect_skill_stats=bool(' in source
    assert 'args.task_summary_out' in source
    assert 'collect_skill_stats=resolved_eval_annotations["annotate_skill"]' not in source


def test_expanded_command_requires_depth(tmp_path):
    summary = tmp_path / "summary.json"
    command = [
        "python", "-m", "src.eval.evaluate_model",
        "--n-envs", "3", "--n-rollouts", "3", "-f", "one_leg",
        "--if-exists", "append", "--max-rollout-steps", "1000",
        "--max-saved-rollouts", "10",
        "--action-type", "pos", "--observation-space", "image",
        "--randomness", "low", "--save-rollouts", "--save-failures",
        "--annotation-source", "vlm", "--tracking-metric-type", "pose",
        "--vlm-base-url", "http://10.71.106.240:8000",
        "--vlm-timeout-seconds", "30", "--vlm-query-interval", "0",
        "--vlm-noise-projection-samples", "200", "--task-summary-out", str(summary),
        "--guidance-point-on-image", "--rollout-suffix-model-name",
        "smoke/rgbd_gp/one_leg", "--wt-path", str(CONDITIONS["rgbd_gp"]["checkpoint"]),
    ]

    with pytest.raises(RuntimeError, match="missing --save-depth-image"):
        validate_expanded_command(
            command,
            condition="rgbd_gp",
            task="one_leg",
            n_rollouts=3,
            summary_path=summary,
            rollout_suffix="smoke/rgbd_gp/one_leg",
            vlm_base_url="http://10.71.106.240:8000",
        )


def test_smoke_gate_requires_two_of_three_for_rgbd_gp_one_leg(tmp_path):
    rows = []
    for condition in CONDITIONS:
        for task in ("one_leg", "round_table", "lamp"):
            path = tmp_path / f"{condition}__{task}.json"
            path.write_text(json.dumps({"n_success": 1}))
            rows.append(
                {
                    "condition": condition,
                    "task": task,
                    "summary_path": str(path),
                    "status": "complete",
                    "return_code": 0,
                    "summary_error": None,
                    "depth_contract_logged": True,
                    "vlm_readiness": {"model_revision": EXPECTED_VLM_REVISION},
                }
            )

    gate = _smoke_gate({"runs": rows})
    assert gate["status"] == "failed"
    assert "rgbd_gp/one_leg success is below required 2/3" in gate["failures"]


def test_smoke_gate_rejects_zero_for_historically_above_half_cell(tmp_path):
    rows = []
    for condition in CONDITIONS:
        for task in ("one_leg", "round_table", "lamp"):
            path = tmp_path / f"{condition}__{task}.json"
            n_success = 2
            if (condition, task) == ("rgbd_gp_skill", "lamp"):
                assert HISTORICAL_CLEAN_SUCCESS[(condition, task)] == (20, 36)
                n_success = 0
            path.write_text(json.dumps({"n_success": n_success}))
            rows.append(
                {
                    "condition": condition,
                    "task": task,
                    "summary_path": str(path),
                    "status": "complete",
                    "return_code": 0,
                    "summary_error": None,
                    "depth_contract_logged": True,
                    "vlm_readiness": {"model_revision": EXPECTED_VLM_REVISION},
                }
            )

    gate = _smoke_gate({"runs": rows})
    assert gate["status"] == "failed"
    assert any(
        "rgbd_gp_skill/lamp is 0/3 despite historical clean success 20/36" in failure
        for failure in gate["failures"]
    )


def test_formal_command_comparison_allows_only_campaign_values():
    smoke = ["python", "--n-rollouts", "3", "--task-summary-out", "smoke.json", "--rollout-suffix-model-name", "smoke/name", "--randomness", "low"]
    formal = ["python", "--n-rollouts", "36", "--task-summary-out", "formal.json", "--rollout-suffix-model-name", "formal/name", "--randomness", "low"]
    assert _normalized_smoke_formal_command(smoke) == _normalized_smoke_formal_command(formal)
    formal[-1] = "med"
    assert _normalized_smoke_formal_command(smoke) != _normalized_smoke_formal_command(formal)


def test_formal_gate_requires_manual_visual_review(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"smoke_gate": {"status": "passed"}}))
    with pytest.raises(RuntimeError, match="manual_review.status=passed"):
        _require_formal_smoke_gate(path)


def test_formal_gate_records_explicit_user_approved_bypass():
    args = SimpleNamespace(
        stage="formal",
        allow_formal_without_smoke=True,
        formal_approval_note="User approved direct formal after three-task original_sft preview.",
        smoke_manifest=None,
    )

    assert _formal_gate_record(args) == {
        "mode": "explicit_user_approved_bypass",
        "smoke_manifest": None,
        "approval_note": (
            "User approved direct formal after three-task original_sft preview."
        ),
    }
