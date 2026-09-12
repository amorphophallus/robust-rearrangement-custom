import json
from pathlib import Path

from scripts.run_clean_train_noise_eval import (
    CONDITIONS,
    FIXED_R180,
    GRASP_NOISE_LEVELS,
    ReplicateConfig,
    _build_command,
    _manifest_key,
    _upsert_manifest,
    _vlm_cover_schedule,
)


def test_vlm_cover_schedule_is_fixed_111_new_invocations():
    schedule = _vlm_cover_schedule(CONDITIONS)

    assert len(schedule) == 111
    assert len(
        {
            (
                condition.condition_id,
                noise.noise_id,
                replicate.replicate_id,
                replicate.simulator_seed,
                replicate.annotation_seed,
            )
            for condition, noise, replicate in schedule
        }
    ) == 111
    assert all(
        not (replicate.replicate_id == 0 and noise.noise_id in {"n0", "n1", "n2", "n3", "n4", "shuffle"})
        for _, noise, replicate in schedule
    )
    assert schedule[0][1].noise_id == "n5"
    assert schedule[-1][1].noise_id == "shuffle"


def test_build_command_forwards_both_seeds_and_unique_suffix():
    condition = CONDITIONS[3]
    replicate = ReplicateConfig(2, 2, 2)
    command = _build_command(
        auto_eval_path=Path("/tmp/auto_eval.sh"),
        task_group="one_leg+round_table+lamp",
        checkpoint=condition.checkpoint,
        flags=condition.flags,
        n_envs=3,
        n_rollouts=36,
        randomness="low",
        condition=condition,
        noise=GRASP_NOISE_LEVELS[-1],
        apply_to=condition.apply_to,
        save_rollouts_count=0,
        replicate=replicate,
    )

    assert command[command.index("--seed") + 1] == "2"
    assert command[command.index("--noise-seed") + 1] == "2"
    assert command[command.index("--annotation-source") + 1] == "scripted"
    assert command[command.index("--max-saved-rollouts") + 1] == "00"
    suffix = command[command.index("--rollout-suffix-model-name") + 1]
    assert "grasp_part/n7_" in suffix
    assert "rep2_sim2_ann2" in suffix


def test_r180_command_uses_fixed_geodesic():
    condition = CONDITIONS[3]
    command = _build_command(
        auto_eval_path=Path("/tmp/auto_eval.sh"),
        task_group="one_leg+round_table+lamp",
        checkpoint=condition.checkpoint,
        flags=condition.flags,
        n_envs=3,
        n_rollouts=36,
        randomness="low",
        condition=condition,
        noise=FIXED_R180,
        apply_to=condition.apply_to,
        save_rollouts_count=0,
        replicate=ReplicateConfig(0, 0, 0),
    )

    assert command[command.index("--noise-mode") + 1] == "fixed_geodesic"
    assert command[command.index("--noise-pos-std-m") + 1] == "0.0"
    assert command[command.index("--noise-ori-std-deg") + 1] == "180.0"


def test_manifest_upsert_keeps_one_row_per_full_key(tmp_path):
    path = tmp_path / "manifest.jsonl"
    row = {
        "condition_id": "gp",
        "noise_id": "n5",
        "replicate_id": 0,
        "simulator_seed": 0,
        "annotation_seed": 0,
        "status": "failed",
    }
    _upsert_manifest(path, row)
    _upsert_manifest(path, {**row, "status": "ok"})

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert _manifest_key(rows[0]) == ("gp", "n5", 0, 0, 0)
