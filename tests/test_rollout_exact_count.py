from src.eval.rollout import _accepted_env_count, _saved_media_env_count_for_round


def test_nondivisible_final_batch_accepts_only_remaining_rollouts():
    assert _accepted_env_count(
        num_envs=8,
        completed_rollouts=32,
        requested_rollouts=36,
        target_mode=False,
    ) == 4


def test_target_mode_accepts_entire_vectorized_batch():
    assert _accepted_env_count(
        num_envs=8,
        completed_rollouts=32,
        requested_rollouts=36,
        target_mode=True,
    ) == 8


def test_media_collection_is_limited_to_remaining_file_save_budget():
    assert _saved_media_env_count_for_round(
        num_envs=18,
        accepted_env_count=18,
        save_rollouts_this_round=True,
        save_rollouts_to_wandb=False,
        save_failures=True,
        max_saved_rollouts=10,
        saved_rollouts_count=0,
    ) == 10


def test_media_collection_keeps_full_batch_when_success_is_required_for_save():
    assert _saved_media_env_count_for_round(
        num_envs=18,
        accepted_env_count=18,
        save_rollouts_this_round=True,
        save_rollouts_to_wandb=False,
        save_failures=False,
        max_saved_rollouts=10,
        saved_rollouts_count=0,
    ) == 18


def test_media_collection_respects_exact_partial_final_batch():
    assert _saved_media_env_count_for_round(
        num_envs=18,
        accepted_env_count=4,
        save_rollouts_this_round=True,
        save_rollouts_to_wandb=False,
        save_failures=True,
        max_saved_rollouts=10,
        saved_rollouts_count=7,
    ) == 3


def test_media_collection_is_disabled_when_round_is_not_saved():
    assert _saved_media_env_count_for_round(
        num_envs=18,
        accepted_env_count=18,
        save_rollouts_this_round=False,
        save_rollouts_to_wandb=False,
        save_failures=True,
        max_saved_rollouts=10,
        saved_rollouts_count=10,
    ) == 0
