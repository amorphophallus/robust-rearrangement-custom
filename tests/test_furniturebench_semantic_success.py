import torch

from src.eval.progress_schema import get_task_progress_labels
from src.eval.rollout import _gate_success_with_scripted_fsm


def test_scripted_fsm_gate_rejects_lamp_physics_success_before_hood_completion():
    expected = get_task_progress_labels("lamp", "skill_states")
    bulb_only = [state for state in expected if not state.startswith("hood-base-")]
    rewards_complete = torch.tensor([True, True])

    accepted = _gate_success_with_scripted_fsm(
        rewards_complete,
        [expected, bulb_only],
        "lamp",
        enabled=True,
    )

    assert accepted.tolist() == [True, False]


def test_scripted_fsm_gate_preserves_legacy_behavior_when_disabled():
    rewards_complete = torch.tensor([[True], [False]])

    accepted = _gate_success_with_scripted_fsm(
        rewards_complete,
        [[], []],
        "lamp",
        enabled=False,
    )

    assert torch.equal(accepted, rewards_complete)


def test_scripted_fsm_gate_accepts_complete_maintained_task_sequences():
    tasks = ("one_leg", "round_table", "lamp")
    histories = [get_task_progress_labels(task, "skill_states") for task in tasks]

    for task, history in zip(tasks, histories):
        accepted = _gate_success_with_scripted_fsm(
            torch.tensor([[True]]),
            [history],
            task,
            enabled=True,
        )
        assert accepted.item() is True


def test_scripted_fsm_gate_leaves_tasks_without_a_maintained_schema_unchanged():
    rewards_complete = torch.tensor([True, False])

    accepted = _gate_success_with_scripted_fsm(
        rewards_complete,
        [[], []],
        "unknown-task",
        enabled=True,
    )

    assert torch.equal(accepted, rewards_complete)
