import torch

from src.eval.perturb_util import PerturbContext, PerturbRunner


def _context(step_idx, skill_states, ee_pos_vel=None):
    return PerturbContext(
        step_idx=step_idx,
        num_envs=len(skill_states),
        device=torch.device("cpu"),
        furniture_name="one_leg",
        task_name="one_leg",
        skill_states=skill_states,
        ee_pos_vel=ee_pos_vel,
    )


# -- random_small -----------------------------------------------------------


def test_random_small_triggers_on_internal_interval():
    runner = PerturbRunner("random_small")
    runner.random_small_interval = 2
    runner.random_small_max_force = 2.0
    runner.reset_episode(num_envs=3, device=torch.device("cpu"))

    first = runner.compute_force(_context(0, [None, None, None]))
    second = runner.compute_force(_context(1, [None, None, None]))
    third = runner.compute_force(_context(2, [None, None, None]))

    assert torch.linalg.norm(first, dim=-1).gt(0).all()
    assert torch.linalg.norm(second, dim=-1).eq(0).all()
    assert torch.linalg.norm(third, dim=-1).gt(0).all()
    assert torch.linalg.norm(first, dim=-1).le(2.0).all()


def test_random_small_applies_force():
    runner = PerturbRunner("random_small")
    assert runner.applies_force
    assert not runner.modifies_action
    assert not runner.subdivides_action


# -- short_large ------------------------------------------------------------


def test_short_large_waits_delay_and_fires_once():
    runner = PerturbRunner("short_large")
    runner.short_large_trigger_state = "place"
    runner.short_large_delay = 1
    runner.short_large_min_force = 4.0
    runner.short_large_max_force = 4.0
    runner.reset_episode(num_envs=2, device=torch.device("cpu"))

    first = runner.compute_force(_context(0, ["base-leg-place", "pick"]))
    second = runner.compute_force(_context(1, ["base-leg-place", "pick"]))
    third = runner.compute_force(_context(2, ["base-leg-place", "pick"]))

    assert torch.linalg.norm(first, dim=-1).eq(0).all()
    assert torch.isclose(torch.linalg.norm(second[0]), torch.tensor(4.0))
    assert torch.linalg.norm(second[1]).eq(0)
    assert torch.linalg.norm(third, dim=-1).eq(0).all()


def test_short_large_applies_force():
    runner = PerturbRunner("short_large")
    assert runner.applies_force
    assert not runner.modifies_action
    assert not runner.subdivides_action


# -- place_slowdown (action-chunk subdivision) ------------------------------


def test_place_slowdown_subdivides_during_place():
    runner = PerturbRunner("place_slowdown")
    runner.place_slowdown_subdivide_ratio = 3.0
    runner.reset_episode(num_envs=2, device=torch.device("cpu"))

    # env 0 in place, env 1 in pick
    ctx = _context(0, ["base-leg-place", "pick"])
    assert runner.get_subdivide_ratio(ctx) == 3.0


def test_place_slowdown_no_subdivide_outside_place():
    runner = PerturbRunner("place_slowdown")
    runner.place_slowdown_subdivide_ratio = 3.0
    runner.reset_episode(num_envs=1, device=torch.device("cpu"))

    ctx = _context(0, ["base-leg-screw"])
    assert runner.get_subdivide_ratio(ctx) == 1.0


def test_place_slowdown_does_not_apply_force_or_mutate_action():
    runner = PerturbRunner("place_slowdown")
    assert not runner.applies_force
    assert not runner.modifies_action
    assert runner.subdivides_action
    assert runner.requires_skill_annotations

    action = torch.tensor([[0.1, 0.2, -0.5], [0.3, 0.1, -0.4]])
    runner.reset_episode(num_envs=2, device=torch.device("cpu"))
    # modify_action is a no-op for place_slowdown
    assert torch.equal(
        runner.modify_action(action.clone(), _context(0, ["place", "pick"])),
        action,
    )


# -- place_drop (gripper-open release) --------------------------------------


def test_place_drop_modifies_action_and_not_force():
    runner = PerturbRunner("place_drop")
    assert runner.modifies_action
    assert not runner.applies_force
    assert not runner.subdivides_action
    assert runner.requires_skill_annotations


def test_place_drop_opens_gripper_after_delay_for_hold_steps():
    runner = PerturbRunner("place_drop")
    runner.place_drop_delay = 5
    runner.place_drop_hold = 15
    runner.reset_episode(num_envs=2, device=torch.device("cpu"))

    action = torch.zeros((2, 10), dtype=torch.float32)
    action[:, -1] = 1.0  # closed

    opened_env0 = opened_env1 = 0
    first_open = None
    for step in range(40):
        # env 0 stays in place; env 1 never in place
        ctx = _context(step, ["leg-table_top-place", None])
        a = runner.modify_action(action.clone(), ctx)
        if a[0, -1] < 0:
            opened_env0 += 1
            if first_open is None:
                first_open = step
        if a[1, -1] < 0:
            opened_env1 += 1

    assert first_open == 5  # delay steps before first open
    assert opened_env0 == 15  # hold steps
    assert opened_env1 == 0  # env not in place never opened

    s = runner.stats.summary()
    assert s["modified_env_steps"] == 15


def test_place_drop_fires_once_per_episode():
    runner = PerturbRunner("place_drop")
    runner.place_drop_delay = 2
    runner.place_drop_hold = 5
    runner.reset_episode(num_envs=1, device=torch.device("cpu"))

    action = torch.zeros((1, 10), dtype=torch.float32)
    action[:, -1] = 1.0

    def run(skills, nsteps):
        opened = 0
        for step in range(nsteps):
            ctx = _context(step, skills)
            a = runner.modify_action(action.clone(), ctx)
            if a[0, -1] < 0:
                opened += 1
        return opened

    o1 = run(["leg-table_top-place"], 30)      # fires once -> 5 open steps
    o2 = run(["leg-table_top-insert"], 10)     # leaves place
    o3 = run(["leg-table_top-place"], 30)      # re-enters place, must NOT refire

    assert o1 == 5
    assert o2 == 0
    assert o3 == 0


# -- stats ------------------------------------------------------------------


def test_stats_tracks_action_modifications():
    runner = PerturbRunner("place_drop")
    runner.place_drop_delay = 0
    runner.place_drop_hold = 1
    runner.reset_episode(num_envs=4, device=torch.device("cpu"))

    action = torch.zeros((4, 3), dtype=torch.float32)
    action[:, -1] = 1.0
    # env 0, 2, 3 in place; env 1 in pick
    runner.modify_action(action.clone(), _context(0, ["place", "pick", "place", "place"]))

    s = runner.stats.summary()
    assert s["modified_steps"] >= 1
    assert s["modified_env_steps"] == 3
