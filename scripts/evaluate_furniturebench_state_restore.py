#!/usr/bin/env python3
"""Probe Isaac Gym restoration from a FurnitureBench state-bank record.

The probe replays an identical short hold-action rollout twice with captured
velocities and once with all actor/DOF velocities zeroed.  It reports tensor
trajectory repeatability, contact forces, and active-part retention.  This is a
capability gate; it is not a policy-success experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Isaac Gym requires its native modules to be imported before PyTorch.
import isaacgym  # noqa: F401
from isaacgym import gymapi as _ig_gymapi  # noqa: F401
from isaacgym import gymtorch as _ig_gymtorch  # noqa: F401

import numpy as np
import torch
from furniture_bench.controllers import control_utils as C
from furniture_bench.sim_config import sim_config

from src.eval.skill_annotation_util import (
    get_annotation_bundle_all_envs,
    reset_skill_annotator,
)
from src.eval.state_bank import (
    capture_furniturebench_state,
    load_state_record,
    rebuild_furniturebench_contact_cache,
    restore_state_record,
    restore_state_records_batch,
    translate_root_state_origin,
)
from src.gym import get_rl_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test full-velocity and zero-velocity Isaac Gym state restoration."
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--contact-rebuild-steps",
        type=int,
        default=1,
        help=(
            "Static physics steps used to rebuild articulated/contact caches "
            "before reapplying the exact state; 0 tests raw tensor restore."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--visualize-state",
        action="store_true",
        help=(
            "Open a single-environment Isaac Gym viewer at the restored state. "
            "One physics step commits GPU tensor setters and rebuilds derived state; "
            "the viewer is then frozen."
        ),
    )
    parser.add_argument(
        "--inspect-seconds",
        type=float,
        default=600.0,
        help="Maximum viewer lifetime in --visualize-state mode (default: 600).",
    )
    parser.add_argument(
        "--inspect-restore-velocity",
        action="store_true",
        help=(
            "Retain recorded actor/DOF velocities in the viewer snapshot. "
            "The default is the experiment's zero-velocity restore."
        ),
    )
    parser.add_argument(
        "--grasp-recovery-probe",
        action="store_true",
        help=(
            "Compare direct restore with a staged restore that pins all furniture "
            "parts while the restored gripper closes to a stable grasp."
        ),
    )
    parser.add_argument(
        "--gripper-settle-steps",
        type=int,
        default=60,
        help="Physics steps with furniture parts pinned (default: 60).",
    )
    parser.add_argument(
        "--grasp-release-steps",
        type=int,
        default=96,
        help="Unpinned hold steps used to test grasp retention (default: 96).",
    )
    parser.add_argument(
        "--annotation-source", choices=("scripted",), required=True
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r} to JSON")


def _hold_action(env) -> torch.Tensor:
    ee_pos, ee_quat = env.get_ee_pose()
    # Rigid-body/Jacobian tensors are derived state and Isaac Gym provides no
    # setter for them. Immediately after a DOF restore they can still reflect
    # each env's pre-restore reset pose. Build one identical target from the
    # reference branch so the repeatability comparison is not confounded by
    # those stale per-env views; the first physics step re-synchronizes them.
    ee_pos = ee_pos[:1].repeat(env.num_envs, 1)
    ee_quat = ee_quat[:1].repeat(env.num_envs, 1)
    gripper = env.last_grasp[:1].reshape(1, 1).repeat(env.num_envs, 1)
    if env.act_rot_repr == "quat":
        orientation = ee_quat
    elif env.act_rot_repr == "rot_6d":
        orientation = C.quaternion_to_rotation_6d(ee_quat)
    elif env.act_rot_repr == "axis":
        orientation = C.quaternion_to_axis_angle(ee_quat)
    else:
        raise ValueError(f"Unsupported rotation representation: {env.act_rot_repr!r}")
    if env.action_type == "pos":
        return torch.cat((ee_pos, orientation, gripper), dim=1)
    if env.action_type == "delta":
        zeros = torch.zeros_like(ee_pos)
        if env.act_rot_repr == "quat":
            identity = torch.zeros_like(orientation)
            identity[:, 3] = 1.0
        elif env.act_rot_repr == "rot_6d":
            identity_quat = torch.zeros_like(ee_quat)
            identity_quat[:, 3] = 1.0
            identity = C.quaternion_to_rotation_6d(identity_quat)
        else:
            identity = torch.zeros_like(orientation)
        return torch.cat((zeros, identity, gripper), dim=1)
    raise ValueError(f"Unsupported action type: {env.action_type!r}")


def _contact_summary(env, env_idx: int, record) -> dict[str, float]:
    inputs = env.get_skill_annotation_inputs(env_idx=env_idx)
    left = float(torch.linalg.norm(inputs["left_finger_force"]).item())
    right = float(torch.linalg.norm(inputs["right_finger_force"]).item())
    part = {
        name: float(torch.linalg.norm(force).item())
        for name, force in inputs["part_contact_forces"].items()
    }
    active_name = record["metadata"].get("annotation_debug", {}).get(
        "active_part"
    )
    return {
        "left_finger": left,
        "right_finger": right,
        "max_part": max(part.values(), default=0.0),
        "active_part": part.get(active_name),
    }


def _active_part_distance(env, env_idx: int, record) -> float | None:
    active_part = record["metadata"].get("annotation_debug", {}).get("active_part")
    local_index = record["physics"]["layout"]["part_actor_local_indices"].get(
        active_part
    )
    if local_index is None:
        return None
    root_by_env = env.root_tensor.view(env.num_envs, -1, 13)
    part_pos = root_by_env[env_idx, int(local_index), :3]
    ee_pos, _ = env.get_ee_pose()
    base_pos = env.rb_states[env.base_idxs[env_idx], :3]
    ee_pos_world = ee_pos[env_idx] + base_pos
    return float(torch.linalg.norm(part_pos - ee_pos_world).item())


def _active_part_position(env, env_idx: int, record):
    active_part = record["metadata"].get("annotation_debug", {}).get(
        "active_part"
    )
    local_index = record["physics"]["layout"]["part_actor_local_indices"].get(
        active_part
    )
    if local_index is None:
        return None
    root_by_env = env.root_tensor.view(env.num_envs, -1, 13)
    return root_by_env[env_idx, int(local_index), :3].detach().cpu().numpy().copy()


def _run_parallel_branches(
    env, record, *, horizon: int, contact_rebuild_steps: int
) -> dict[str, dict]:
    branch_specs = (
        ("full_velocity_a", True),
        ("full_velocity_b", True),
        ("zero_velocity_a", False),
        ("zero_velocity_b", False),
    )
    if contact_rebuild_steps > 0:
        for env_idx, _ in enumerate(branch_specs):
            restore_state_record(
                env,
                record,
                env_idx=env_idx,
                restore_velocity=False,
                restore_rng=env_idx == 0,
            )
        rebuild_furniturebench_contact_cache(
            env, physics_steps=contact_rebuild_steps
        )

    for env_idx, (_, restore_velocity) in enumerate(branch_specs):
        restore_state_record(
            env,
            record,
            env_idx=env_idx,
            restore_velocity=restore_velocity,
            restore_rng=env_idx == 0,
        )
    annotations = get_annotation_bundle_all_envs(
        env,
        annotate_wrist_camera=False,
        resize_images=False,
        enable_verify=False,
    )
    action = _hold_action(env)
    branches = {
        name: {
            "annotation_after_restore": {
                "skill": annotations[env_idx].get("skill"),
                "skill_state": annotations[env_idx].get("skill_state"),
                "assembly_step": annotations[env_idx].get("assembly_step"),
            },
            "trajectory": [],
        }
        for env_idx, (name, _) in enumerate(branch_specs)
    }
    for step in range(horizon + 1):
        for env_idx, (name, _) in enumerate(branch_specs):
            state = capture_furniturebench_state(env, env_idx=env_idx)
            branches[name]["trajectory"].append(
                {
                    "step": step,
                    "root_state": state["root_state"],
                    "dof_state": state["dof_state"],
                    "env_origin": np.asarray(
                        state["layout"]["env_origin"], dtype=np.float32
                    ),
                    "contact": _contact_summary(env, env_idx, record),
                    "active_part_distance": _active_part_distance(
                        env, env_idx, record
                    ),
                    "active_part_position": _active_part_position(
                        env, env_idx, record
                    ),
                }
            )
        if step < horizon:
            env.step(action.clone(), sample_perturbations=False)
    return branches


def _trajectory_delta(first: dict, second: dict) -> dict[str, float]:
    root_pos = []
    root_vel = []
    dof_pos = []
    dof_vel = []
    per_step = []
    for left, right in zip(first["trajectory"], second["trajectory"]):
        left_root = left["root_state"]
        right_root = right["root_state"]
        current = {
            "step": int(left["step"]),
            "root_pose_max_abs": float(
                np.max(
                    np.abs(
                        left_root[:, :7] - right_root[:, :7]
                    )
                )
            ),
            "root_velocity_max_abs": float(
                np.max(
                    np.abs(
                        left_root[:, 7:] - right_root[:, 7:]
                    )
                )
            ),
            "dof_position_max_abs": float(
                np.max(
                    np.abs(
                        left["dof_state"][:, 0] - right["dof_state"][:, 0]
                    )
                )
            ),
            "dof_velocity_max_abs": float(
                np.max(
                    np.abs(
                        left["dof_state"][:, 1] - right["dof_state"][:, 1]
                    )
                )
            ),
        }
        root_pos.append(current["root_pose_max_abs"])
        root_vel.append(current["root_velocity_max_abs"])
        dof_pos.append(current["dof_position_max_abs"])
        dof_vel.append(current["dof_velocity_max_abs"])
        per_step.append(current)
    return {
        "root_pose_max_abs": float(max(root_pos, default=0.0)),
        "root_velocity_max_abs": float(max(root_vel, default=0.0)),
        "dof_position_max_abs": float(max(dof_pos, default=0.0)),
        "dof_velocity_max_abs": float(max(dof_vel, default=0.0)),
        "per_step": per_step,
    }


def _branch_summary(branch: dict) -> dict:
    contacts = branch["trajectory"]
    distances = [
        sample["active_part_distance"]
        for sample in contacts
        if sample["active_part_distance"] is not None
    ]
    part_positions = [
        sample["active_part_position"]
        for sample in contacts
        if sample["active_part_position"] is not None
    ]
    initial_dof = contacts[0]["dof_state"]
    final_dof = contacts[-1]["dof_state"]
    return {
        "annotation_after_restore": branch["annotation_after_restore"],
        "initial_contact": contacts[0]["contact"],
        "post_first_step_contact": (
            contacts[1]["contact"] if len(contacts) > 1 else None
        ),
        "final_contact": contacts[-1]["contact"],
        "active_part_distance_initial_m": distances[0] if distances else None,
        "active_part_distance_final_m": distances[-1] if distances else None,
        "active_part_distance_max_change_m": (
            max(abs(value - distances[0]) for value in distances)
            if distances
            else None
        ),
        "active_part_displacement_m": (
            float(np.linalg.norm(part_positions[-1] - part_positions[0]))
            if part_positions
            else None
        ),
        "gripper_width_initial_m": float(initial_dof[-2:, 0].sum()),
        "gripper_width_final_m": float(final_dof[-2:, 0].sum()),
    }


def _visualize_static_state(env, record, args) -> int:
    restore_state_record(
        env,
        record,
        env_idx=0,
        restore_velocity=args.inspect_restore_velocity,
        restore_rng=True,
    )
    # With the GPU pipeline, tensor setters are queued and are not reflected in
    # PhysX or the viewer until simulate() is called. Advance exactly once to
    # commit the restored root/DOF state and reconstruct derived/contact state,
    # then freeze the viewer. This is deliberately reported as a one-step
    # inspection rather than an exact zero-step physics checkpoint restore.
    rebuild_furniturebench_contact_cache(env, physics_steps=1)
    env.isaac_gym.step_graphics(env.sim)
    env.isaac_gym.draw_viewer(env.viewer, env.sim, False)

    metadata = record["metadata"]
    velocity_mode = "recorded" if args.inspect_restore_velocity else "zeroed"
    summary = {
        "state": str(args.state.expanduser().resolve()),
        "task": metadata.get("task"),
        "skill": metadata.get("skill"),
        "skill_state": metadata.get("skill_state"),
        "skill_stage": metadata.get("skill_stage"),
        "skill_visit": metadata.get("skill_visit_index"),
        "frames_after_skill_start": metadata.get("skill_frame_offset"),
        "active_part": metadata.get("annotation_debug", {}).get("active_part"),
        "velocity_mode": velocity_mode,
        "physics_steps_after_restore": 1,
        "exact_zero_step_physics_restore": False,
    }
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    print(
        "Isaac Gym viewer is open. Close the viewer to exit; "
        f"automatic timeout is {args.inspect_seconds:g} seconds.",
        flush=True,
    )

    deadline = time.monotonic() + args.inspect_seconds
    while (
        time.monotonic() < deadline
        and not env.isaac_gym.query_viewer_has_closed(env.viewer)
    ):
        # Graphics updates do not advance PhysX. The state remains frozen after
        # the single commit/contact-reconstruction step above.
        env.isaac_gym.step_graphics(env.sim)
        env.isaac_gym.draw_viewer(env.viewer, env.sim, False)
        time.sleep(1.0 / 30.0)
    return 0


def _set_restored_hold_controls(env, record) -> None:
    saved_dof = torch.as_tensor(
        record["physics"]["dof_state"][:, 0],
        device=env.device,
        dtype=env.dof_pos.dtype,
    )
    position_targets = saved_dof.unsqueeze(0).repeat(env.num_envs, 1)
    efforts = torch.zeros_like(position_targets)
    # FurnitureBench uses negative effort to close both Franka fingers.
    efforts[:, 7:9] = -float(sim_config["robot"]["gripper_torque"])
    env.isaac_gym.set_dof_position_target_tensor(
        env.sim, _ig_gymtorch.unwrap_tensor(position_targets.contiguous())
    )
    env.isaac_gym.set_dof_actuation_force_tensor(
        env.sim, _ig_gymtorch.unwrap_tensor(efforts.contiguous())
    )


def _pin_saved_parts(env, record, *, env_idx: int) -> None:
    root_by_env = env.root_tensor.view(env.num_envs, -1, 13)
    actors_per_env = root_by_env.shape[1]
    origin = env.isaac_gym.get_env_origin(env.envs[env_idx])
    target_origin = np.asarray([origin.x, origin.y, origin.z], dtype=np.float32)
    saved_root_np = translate_root_state_origin(
        record["physics"]["root_state"],
        saved_origin=record["physics"]["layout"]["env_origin"],
        target_origin=target_origin,
    )
    saved_root = torch.as_tensor(
        saved_root_np,
        device=env.device,
        dtype=root_by_env.dtype,
    )
    global_indices = torch.tensor(
        env.part_actor_idx_by_env[env_idx],
        device=env.device,
        dtype=torch.int32,
    )
    local_indices = global_indices.to(torch.long) - env_idx * actors_per_env
    root_by_env[env_idx, local_indices, :7] = saved_root[local_indices, :7]
    root_by_env[env_idx, local_indices, 7:13] = 0.0
    ok = env.isaac_gym.set_actor_root_state_tensor_indexed(
        env.sim,
        _ig_gymtorch.unwrap_tensor(env.root_tensor),
        _ig_gymtorch.unwrap_tensor(global_indices),
        global_indices.numel(),
    )
    if ok is False:
        raise RuntimeError("Isaac Gym rejected pinned furniture-part state")


def _simulate_and_refresh(env) -> None:
    env.isaac_gym.simulate(env.sim)
    env.isaac_gym.fetch_results(env.sim, True)
    env.isaac_gym.refresh_actor_root_state_tensor(env.sim)
    env.isaac_gym.refresh_dof_state_tensor(env.sim)
    env.isaac_gym.refresh_dof_force_tensor(env.sim)
    env.isaac_gym.refresh_rigid_body_state_tensor(env.sim)
    env.isaac_gym.refresh_net_contact_force_tensor(env.sim)
    env.isaac_gym.refresh_jacobian_tensors(env.sim)


def _grasp_probe_sample(env, record, *, env_idx: int, step: int) -> dict:
    state = capture_furniturebench_state(env, env_idx=env_idx)
    active_position = _active_part_position(env, env_idx, record)
    return {
        "step": int(step),
        "gripper_width_m": float(state["dof_state"][-2:, 0].sum()),
        "finger_velocity_max_abs": float(
            np.abs(state["dof_state"][-2:, 1]).max()
        ),
        "active_part_position": active_position,
        "contact": _contact_summary(env, env_idx, record),
    }


def _grasp_branch_summary(samples: list[dict]) -> dict:
    positions = [
        sample["active_part_position"]
        for sample in samples
        if sample["active_part_position"] is not None
    ]
    return {
        "initial": samples[0],
        "after_settle": samples[1],
        "final": samples[-1],
        "active_part_release_displacement_m": (
            float(np.linalg.norm(positions[-1] - positions[1]))
            if len(positions) >= 2
            else None
        ),
        "minimum_release_gripper_width_m": float(
            min(sample["gripper_width_m"] for sample in samples[2:])
        ),
    }


def _run_grasp_recovery_probe(env, record, args) -> int:
    direct_idx, staged_idx = 0, 1
    restore_state_records_batch(
        env,
        [record, record],
        env_indices=[direct_idx, staged_idx],
        restore_velocity=False,
        restore_rng=True,
        render_cameras=False,
    )
    _set_restored_hold_controls(env, record)

    branches = {"direct": [], "pinned_settle": []}
    # The queued batched restore is committed by the first simulation step.
    _simulate_and_refresh(env)
    branches["direct"].append(
        _grasp_probe_sample(env, record, env_idx=direct_idx, step=0)
    )
    branches["pinned_settle"].append(
        _grasp_probe_sample(env, record, env_idx=staged_idx, step=0)
    )

    for _ in range(max(0, args.gripper_settle_steps - 1)):
        _pin_saved_parts(env, record, env_idx=staged_idx)
        _simulate_and_refresh(env)
    branches["direct"].append(
        _grasp_probe_sample(
            env, record, env_idx=direct_idx, step=args.gripper_settle_steps
        )
    )
    branches["pinned_settle"].append(
        _grasp_probe_sample(
            env, record, env_idx=staged_idx, step=args.gripper_settle_steps
        )
    )

    # Reset the pinned branch's parts one final time. On the next simulate they
    # are released with their saved pose and zero velocity, while the fingers
    # retain the stable obstruction reached during settling.
    _pin_saved_parts(env, record, env_idx=staged_idx)
    sample_stride = max(1, args.grasp_release_steps // 12)
    for release_step in range(1, args.grasp_release_steps + 1):
        _simulate_and_refresh(env)
        if release_step % sample_stride == 0 or release_step == args.grasp_release_steps:
            branches["direct"].append(
                _grasp_probe_sample(
                    env, record, env_idx=direct_idx, step=release_step
                )
            )
            branches["pinned_settle"].append(
                _grasp_probe_sample(
                    env, record, env_idx=staged_idx, step=release_step
                )
            )

    summaries = {
        name: _grasp_branch_summary(samples) for name, samples in branches.items()
    }
    direct = summaries["direct"]
    staged = summaries["pinned_settle"]
    direct_final_contact = direct["final"]["contact"]
    final_contact = staged["final"]["contact"]
    direct_gates = {
        "direct_gripper_obstructed_after_settle": (
            direct["after_settle"]["gripper_width_m"] > 0.02
        ),
        "direct_gripper_obstructed_through_release": (
            direct["minimum_release_gripper_width_m"] > 0.02
        ),
        "direct_active_part_stable_within_5mm": (
            direct["active_part_release_displacement_m"] is not None
            and direct["active_part_release_displacement_m"] <= 0.005
        ),
        "direct_final_finger_contact": (
            direct_final_contact["left_finger"] > 0.1
            and direct_final_contact["right_finger"] > 0.1
        ),
    }
    staged_gates = {
        "staged_gripper_obstructed_after_settle": (
            staged["after_settle"]["gripper_width_m"] > 0.02
        ),
        "staged_gripper_obstructed_through_release": (
            staged["minimum_release_gripper_width_m"] > 0.02
        ),
        "staged_active_part_stable_within_5mm": (
            staged["active_part_release_displacement_m"] is not None
            and staged["active_part_release_displacement_m"] <= 0.005
        ),
        "staged_final_finger_contact": (
            final_contact["left_finger"] > 0.1
            and final_contact["right_finger"] > 0.1
        ),
    }
    result = {
        "schema": "rr-furniturebench-grasp-recovery-probe-v1",
        "state_path": str(args.state.expanduser().resolve()),
        "state_metadata": record["metadata"],
        "command": sys.argv,
        "protocol": {
            "velocity_mode": "zeroed",
            "gripper_settle_steps": args.gripper_settle_steps,
            "grasp_release_steps": args.grasp_release_steps,
            "pinned_parts": sorted(
                record["physics"]["layout"]["part_actor_local_indices"]
            ),
            "pin_method": (
                "Reapply every furniture actor's saved root pose and zero "
                "velocity before each PhysX settle step; retain collisions."
            ),
        },
        "branches": summaries,
        "gates": {**direct_gates, **staged_gates},
        "recommended_protocol": (
            "direct" if all(direct_gates.values()) else "pinned_settle"
        ),
    }
    output = args.output
    if output is None:
        output = args.state.with_suffix("").with_suffix(
            ".grasp_recovery_probe.json"
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    rendered = json.dumps(result, indent=2, sort_keys=True, default=_json_default)
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(rendered)
    print(f"grasp_recovery_probe={output}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.contact_rebuild_steps < 0:
        raise ValueError("--contact-rebuild-steps must be non-negative")
    if args.inspect_seconds <= 0:
        raise ValueError("--inspect-seconds must be positive")
    if args.gripper_settle_steps <= 0:
        raise ValueError("--gripper-settle-steps must be positive")
    if args.grasp_release_steps <= 0:
        raise ValueError("--grasp-release-steps must be positive")
    if args.visualize_state and args.grasp_recovery_probe:
        raise ValueError("--visualize-state and --grasp-recovery-probe are exclusive")
    record = load_state_record(args.state)
    metadata = record["metadata"]
    task = str(metadata["task"])
    action_type = str(metadata.get("action_type", "pos"))
    randomness = str(metadata.get("randomness", "low"))
    randomness = {"0": "low", "1": "med", "2": "high"}.get(
        randomness, randomness
    )
    act_rot_repr = str(metadata.get("act_rot_repr", "rot_6d"))

    env = get_rl_env(
        gpu_id=args.gpu,
        task=task,
        num_envs=(1 if args.visualize_state else 2 if args.grasp_recovery_probe else 4),
        randomness=randomness,
        max_env_steps=max(100, args.horizon + 10),
        resize_img=False,
        observation_space="image",
        act_rot_repr=act_rot_repr,
        action_type=action_type,
        april_tags=False,
        verbose=args.verbose,
        headless=not args.visualize_state,
    )
    env.reset()
    reset_skill_annotator(env)

    if args.visualize_state:
        return _visualize_static_state(env, record, args)
    if args.grasp_recovery_probe:
        return _run_grasp_recovery_probe(env, record, args)

    branches = _run_parallel_branches(
        env,
        record,
        horizon=args.horizon,
        contact_rebuild_steps=args.contact_rebuild_steps,
    )
    full_first = branches["full_velocity_a"]
    full_repeat = branches["full_velocity_b"]
    zero_velocity = branches["zero_velocity_a"]
    zero_velocity_repeat = branches["zero_velocity_b"]
    full_repeat_delta = _trajectory_delta(full_first, full_repeat)
    zero_repeat_delta = _trajectory_delta(zero_velocity, zero_velocity_repeat)
    full_zero_delta = _trajectory_delta(full_first, zero_velocity)
    full_summary = _branch_summary(full_first)
    zero_summary = _branch_summary(zero_velocity)
    zero_final_contact = zero_summary["final_contact"]
    contact_recovered = all(
        zero_final_contact.get(name) is not None
        and zero_final_contact[name] > 0.1
        for name in ("left_finger", "right_finger", "active_part")
    )
    gripper_obstruction_retained = zero_summary["gripper_width_final_m"] > 0.005

    result = {
        "schema": "rr-furniturebench-state-restore-probe-v1",
        "state_path": str(args.state.expanduser().resolve()),
        "state_metadata": metadata,
        "command": sys.argv,
        "horizon": args.horizon,
        "contact_rebuild_steps": args.contact_rebuild_steps,
        "parallel_full_velocity_repeat_delta": full_repeat_delta,
        "parallel_zero_velocity_repeat_delta": zero_repeat_delta,
        "full_vs_zero_velocity_delta": full_zero_delta,
        "full_velocity": full_summary,
        "zero_velocity": zero_summary,
        "gates": {
            "positions_exact_immediately_after_restore": (
                full_zero_delta["per_step"][0]["dof_position_max_abs"] == 0.0
            ),
            "captured_velocity_was_removed": (
                full_zero_delta["per_step"][0]["dof_velocity_max_abs"] > 0.0
                or full_zero_delta["per_step"][0]["root_velocity_max_abs"] > 0.0
            ),
            "zero_velocity_replicas_equal_at_restore": (
                zero_repeat_delta["per_step"][0]["dof_position_max_abs"] == 0.0
                and zero_repeat_delta["per_step"][0]["dof_velocity_max_abs"]
                == 0.0
            ),
            "zero_velocity_short_horizon_dof_repeatable_at_1e-4": (
                zero_repeat_delta["dof_position_max_abs"] <= 1e-4
            ),
            "gripper_active_part_contact_recovered_by_horizon": contact_recovered,
            "closed_gripper_obstruction_retained_by_horizon": (
                gripper_obstruction_retained
            ),
        },
        "contact_cache_serialized": False,
        "interpretation": (
            "Two full-velocity and two zero-velocity branches run in parallel "
            "environments. Isaac Gym exposes actor-root and DOF velocities "
            "but not articulated rigid-body/Jacobian setters or contact-solver cache "
            "serialization. Static rebuild steps reconstruct those derived states, "
            "after which the exact saved actor/DOF state is reapplied."
        ),
    }
    output = args.output
    if output is None:
        output = args.state.with_suffix("").with_suffix(".restore_probe.json")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    rendered = json.dumps(
        result, indent=2, sort_keys=True, default=_json_default
    )
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(rendered)
    print(f"restore_probe={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
