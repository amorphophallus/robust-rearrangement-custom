"""FurnitureBench fixed-intermediate-state capture and restoration.

The state contract deliberately stores Isaac Gym actor-root and DOF tensors,
instead of reconstructing state from policy observations.  Actor-root rows are
13-vectors ``[position, quaternion, linear velocity, angular velocity]`` and
DOF rows are ``[position, velocity]``.  FurnitureBench exposes actor-root
positions in each environment's local frame, even when parallel environments
have nonzero layout origins.  Isaac Gym does not expose its contact solver
cache, so contact continuity must be validated empirically after restore.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pickle
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch


STATE_BANK_SCHEMA = "rr-furniturebench-state-v2"


def _cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _scalar_or_array_for_env(value: Any, env_idx: int) -> Any:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return _cpu_copy(value[env_idx])
    if isinstance(value, (list, tuple)):
        return _cpu_copy(value[env_idx])
    return _cpu_copy(value)


def _actor_layout(env, env_idx: int, actors_per_env: int) -> dict[str, Any]:
    global_start = env_idx * actors_per_env
    origin = env.isaac_gym.get_env_origin(env.envs[env_idx])
    part_indices = {}
    for part, global_index in zip(
        env.furnitures[env_idx].parts,
        env.part_actor_idx_by_env[env_idx],
    ):
        part_indices[str(part.name)] = int(global_index) - global_start
    return {
        "actors_per_env": int(actors_per_env),
        "global_actor_start": int(global_start),
        "root_state_frame": "env-local",
        "env_origin": np.asarray(
            [origin.x, origin.y, origin.z], dtype=np.float32
        ),
        "part_actor_local_indices": part_indices,
        "franka_actor_global_index": int(
            env.franka_actor_idxs_all_t[env_idx].reshape(-1)[0].item()
        ),
    }


def capture_furniturebench_state(env, env_idx: int = 0) -> dict[str, Any]:
    """Capture one environment without advancing simulation."""

    if not 0 <= env_idx < int(env.num_envs):
        raise IndexError(f"env_idx {env_idx} outside [0, {env.num_envs})")
    # FurnitureBench refreshes rigid-body state after each simulation step but
    # not the separate actor-root tensor. Without this explicit refresh, free
    # furniture actors retain stale reset-time poses in saved records.
    env.isaac_gym.refresh_actor_root_state_tensor(env.sim)
    root_by_env = env.root_tensor.view(int(env.num_envs), -1, 13)
    dof_by_env = env.dof_states.view(int(env.num_envs), -1, 2)
    root_state = _as_numpy(root_by_env[env_idx]).astype(np.float32, copy=False)
    dof_state = _as_numpy(dof_by_env[env_idx]).astype(np.float32, copy=False)
    if root_state.ndim != 2 or root_state.shape[1] != 13:
        raise ValueError(f"Unexpected actor root-state shape: {root_state.shape}")
    if dof_state.ndim != 2 or dof_state.shape[1] != 2:
        raise ValueError(f"Unexpected DOF-state shape: {dof_state.shape}")

    runtime = {}
    for name in (
        "env_steps",
        "last_grasp",
        "already_assembled",
        "consecutive_assembled_steps",
        "reward",
        "done",
        "lb_rest_poses",
        "_moving_bulbs",
    ):
        if hasattr(env, name):
            runtime[name] = _scalar_or_array_for_env(
                getattr(env, name), env_idx
            )
    runtime["move_neutral"] = bool(getattr(env, "move_neutral", False))
    runtime["assemble_idx"] = int(getattr(env, "assemble_idx", 0))

    return {
        "root_state_refreshed": True,
        "root_state": root_state,
        "dof_state": dof_state,
        "runtime": runtime,
        "layout": _actor_layout(env, env_idx, root_state.shape[0]),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state().cpu(),
            "torch_cuda": (
                torch.cuda.get_rng_state(env.device).cpu()
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def state_arrays_for_restore(
    physics_state: Mapping[str, Any], *, restore_velocity: bool
) -> tuple[np.ndarray, np.ndarray]:
    root_state = np.asarray(physics_state["root_state"], dtype=np.float32).copy()
    dof_state = np.asarray(physics_state["dof_state"], dtype=np.float32).copy()
    if root_state.ndim != 2 or root_state.shape[1] != 13:
        raise ValueError(
            f"root_state must have shape [actors, 13], got {root_state.shape}"
        )
    if dof_state.ndim != 2 or dof_state.shape[1] != 2:
        raise ValueError(f"dof_state must have shape [dofs, 2], got {dof_state.shape}")
    if not restore_velocity:
        root_state[:, 7:13] = 0.0
        dof_state[:, 1] = 0.0
    return root_state, dof_state


def translate_root_state_origin(
    root_state: np.ndarray,
    *,
    saved_origin: Any,
    target_origin: Any,
) -> np.ndarray:
    """Translate world-frame actor roots between parallel env origins."""

    translated = np.asarray(root_state, dtype=np.float32).copy()
    saved = np.asarray(saved_origin, dtype=np.float32).reshape(3)
    target = np.asarray(target_origin, dtype=np.float32).reshape(3)
    translated[:, :3] += target - saved
    return translated


def _root_state_for_target_env(
    env, physics_state: Mapping[str, Any], root_state: np.ndarray, env_idx: int
) -> np.ndarray:
    layout = physics_state.get("layout", {})
    # FurnitureBench's root tensor is env-local.  Older v2 records did not
    # declare the frame, so treat a missing marker as env-local too.  Translating
    # those records by the visual grid origin moves actors away from the target
    # camera when restoring a state captured from env_idx > 0 into env 0.
    frame = layout.get("root_state_frame", "env-local")
    if frame == "env-local":
        return np.asarray(root_state, dtype=np.float32).copy()
    if frame != "sim-world":
        raise ValueError(f"Unsupported actor-root coordinate frame: {frame!r}")

    origin = env.isaac_gym.get_env_origin(env.envs[env_idx])
    target_origin = np.asarray([origin.x, origin.y, origin.z], dtype=np.float32)
    saved_origin = layout.get(
        "env_origin", np.zeros(3, dtype=np.float32)
    )
    return translate_root_state_origin(
        root_state, saved_origin=saved_origin, target_origin=target_origin
    )


def _restore_rng(rng_state: Mapping[str, Any], device) -> None:
    if not rng_state:
        return
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.random.set_rng_state(
        torch.as_tensor(rng_state["torch_cpu"], dtype=torch.uint8)
    )
    cuda_state = rng_state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(
            torch.as_tensor(cuda_state, dtype=torch.uint8), device=device
        )


def _assign_runtime_row(target: Any, env_idx: int, value: Any, device) -> None:
    if torch.is_tensor(target):
        restored = torch.as_tensor(value, device=target.device, dtype=target.dtype)
        target[env_idx].copy_(restored.reshape(target[env_idx].shape))
    elif isinstance(target, np.ndarray):
        target[env_idx] = np.asarray(value, dtype=target.dtype).reshape(
            target[env_idx].shape
        )
    elif isinstance(target, list):
        target[env_idx] = copy.deepcopy(value)
    else:
        raise TypeError(f"Unsupported per-environment runtime field: {type(target)!r}")


def refresh_furniturebench_tensors(env, *, render_cameras: bool = True) -> None:
    """Refresh Isaac Gym tensor views without stepping physics."""

    gym = env.isaac_gym
    gym.refresh_actor_root_state_tensor(env.sim)
    gym.refresh_dof_state_tensor(env.sim)
    gym.refresh_dof_force_tensor(env.sim)
    gym.refresh_rigid_body_state_tensor(env.sim)
    gym.refresh_net_contact_force_tensor(env.sim)
    gym.refresh_jacobian_tensors(env.sim)
    if render_cameras and getattr(env, "render_cameras", False):
        gym.step_graphics(env.sim)
        gym.render_all_camera_sensors(env.sim)


def rebuild_furniturebench_contact_cache(env, *, physics_steps: int = 1) -> None:
    """Advance static restored geometry to rebuild derived/contact state.

    Isaac Gym has setters for actor-root and DOF state but not for articulated
    rigid-body poses, Jacobians, or the PhysX contact warm-start cache. Callers
    should first restore every env with zero velocity, invoke this function,
    then reapply the exact saved state (including the desired velocity mode).
    """

    if physics_steps <= 0:
        raise ValueError("physics_steps must be positive")
    for _ in range(int(physics_steps)):
        env.isaac_gym.simulate(env.sim)
    env.isaac_gym.fetch_results(env.sim, True)
    refresh_furniturebench_tensors(env, render_cameras=True)


def settle_gripper_with_pinned_furniture(
    env,
    records: Iterable[Mapping[str, Any]],
    *,
    physics_steps: int = 3,
    arm_dof_count: int = 7,
) -> None:
    """Rebuild contacts while furniture and arm joints remain pinned.

    The two Franka finger joints are allowed to move into a contact-consistent
    configuration.  After each settling step, all actor roots and arm joints
    are restored to their saved zero-velocity state while the settled finger
    positions are retained.  One final unpinned physics step applies the last
    queued setters and verifies that the released state is physically stable.
    """

    from isaacgym import gymtorch

    records = list(records)
    if len(records) != int(env.num_envs):
        raise ValueError("pinned settle requires one record for every environment")
    if physics_steps <= 0:
        raise ValueError("physics_steps must be positive")

    root_by_env = env.root_tensor.view(int(env.num_envs), -1, 13)
    dof_by_env = env.dof_states.view(int(env.num_envs), -1, 2)
    if not 0 < int(arm_dof_count) < int(dof_by_env.shape[1]):
        raise ValueError("arm_dof_count must leave at least one gripper DOF")
    actors_per_env = int(root_by_env.shape[1])
    saved_roots = []
    saved_dofs = []
    for env_idx, record in enumerate(records):
        root_state, dof_state = state_arrays_for_restore(
            record["physics"], restore_velocity=False
        )
        saved_roots.append(
            _root_state_for_target_env(
                env, record["physics"], root_state, env_idx
            )
        )
        saved_dofs.append(dof_state)

    actor_indices = torch.arange(
        int(env.num_envs) * actors_per_env,
        device=env.device,
        dtype=torch.int32,
    )
    franka_indices = env.franka_actor_idxs_all_t.reshape(-1).to(
        device=env.device, dtype=torch.int32
    )

    for _ in range(int(physics_steps)):
        env.isaac_gym.simulate(env.sim)
        env.isaac_gym.fetch_results(env.sim, True)
        refresh_furniturebench_tensors(env, render_cameras=False)
        settled_fingers = dof_by_env[:, arm_dof_count:, 0].clone()
        for env_idx, (root_state, dof_state) in enumerate(
            zip(saved_roots, saved_dofs)
        ):
            root_by_env[env_idx].copy_(
                torch.as_tensor(
                    root_state, device=env.device, dtype=root_by_env.dtype
                )
            )
            dof_by_env[env_idx, :arm_dof_count].copy_(
                torch.as_tensor(
                    dof_state[:arm_dof_count],
                    device=env.device,
                    dtype=dof_by_env.dtype,
                )
            )
            dof_by_env[env_idx, arm_dof_count:, 0].copy_(
                settled_fingers[env_idx]
            )
            dof_by_env[env_idx, arm_dof_count:, 1].zero_()

        ok = env.isaac_gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.root_tensor),
            gymtorch.unwrap_tensor(actor_indices),
            actor_indices.numel(),
        )
        if ok is False:
            raise RuntimeError("Isaac Gym rejected pinned actor-root restoration")
        ok = env.isaac_gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_states),
            gymtorch.unwrap_tensor(franka_indices),
            franka_indices.numel(),
        )
        if ok is False:
            raise RuntimeError("Isaac Gym rejected pinned DOF restoration")
        env.isaac_gym.set_dof_position_target_tensor(
            env.sim, gymtorch.unwrap_tensor(env.dof_pos.contiguous())
        )
        zero_effort = torch.zeros_like(env.dof_pos)
        env.isaac_gym.set_dof_actuation_force_tensor(
            env.sim, gymtorch.unwrap_tensor(zero_effort)
        )

    # Apply the final queued pin, then release for one step so the audit and
    # first policy observation see an applied, physically active state.
    env.isaac_gym.simulate(env.sim)
    env.isaac_gym.fetch_results(env.sim, True)
    refresh_furniturebench_tensors(env, render_cameras=True)


def restore_furniturebench_state(
    env,
    physics_state: Mapping[str, Any],
    *,
    env_idx: int = 0,
    restore_velocity: bool = True,
    restore_rng: bool = True,
    render_cameras: bool = True,
) -> None:
    """Restore a captured environment without an implicit simulation step."""

    from isaacgym import gymtorch

    if not 0 <= env_idx < int(env.num_envs):
        raise IndexError(f"env_idx {env_idx} outside [0, {env.num_envs})")
    root_state, dof_state = state_arrays_for_restore(
        physics_state, restore_velocity=restore_velocity
    )
    root_state = _root_state_for_target_env(
        env, physics_state, root_state, env_idx
    )
    root_by_env = env.root_tensor.view(int(env.num_envs), -1, 13)
    dof_by_env = env.dof_states.view(int(env.num_envs), -1, 2)
    if tuple(root_by_env[env_idx].shape) != tuple(root_state.shape):
        raise ValueError(
            f"Snapshot/environment actor mismatch: {root_state.shape} vs "
            f"{tuple(root_by_env[env_idx].shape)}"
        )
    if tuple(dof_by_env[env_idx].shape) != tuple(dof_state.shape):
        raise ValueError(
            f"Snapshot/environment DOF mismatch: {dof_state.shape} vs "
            f"{tuple(dof_by_env[env_idx].shape)}"
        )

    root_by_env[env_idx].copy_(
        torch.as_tensor(root_state, device=env.device, dtype=root_by_env.dtype)
    )
    dof_by_env[env_idx].copy_(
        torch.as_tensor(dof_state, device=env.device, dtype=dof_by_env.dtype)
    )

    actors_per_env = root_by_env.shape[1]
    actor_indices = torch.arange(
        env_idx * actors_per_env,
        (env_idx + 1) * actors_per_env,
        device=env.device,
        dtype=torch.int32,
    )
    ok = env.isaac_gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_tensor),
        gymtorch.unwrap_tensor(actor_indices),
        actor_indices.numel(),
    )
    if ok is False:
        raise RuntimeError("Isaac Gym rejected actor-root state restoration")

    franka_actor_idx = env.franka_actor_idxs_all_t[env_idx].reshape(-1).to(
        device=env.device, dtype=torch.int32
    )
    ok = env.isaac_gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_states),
        gymtorch.unwrap_tensor(franka_actor_idx),
        franka_actor_idx.numel(),
    )
    if ok is False:
        raise RuntimeError("Isaac Gym rejected DOF state restoration")

    # Prevent a stale controller target from producing an impulse before the
    # first restored policy action establishes its own target.
    env.isaac_gym.set_dof_position_target_tensor(
        env.sim, gymtorch.unwrap_tensor(env.dof_pos.contiguous())
    )
    zero_effort = torch.zeros_like(env.dof_pos)
    env.isaac_gym.set_dof_actuation_force_tensor(
        env.sim, gymtorch.unwrap_tensor(zero_effort)
    )

    runtime = physics_state.get("runtime", {})
    for name in (
        "env_steps",
        "last_grasp",
        "already_assembled",
        "consecutive_assembled_steps",
        "reward",
        "done",
        "lb_rest_poses",
        "_moving_bulbs",
    ):
        if name in runtime and hasattr(env, name):
            _assign_runtime_row(getattr(env, name), env_idx, runtime[name], env.device)
    if "move_neutral" in runtime:
        env.move_neutral = bool(runtime["move_neutral"])
    if "assemble_idx" in runtime:
        env.assemble_idx = int(runtime["assemble_idx"])
    if restore_rng:
        _restore_rng(physics_state.get("rng", {}), env.device)
    # Do not call a tensor refresh here. With the GPU pipeline these setters
    # are queued until the next simulate() call; refreshing before then can
    # replace the just-written input buffers with stale pre-restore state.
    del render_cameras  # retained for backwards-compatible call signatures


def validate_state_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != STATE_BANK_SCHEMA:
        raise ValueError(f"Unsupported state-bank schema: {record.get('schema')!r}")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("State record is missing metadata")
    if metadata.get("annotation_source") != "scripted":
        raise ValueError("State bank requires annotation_source='scripted'")
    required = (
        "task",
        "episode_index",
        "frame_index",
        "skill_state",
        "skill_frame_offset",
    )
    missing = [name for name in required if metadata.get(name) is None]
    if missing:
        raise ValueError(f"State record is missing metadata fields: {missing}")
    if record["physics"].get("root_state_refreshed") is not True:
        raise ValueError(
            "State record does not certify a refreshed actor-root tensor"
        )
    state_arrays_for_restore(record["physics"], restore_velocity=True)


def save_state_record(path: Path, record: Mapping[str, Any]) -> str:
    validate_state_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(dict(record), protocol=pickle.HIGHEST_PROTOCOL)
    compressed = gzip.compress(payload, compresslevel=6)
    digest = hashlib.sha256(compressed).hexdigest()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(compressed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return digest


def load_state_record(path: Path) -> dict[str, Any]:
    with gzip.open(Path(path), "rb") as stream:
        record = pickle.load(stream)
    validate_state_record(record)
    return record


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return text or "none"


def _enum_or_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = getattr(value, "value", value)
    return str(value)


def _active_part_stage(
    env, env_idx: int, annotation: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose the scripted skill stage as state-selection metadata."""

    active_name = annotation.get("debug", {}).get("active_part")
    annotators = getattr(env, "_skill_annotators", None)
    if active_name is None or annotators is None or env_idx >= len(annotators):
        return {
            "active_part": active_name,
            "skill_stage": None,
            "part_skill_stage": None,
            "legacy_part_fsm_state": None,
        }
    for part in annotators[env_idx].furniture.parts:
        if str(part.name) == str(active_name):
            return {
                "active_part": str(active_name),
                "skill_stage": _enum_or_string(
                    getattr(part, "skill_state", None)
                ),
                "part_skill_stage": _enum_or_string(
                    getattr(part, "skill_state", None)
                ),
                # This low-level legacy FSM is not what drives the newer
                # skill annotator for every furniture type, but retaining it
                # is useful for auditing older task implementations.
                "legacy_part_fsm_state": _enum_or_string(
                    getattr(part, "_state", None)
                ),
            }
    return {
        "active_part": str(active_name),
        "skill_stage": None,
        "part_skill_stage": None,
        "legacy_part_fsm_state": None,
    }


def _randomness_name(randomness: Any) -> str:
    name = getattr(randomness, "name", None)
    if name is None:
        return str(randomness).lower()
    return {
        "medium": "med",
        "medium_collect": "med_collect",
    }.get(str(name).lower(), str(name).lower())


@dataclass
class _EnvSkillClock:
    skill_state: Optional[str] = None
    visit_idx: int = -1
    start_frame: int = 0
    visit_counts: dict[str, int] = field(default_factory=dict)

    def update(self, skill_state: str, frame_index: int) -> tuple[int, int]:
        if skill_state != self.skill_state:
            self.skill_state = skill_state
            self.start_frame = int(frame_index)
            self.visit_idx = self.visit_counts.get(skill_state, 0)
            self.visit_counts[skill_state] = self.visit_idx + 1
        return self.visit_idx, int(frame_index) - self.start_frame


class StateBankRecorder:
    """Write selected live simulator states and an append-only JSONL manifest."""

    def __init__(
        self,
        output_dir: Path,
        *,
        skill_offsets: Iterable[int] = (0,),
        stride: int = 0,
        include_preview: bool = True,
        source_metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir = self.output_dir / "states"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        if self.manifest_path.exists() or (
            self.states_dir.exists() and any(self.states_dir.iterdir())
        ):
            raise FileExistsError(
                f"Refusing to mix state-bank campaigns in {self.output_dir}"
            )
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.skill_offsets = frozenset(int(value) for value in skill_offsets)
        if any(value < 0 for value in self.skill_offsets):
            raise ValueError("skill offsets must be non-negative")
        self.stride = int(stride)
        if self.stride < 0:
            raise ValueError("state-bank stride must be non-negative")
        if not self.skill_offsets and self.stride == 0:
            raise ValueError("Specify at least one skill offset or a positive stride")
        self.include_preview = bool(include_preview)
        self.source_metadata = _cpu_copy(dict(source_metadata or {}))
        campaign = {
            "schema": STATE_BANK_SCHEMA,
            "annotation_source": "scripted",
            "skill_offsets": sorted(self.skill_offsets),
            "stride": self.stride,
            "include_preview": self.include_preview,
            "source": _jsonable(self.source_metadata),
        }
        campaign_path = self.output_dir / "campaign.json"
        temporary_path = campaign_path.with_name(f".{campaign_path.name}.tmp")
        temporary_path.write_text(
            json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary_path, campaign_path)
        self._clocks: dict[tuple[int, int], _EnvSkillClock] = {}

    def should_capture(self, offset: int) -> bool:
        return offset in self.skill_offsets or (
            self.stride > 0 and offset % self.stride == 0
        )

    def capture(
        self,
        env,
        *,
        annotations: list[Mapping[str, Any]],
        observation: Optional[Mapping[str, Any]],
        frame_index: int,
        episode_offset: int,
    ) -> list[Path]:
        written = []
        for env_idx, annotation in enumerate(annotations):
            skill_state = annotation.get("skill_state")
            if skill_state is None:
                continue
            skill_state = str(skill_state)
            episode_index = int(episode_offset) + env_idx
            clock = self._clocks.setdefault(
                (episode_index, env_idx), _EnvSkillClock()
            )
            visit_idx, offset = clock.update(skill_state, frame_index)
            if not self.should_capture(offset):
                continue

            preview = {}
            if self.include_preview and observation is not None:
                for key in ("color_image1", "color_image2"):
                    value = observation.get(key)
                    if value is not None:
                        preview[key] = _as_numpy(value[env_idx])

            annotator_state = None
            annotators = getattr(env, "_skill_annotators", None)
            if annotators is not None and env_idx < len(annotators):
                state_dict = getattr(annotators[env_idx], "state_dict", None)
                if callable(state_dict):
                    annotator_state = state_dict()

            stage = _active_part_stage(env, env_idx, annotation)
            randomness = _randomness_name(getattr(env, "randomness", "low"))

            metadata = {
                "task": str(
                    getattr(env, "furniture_name", getattr(env, "task_name", ""))
                ),
                "episode_index": episode_index,
                "env_index": env_idx,
                "frame_index": int(frame_index),
                "skill": annotation.get("skill"),
                "skill_state": skill_state,
                **stage,
                "assembly_step": annotation.get("assembly_step"),
                "skill_visit_index": int(visit_idx),
                "skill_frame_offset": int(offset),
                "annotation_source": "scripted",
                "randomness": randomness,
                "action_type": str(getattr(env, "action_type", "pos")),
                "act_rot_repr": str(getattr(env, "act_rot_repr", "quat")),
                "control_mode": str(getattr(env, "ctrl_mode", "diffik")),
                "guidance_frame": annotation.get("guidance_frame"),
                "guidance_point": _cpu_copy(annotation.get("guidance_point_clean")),
                "guidance_pose": _cpu_copy(annotation.get("guidance_pose_clean")),
                "guidance_point_2d": _cpu_copy(
                    annotation.get("guidance_point_2d", {})
                ),
                "camera_info": _cpu_copy(annotation.get("camera_info", {})),
                "annotation_debug": _cpu_copy(annotation.get("debug", {})),
            }
            record = {
                "schema": STATE_BANK_SCHEMA,
                "metadata": metadata,
                "physics": capture_furniturebench_state(env, env_idx),
                "annotation_state": annotator_state,
                "preview": preview,
                "source": self.source_metadata,
            }
            stem = (
                f"{_safe_component(metadata['task'])}__ep{episode_index:06d}"
                f"__{_safe_component(skill_state)}__v{visit_idx:02d}"
                f"__f{frame_index:05d}__o{offset:04d}"
            )
            path = self.states_dir / f"{stem}.pkl.gz"
            digest = save_state_record(path, record)
            manifest_entry = {
                **_jsonable(metadata),
                "path": str(path.relative_to(self.output_dir)),
                "sha256": digest,
                "root_linear_speed_max": float(
                    np.linalg.norm(
                        record["physics"]["root_state"][:, 7:10], axis=1
                    ).max()
                ),
                "root_angular_speed_max": float(
                    np.linalg.norm(
                        record["physics"]["root_state"][:, 10:13], axis=1
                    ).max()
                ),
                "dof_speed_max": float(
                    np.abs(record["physics"]["dof_state"][:, 1]).max()
                ),
            }
            with self.manifest_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest_entry, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            written.append(path)
        return written


def restore_state_record(
    env,
    record: Mapping[str, Any],
    *,
    env_idx: int = 0,
    restore_velocity: bool = True,
    restore_rng: bool = True,
) -> None:
    validate_state_record(record)
    task = str(getattr(env, "furniture_name", getattr(env, "task_name", "")))
    if task != str(record["metadata"]["task"]):
        raise ValueError(
            f"State task {record['metadata']['task']!r} does not match env {task!r}"
        )
    restore_furniturebench_state(
        env,
        record["physics"],
        env_idx=env_idx,
        restore_velocity=restore_velocity,
        restore_rng=restore_rng,
    )
    annotation_state = record.get("annotation_state")
    annotators = getattr(env, "_skill_annotators", None)
    if annotation_state is not None and annotators is not None:
        load_state_dict = getattr(annotators[env_idx], "load_state_dict", None)
        if callable(load_state_dict):
            load_state_dict(annotation_state, device=env.device)


def restore_state_records_batch(
    env,
    records: Iterable[Mapping[str, Any]],
    *,
    env_indices: Optional[Iterable[int]] = None,
    restore_velocity: bool = True,
    restore_rng: bool = True,
    render_cameras: bool = True,
) -> None:
    """Restore several parallel environments with one call per Gym setter.

    Isaac Gym's GPU pipeline permits each tensor setter only once between
    simulation steps.  This helper is therefore required for paired restore
    experiments; calling :func:`restore_state_record` once per environment can
    enqueue multiple root/DOF setters before the next ``simulate`` call.
    """

    from isaacgym import gymtorch

    records = list(records)
    if not records:
        raise ValueError("records must not be empty")
    if env_indices is None:
        env_indices = range(len(records))
    env_indices = [int(index) for index in env_indices]
    if len(env_indices) != len(records):
        raise ValueError("env_indices and records must have the same length")
    if len(set(env_indices)) != len(env_indices):
        raise ValueError("env_indices must be unique")

    task = str(getattr(env, "furniture_name", getattr(env, "task_name", "")))
    root_by_env = env.root_tensor.view(int(env.num_envs), -1, 13)
    dof_by_env = env.dof_states.view(int(env.num_envs), -1, 2)
    actors_per_env = int(root_by_env.shape[1])
    actor_indices = []
    franka_indices = []

    for env_idx, record in zip(env_indices, records):
        if not 0 <= env_idx < int(env.num_envs):
            raise IndexError(f"env_idx {env_idx} outside [0, {env.num_envs})")
        validate_state_record(record)
        if task != str(record["metadata"]["task"]):
            raise ValueError(
                f"State task {record['metadata']['task']!r} does not match env {task!r}"
            )
        root_state, dof_state = state_arrays_for_restore(
            record["physics"], restore_velocity=restore_velocity
        )
        root_state = _root_state_for_target_env(
            env, record["physics"], root_state, env_idx
        )
        if tuple(root_by_env[env_idx].shape) != tuple(root_state.shape):
            raise ValueError(
                f"Snapshot/environment actor mismatch: {root_state.shape} vs "
                f"{tuple(root_by_env[env_idx].shape)}"
            )
        if tuple(dof_by_env[env_idx].shape) != tuple(dof_state.shape):
            raise ValueError(
                f"Snapshot/environment DOF mismatch: {dof_state.shape} vs "
                f"{tuple(dof_by_env[env_idx].shape)}"
            )
        root_by_env[env_idx].copy_(
            torch.as_tensor(root_state, device=env.device, dtype=root_by_env.dtype)
        )
        dof_by_env[env_idx].copy_(
            torch.as_tensor(dof_state, device=env.device, dtype=dof_by_env.dtype)
        )
        actor_indices.extend(
            range(env_idx * actors_per_env, (env_idx + 1) * actors_per_env)
        )
        franka_indices.extend(
            int(value)
            for value in env.franka_actor_idxs_all_t[env_idx].reshape(-1).tolist()
        )

        runtime = record["physics"].get("runtime", {})
        for name in (
            "env_steps",
            "last_grasp",
            "already_assembled",
            "consecutive_assembled_steps",
            "reward",
            "done",
            "lb_rest_poses",
            "_moving_bulbs",
        ):
            if name in runtime and hasattr(env, name):
                _assign_runtime_row(
                    getattr(env, name), env_idx, runtime[name], env.device
                )

        annotation_state = record.get("annotation_state")
        annotators = getattr(env, "_skill_annotators", None)
        if annotation_state is not None and annotators is not None:
            load_state_dict = getattr(annotators[env_idx], "load_state_dict", None)
            if callable(load_state_dict):
                load_state_dict(annotation_state, device=env.device)

    first_runtime = records[0]["physics"].get("runtime", {})
    if "move_neutral" in first_runtime:
        env.move_neutral = bool(first_runtime["move_neutral"])
    if "assemble_idx" in first_runtime:
        env.assemble_idx = int(first_runtime["assemble_idx"])

    actor_indices_t = torch.tensor(
        actor_indices, device=env.device, dtype=torch.int32
    )
    ok = env.isaac_gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_tensor),
        gymtorch.unwrap_tensor(actor_indices_t),
        actor_indices_t.numel(),
    )
    if ok is False:
        raise RuntimeError("Isaac Gym rejected batched actor-root restoration")

    franka_indices_t = torch.tensor(
        franka_indices, device=env.device, dtype=torch.int32
    )
    ok = env.isaac_gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_states),
        gymtorch.unwrap_tensor(franka_indices_t),
        franka_indices_t.numel(),
    )
    if ok is False:
        raise RuntimeError("Isaac Gym rejected batched DOF restoration")

    env.isaac_gym.set_dof_position_target_tensor(
        env.sim, gymtorch.unwrap_tensor(env.dof_pos.contiguous())
    )
    zero_effort = torch.zeros_like(env.dof_pos)
    env.isaac_gym.set_dof_actuation_force_tensor(
        env.sim, gymtorch.unwrap_tensor(zero_effort)
    )
    if restore_rng:
        _restore_rng(records[0]["physics"].get("rng", {}), env.device)
    # As above, refreshing before simulate() can clobber the queued restore.
    del render_cameras
