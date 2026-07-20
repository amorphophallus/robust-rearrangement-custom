from __future__ import annotations

import argparse
import gzip
import json
import lzma
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image


CAMERA_KEY_TO_NAME = {
    "color_image2": "front",
    "color_image1": "wrist",
}
CAMERA_NAME_TO_KEY = {value: key for key, value in CAMERA_KEY_TO_NAME.items()}

STATE_INFO_BASE_KEYS = (
    "ee_pos_sim",
    "ee_quat_sim",
    "ee_pos_vel",
    "ee_ori_vel",
    "gripper_width",
)
STATE_INFO_EXTRA_KEYS = (
    "joint_positions",
    "joint_velocities",
    "joint_torques",
)
COORDINATE_FRAMES = {
    "target_point_2d": (
        "front-camera resized image pixel coordinates [u, v]; "
        "u increases left-to-right and v increases top-to-bottom"
    ),
    "target_point_3d": "sim_local position in meters, same frame as state_info.base.ee_pos_sim",
    "state_info.base.ee_pos_sim": "sim_local position in meters",
    "state_info.base.ee_quat_sim": "sim_local end-effector orientation quaternion [x, y, z, w]",
    "state_info.base.ee_pos_vel": "end-effector linear velocity from rollout robot_state",
    "state_info.base.ee_ori_vel": "end-effector angular velocity from rollout robot_state",
}

TASK_ALIASES = {
    "oneleg": "one_leg",
    "one-leg": "one_leg",
    "roundtable": "round_table",
    "round-table": "round_table",
    "round_tabl": "round_table",
}

BASE_SYSTEM_PROMPT = (
    "You are a vision-language robot policy assistant for furniture assembly. "
    "Use the images and robot proprioceptive state to predict the current skill "
    "and the next target point. Return strict JSON only; do not output any "
    "extra explanation. target_point_2d is a front-camera image pixel coordinate "
    "[u, v], where u increases from left to right and v increases from top to bottom. "
    "target_point_3d and state_info.base.ee_pos_sim are expressed in the same "
    "sim_local coordinate frame."
)

TASK_SYSTEM_PROMPTS = {
    "one_leg": (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        "Task: one_leg assembly. The goal is to attach one leg to the tabletop. "
        "Part shape description: the tabletop is a flat panel with 4 visible leg socket "
        "near each corner; the leg is a long narrow support piece with a connector end "
        "that can be inserted into the tabletop socket; the obstacle is a fixed block "
        "used as an alignment reference. "
        "Interpret the current frame according to the following assembly process. "
        "Step 1: use the gripper to grasp the middle region of the rear edge of the tabletop. "
        "Step 2: gently push the tabletop until it is tightly aligned against the obstacle "
        "at the lower-right corner. "
        "Step 3: approach and grasp the rightmost leg. "
        "Step 4: move the leg above the leg socket at the lower-right corner of the tabletop, "
        "and align the connector end of the leg with the socket opening. "
        "Step 5: insert the connector end of the leg into the tabletop leg socket. "
        "Step 6: rotate or screw the leg until it is fixed to the tabletop."
    ),
    "round_table": (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        "Task: round_table assembly. The goal is to connect the tabletop, leg, and base "
        "Part shape description: the tabletop is a round flat top plate with a leg socket; "
        "the leg is a long narrow support post with connector ends; the base is a round "
        "support piece with a central connector region that mates with the leg; the obstacle "
        "is a fixed block used as an alignment reference. "
        "Interpret the current frame according to the following assembly process. "
        "Step 1: use the gripper to contact the left-rear edge of the tabletop socket, "
        "then push it toward the lower-right corner until it is tightly aligned against the obstacle. "
        "Step 2: grasp the leg and move the tabletop-side connector end of the leg "
        "to the opening of the tabletop leg socket. "
        "Step 3: insert the leg into the tabletop leg socket, then rotate or screw the leg "
        "until it is tightened. "
        "Step 4: grasp the base. "
        "Step 5: align the central connector region of the base with the socket on the leg, "
        "then place and insert it. "
        "Step 6: rotate and tighten the base until the base is fixed to the leg, "
        "completing the round_table assembly."
    ),
    "lamp": (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        "Task: lamp assembly. The goal is to connect the base, bulb, and hood in sequence "
        "Part shape description: the base is the lower support piece with a bulb socket; "
        "the bulb is the smaller rounded part that inserts into the base socket; "
        "the hood is the larger shade-like cover. "
        "the obstacle is a fixed block used as an alignment reference. "
        "Interpret the current frame according to the following assembly process. "
        "Step 1: use the gripper to contact the left-rear edge of the base, then push it "
        "toward the lower-right corner until it is tightly aligned against the obstacle. "
        "Step 2: grasp the bulb and move it above the bulb socket on the base. "
        "Step 3: place the bulb at the opening of the base bulb socket, then insert it. "
        "Step 4: rotate or screw the bulb until it is tightened. "
        "Step 5: grasp the hood and move it above the assembled base-bulb subassembly. "
        "Step 6: align the central mounting hole of the hood with the top connection position "
        "of the base-bulb subassembly, then place the hood in position."
    ),
}

STATE_INFO_PLACEHOLDER = "<state_info>"
OUTPUT_JSON_EXAMPLE = (
    '{"skill": "pick", "target_point_2d": [160.0, 153.0], '
    '"target_point_3d": [0.160508, 0.000166, 0.430685]}'
)

DEFAULT_USER_PROMPT = (
    "This is the front camera image:\n"
    "<image>\n"
    "This is the wrist camera image:\n"
    "<image>\n"
    "This is the robot proprioceptive state information:\n"
    f"{STATE_INFO_PLACEHOLDER}\n"
    "Please analyze the images and state information, then provide the current skill "
    "and target point. Return the answer in JSON format exactly like this example: "
    f"{OUTPUT_JSON_EXAMPLE}"
)


def _default_data_dir_raw() -> Path:
    return Path.cwd()


def _prepend_env_path(value: str, path: Path) -> str:
    path_text = str(path)
    if not value:
        return path_text
    parts = value.split(os.pathsep)
    if path_text in parts:
        return value
    return os.pathsep.join([path_text, value])


def _ensure_data_dir_raw(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    target_env = dict(os.environ if env is None else env)
    target_env.setdefault("DATA_DIR_RAW", str(_default_data_dir_raw()))
    conda_lib = Path(sys.prefix) / "lib"
    if conda_lib.exists():
        target_env["LD_LIBRARY_PATH"] = _prepend_env_path(
            target_env.get("LD_LIBRARY_PATH", ""),
            conda_lib,
        )
    return target_env


def _normalize_task(task: str) -> str:
    normalized = task.strip()
    return TASK_ALIASES.get(normalized, normalized)


def _parse_task_counts(values: Optional[list[str]], default_count: Optional[int]) -> dict[str, int]:
    task_counts: dict[str, int] = {}
    if values:
        for value in values:
            if "=" not in value:
                if default_count is None:
                    raise ValueError(
                        f"Task spec {value!r} must be TASK=COUNT when --rollouts-per-task is not set."
                    )
                task_counts[_normalize_task(value)] = int(default_count)
                continue
            task, count = value.split("=", 1)
            task_counts[_normalize_task(task)] = int(count)

    if not task_counts:
        raise ValueError("No tasks specified. Use --task-rollout TASK=COUNT or --tasks with --rollouts-per-task.")

    for task, count in task_counts.items():
        if count <= 0:
            raise ValueError(f"Rollout count for {task} must be positive, got {count}.")
    return task_counts


def _load_pickle(path: Path) -> Any:
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def _jsonify(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist(), ndigits=ndigits)
    if isinstance(value, np.generic):
        return _jsonify(value.item(), ndigits=ndigits)
    if isinstance(value, dict):
        return {str(k): _jsonify(v, ndigits=ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v, ndigits=ndigits) for v in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(value, ndigits)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _safe_stem(path: Path) -> str:
    name = path.name
    for suffix in (".pkl.xz", ".pkl.gz", ".pickle", ".pkl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _as_image_uint8(array: Any) -> np.ndarray:
    image = np.asarray(array)
    image = np.squeeze(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HxWxC image, got shape {image.shape}.")
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def _save_png(array: Any, path: Path) -> None:
    image = _as_image_uint8(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _save_depth(array: Any, path: Path) -> bool:
    if array is None:
        return False
    depth = np.asarray(array)
    if depth.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth.astype(np.float32, copy=False))
    return True


def _system_prompt_for_task(task: Optional[str], override: Optional[str]) -> str:
    if override:
        return override
    return TASK_SYSTEM_PROMPTS.get(str(task), BASE_SYSTEM_PROMPT)


def _has_sim_local_eepose(robot_state: Any) -> bool:
    return (
        isinstance(robot_state, dict)
        and robot_state.get("ee_pos_sim") is not None
        and robot_state.get("ee_quat_sim") is not None
    )


def _target_point_2d_payload(guidance_point_2d: Any) -> Any:
    if not isinstance(guidance_point_2d, dict):
        return None
    return _jsonify(guidance_point_2d.get("color_image2"))


def _state_info_payload(obs: dict[str, Any]) -> dict[str, Any]:
    robot_state = obs.get("robot_state")
    if not isinstance(robot_state, dict):
        return {
            "base": {key: None for key in STATE_INFO_BASE_KEYS},
            "extra": {
                key: None
                for key in (*STATE_INFO_EXTRA_KEYS, "parts_poses")
            },
        }

    base = {
        key: _jsonify(robot_state.get(key))
        for key in STATE_INFO_BASE_KEYS
    }
    if base["ee_pos_sim"] is None and robot_state.get("ee_pos") is not None:
        base["ee_pos_sim"] = _jsonify(robot_state.get("ee_pos"))
    if base["ee_quat_sim"] is None and robot_state.get("ee_quat") is not None:
        base["ee_quat_sim"] = _jsonify(robot_state.get("ee_quat"))
    state_info = {
        "base": base,
        "extra": {
            key: _jsonify(robot_state.get(key))
            for key in STATE_INFO_EXTRA_KEYS
        },
    }
    if obs.get("parts_poses") is not None:
        state_info["extra"]["parts_poses"] = _jsonify(obs.get("parts_poses"))
    return state_info


def _assistant_payload(
    obs: dict[str, Any],
    task: Optional[str],
) -> dict[str, Any]:
    return {
        "skill": _jsonify(obs.get("skill")),
        "target_point_2d": _target_point_2d_payload(obs.get("guidance_point_2d")),
        "target_point_3d": _jsonify(obs.get("guidance_point")),
    }


def _user_prompt_with_state_info_placeholder(user_prompt: str) -> str:
    if STATE_INFO_PLACEHOLDER in user_prompt:
        return user_prompt
    return f"{user_prompt}\n{STATE_INFO_PLACEHOLDER}"


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _iter_frame_indices(num_obs: int, frame_stride: int, max_frames: int) -> Iterable[int]:
    emitted = 0
    for idx in range(0, num_obs, frame_stride):
        if max_frames > 0 and emitted >= max_frames:
            break
        emitted += 1
        yield idx


def _task_from_pickle(path: Path, data: dict[str, Any]) -> Optional[str]:
    task = data.get("task", data.get("furniture"))
    if task is not None:
        return str(task)
    parts = path.parts
    if "sim" in parts:
        idx = parts.index("sim")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _write_records(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    formats: str,
) -> dict[str, str]:
    def user_assistant_content(record: dict[str, Any]) -> tuple[str, str]:
        user_text = ""
        assistant_text = ""
        for message in record.get("messages", []):
            role = message.get("role")
            if role == "user" and not user_text:
                user_text = message.get("content", "")
            elif role == "assistant" and not assistant_text:
                assistant_text = message.get("content", "")
        return user_text, assistant_text

    outputs: dict[str, str] = {}
    if formats in ("both", "messages-jsonl"):
        messages_path = output_dir / "messages.jsonl"
        with messages_path.open("w") as f:
            for record in records:
                f.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "images": record["image"],
                            "state_info": record.get("state_info"),
                            "messages": record["messages"],
                            "metadata": record["metadata"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        outputs["messages_jsonl"] = str(messages_path)

    if formats in ("both", "sharegpt-json"):
        sharegpt_path = output_dir / "qwen_llava_sharegpt.json"
        sharegpt_records = []
        for record in records:
            user_text, assistant_text = user_assistant_content(record)
            sharegpt_records.append(
                {
                    "id": record["id"],
                    "image": record["image"],
                    "state_info": record.get("state_info"),
                    "conversations": [
                        {"from": "human", "value": user_text},
                        {"from": "gpt", "value": assistant_text},
                    ],
                    "metadata": record["metadata"],
                }
            )
        with sharegpt_path.open("w") as f:
            json.dump(sharegpt_records, f, indent=2, ensure_ascii=False)
        outputs["sharegpt_json"] = str(sharegpt_path)

    return outputs


def _read_existing_messages_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            records.append(
                {
                    "id": payload["id"],
                    "image": payload.get("images", payload.get("image", [])),
                    "state_info": payload.get("state_info"),
                    "messages": payload["messages"],
                    "metadata": payload.get("metadata", {}),
                }
            )
    return records


def _read_existing_sharegpt_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}.")

    records: list[dict[str, Any]] = []
    for item in payload:
        conversations = item.get("conversations", [])
        user_text = ""
        assistant_text = ""
        for turn in conversations:
            if turn.get("from") == "human" and not user_text:
                user_text = turn.get("value", "")
            elif turn.get("from") == "gpt" and not assistant_text:
                assistant_text = turn.get("value", "")
        records.append(
            {
                "id": item["id"],
                "image": item.get("image", []),
                "state_info": item.get("state_info"),
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "metadata": item.get("metadata", {}),
            }
        )
    return records


def _read_existing_records(output_dir: Path) -> list[dict[str, Any]]:
    messages_path = output_dir / "messages.jsonl"
    if messages_path.exists():
        return _read_existing_messages_jsonl(messages_path)
    return _read_existing_sharegpt_json(output_dir / "qwen_llava_sharegpt.json")


def _prepare_output_dir(output_dir: Path, output_mode: str) -> list[dict[str, Any]]:
    generated_paths = [
        output_dir / "messages.jsonl",
        output_dir / "qwen_llava_sharegpt.json",
        output_dir / "manifest.json",
        output_dir / "images",
        output_dir / "depth",
    ]
    existing_paths = [path for path in generated_paths if path.exists()]

    if output_mode == "error" and existing_paths:
        existing = "\n".join(f"  {path}" for path in existing_paths)
        raise FileExistsError(
            "Output dataset already exists. Use --output-mode append or overwrite.\n"
            f"{existing}"
        )

    if output_mode == "overwrite":
        for path in existing_paths:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return []

    if output_mode == "append":
        return _read_existing_records(output_dir)

    return []


def gather_pickle_paths(args: argparse.Namespace, task_counts: dict[str, int]) -> list[Path]:
    env = _ensure_data_dir_raw()
    os.environ.update({"DATA_DIR_RAW": env["DATA_DIR_RAW"]})

    if args.input_dir:
        paths: list[Path] = []
        for input_dir in args.input_dir:
            root = Path(input_dir).expanduser().resolve()
            if root.is_file():
                paths.append(root)
                continue
            outcome_root = root / args.demo_outcome
            search_root = outcome_root if outcome_root.exists() else root
            paths.extend(
                sorted(
                    search_root.rglob("*.pkl*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            )
        return paths

    paths = []
    for task in task_counts:
        raw_root = (
            Path(env["DATA_DIR_RAW"])
            / "raw"
            / "diffik"
            / "sim"
            / task
            / "rollout"
            / args.randomness
        )
        if args.suffix:
            raw_root = raw_root / args.suffix
        raw_root = raw_root / args.demo_outcome
        task_paths = sorted(raw_root.glob("*.pkl*")) if raw_root.exists() else []
        paths.extend(sorted(task_paths, key=lambda p: p.stat().st_mtime, reverse=True))
    return paths


def convert_pickles_to_vlm_sft(args: argparse.Namespace) -> dict[str, Any]:
    task_counts = _parse_task_counts(args.task_rollout or args.tasks, args.rollouts_per_task)
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_records = _prepare_output_dir(output_dir, args.output_mode)
    existing_ids = {str(record["id"]) for record in existing_records}

    pickle_paths = gather_pickle_paths(args, task_counts)
    selected_per_task: dict[str, int] = defaultdict(int)
    new_records: list[dict[str, Any]] = []
    skipped = Counter()

    for pickle_path in pickle_paths:
        try:
            data = _load_pickle(pickle_path)
        except Exception as exc:
            skipped[f"load_error:{type(exc).__name__}"] += 1
            continue
        if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
            skipped["invalid_structure"] += 1
            continue

        task = _normalize_task(str(_task_from_pickle(pickle_path, data)))
        if task not in task_counts:
            skipped["task_not_requested"] += 1
            continue
        if selected_per_task[task] >= task_counts[task]:
            skipped["task_limit_reached"] += 1
            continue

        observations = data["observations"]
        if not observations:
            skipped["empty_observations"] += 1
            continue

        rollout_stem = _safe_stem(pickle_path)
        rollout_index = selected_per_task[task]
        emitted_for_rollout = 0

        for frame_idx in _iter_frame_indices(
            len(observations),
            frame_stride=args.frame_stride,
            max_frames=args.max_frames_per_rollout,
        ):
            obs = observations[frame_idx]
            if not isinstance(obs, dict):
                skipped["invalid_observation"] += 1
                continue
            if obs.get("color_image2") is None or obs.get("color_image1") is None:
                skipped["missing_required_images"] += 1
                continue
            if not args.allow_legacy_eepose and not _has_sim_local_eepose(
                obs.get("robot_state")
            ):
                skipped["missing_sim_local_eepose"] += 1
                continue

            sample_id = f"{task}_{rollout_index:05d}_{rollout_stem}_frame_{frame_idx:05d}"
            if sample_id in existing_ids:
                skipped["duplicate_sample_id"] += 1
                continue
            front_path = image_dir / task / f"{sample_id}_front.png"
            wrist_path = image_dir / task / f"{sample_id}_wrist.png"
            try:
                _save_png(obs["color_image2"], front_path)
                _save_png(obs["color_image1"], wrist_path)
            except Exception as exc:
                skipped[f"image_error:{type(exc).__name__}"] += 1
                continue

            depth_paths: dict[str, Optional[str]] = {"front": None, "wrist": None}
            if args.save_depth_npy:
                front_depth_path = depth_dir / task / f"{sample_id}_front_depth.npy"
                wrist_depth_path = depth_dir / task / f"{sample_id}_wrist_depth.npy"
                if _save_depth(obs.get("depth_image2"), front_depth_path):
                    depth_paths["front"] = _relative_to(front_depth_path, output_dir)
                if _save_depth(obs.get("depth_image1"), wrist_depth_path):
                    depth_paths["wrist"] = _relative_to(wrist_depth_path, output_dir)

            state_info = _state_info_payload(obs)
            assistant_obj = _assistant_payload(obs, task=task)
            assistant_text = json.dumps(assistant_obj, ensure_ascii=False, sort_keys=True)
            user_text = _user_prompt_with_state_info_placeholder(args.user_prompt)

            image_paths = [
                _relative_to(front_path, output_dir),
                _relative_to(wrist_path, output_dir),
            ]
            record = {
                "id": sample_id,
                "image": image_paths,
                "state_info": state_info,
                "messages": [
                    {
                        "role": "system",
                        "content": _system_prompt_for_task(task, args.system_prompt),
                    },
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "metadata": {
                    "task": task,
                    "source_pickle": str(pickle_path),
                    "rollout_index_for_task": rollout_index,
                    "frame_index": frame_idx,
                    "success": _jsonify(data.get("success")),
                    "action_type": _jsonify(data.get("action_type")),
                    "camera_map": {"front": "color_image2", "wrist": "color_image1"},
                    "coordinate_frames": COORDINATE_FRAMES,
                    "depth": depth_paths,
                },
            }
            new_records.append(record)
            existing_ids.add(sample_id)
            emitted_for_rollout += 1

        if emitted_for_rollout > 0:
            selected_per_task[task] += 1

        if all(selected_per_task[task] >= limit for task, limit in task_counts.items()):
            break

    records = [*existing_records, *new_records]
    outputs = _write_records(output_dir=output_dir, records=records, formats=args.format)
    manifest = {
        "name": "VLM data generator",
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "output_mode": args.output_mode,
        "formats": outputs,
        "num_samples": len(records),
        "num_existing_samples": len(existing_records),
        "num_new_samples": len(new_records),
        "requested_rollouts_per_task": task_counts,
        "selected_rollouts_per_task": dict(selected_per_task),
        "new_samples_per_task": dict(
            Counter(str(record["metadata"].get("task", "unknown")) for record in new_records)
        ),
        "samples_per_task": dict(
            Counter(str(record["metadata"].get("task", "unknown")) for record in records)
        ),
        "skipped": dict(skipped),
        "schema": {
            "images": ["front RGB PNG", "wrist RGB PNG"],
            "assistant_json_keys": [
                "skill",
                "target_point_2d",
                "target_point_3d",
            ],
            "state_info_base_keys": list(STATE_INFO_BASE_KEYS),
            "state_info_extra_keys": [
                *STATE_INFO_EXTRA_KEYS,
                "parts_poses",
            ],
            "camera_key_map": {"front": "color_image2", "wrist": "color_image1"},
            "coordinate_frames": COORDINATE_FRAMES,
            "depth_arrays": "Optional .npy paths under metadata.depth.",
        },
        "source": {
            "randomness": args.randomness,
            "suffix": args.suffix,
            "demo_outcome": args.demo_outcome,
            "input_dir": args.input_dir,
            "frame_stride": args.frame_stride,
            "max_frames_per_rollout": args.max_frames_per_rollout,
            "allow_legacy_eepose": args.allow_legacy_eepose,
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    outputs["manifest"] = str(manifest_path)
    return manifest


def _checkpoint_args(args: argparse.Namespace) -> list[str]:
    provided = [
        args.wt_path is not None,
        args.run_id is not None,
        args.sweep_id is not None,
        args.project_id is not None,
    ]
    if sum(provided) != 1:
        raise ValueError("Exactly one of --wt-path, --run-id, --sweep-id, or --project-id is required.")
    if args.wt_path is not None:
        return ["--wt-path", args.wt_path]
    if args.run_id is not None:
        return ["--run-id", args.run_id]
    if args.sweep_id is not None:
        return ["--sweep-id", args.sweep_id]
    return ["--project-id", args.project_id]


TASK_ROLLOUT_EVAL_CONFIG = {
    # Mirrored from /data/hy/gpu-snatcher/auto_data_preparation.sh.
    "one_leg": {"max_rollout_steps": 700, "rollout_after_success": 200},
    "round_table": {"max_rollout_steps": 1000, "rollout_after_success": 100},
    "lamp": {"max_rollout_steps": 1000, "rollout_after_success": 20},
}


def _task_rollout_eval_config(task: str) -> dict[str, int]:
    return TASK_ROLLOUT_EVAL_CONFIG.get(task, {})


def run_rollouts(args: argparse.Namespace, task_counts: Optional[dict[str, int]] = None) -> list[Path]:
    task_counts = task_counts or _parse_task_counts(args.task_rollout or args.tasks, args.rollouts_per_task)
    env = _ensure_data_dir_raw()
    rollout_roots: list[Path] = []
    demo_outcome = getattr(args, "demo_outcome", "success")

    for task, count in task_counts.items():
        task_eval_config = _task_rollout_eval_config(task)
        max_rollout_steps = (
            args.max_rollout_steps
            if args.max_rollout_steps is not None
            else task_eval_config.get("max_rollout_steps")
        )
        rollout_after_success = (
            args.rollout_after_success
            if args.rollout_after_success is not None
            else task_eval_config.get("rollout_after_success")
        )
        n_rollouts = args.n_envs if args.target_successes else count

        cmd = [
            sys.executable,
            "-m",
            "src.eval.evaluate_model",
            *_checkpoint_args(args),
            "--gpu",
            str(args.gpu),
            "--n-envs",
            str(args.n_envs),
            "--n-rollouts",
            str(n_rollouts),
            "--randomness",
            args.randomness,
            "-f",
            task,
            "--if-exists",
            args.if_exists,
            "--action-type",
            args.action_type,
            "--observation-space",
            "image",
            "--save-rollouts",
            "--save-depth-image",
            "--annotate-skill",
            "--output-only-pickle",
            "--max-saved-rollouts",
            str(count),
        ]
        if max_rollout_steps is not None:
            cmd.extend(["--max-rollout-steps", str(max_rollout_steps)])
        if args.target_successes:
            cmd.extend(["--target-successes", str(count)])
        if args.save_failures:
            cmd.append("--save-failures")
        if args.full_length_rollout:
            cmd.append("--full-length-rollout")
        if rollout_after_success is not None:
            cmd.extend(["--rollout-after-success", str(rollout_after_success)])
        if args.rollout_run_name:
            cmd.extend(["--rollout-suffix-model-name", args.rollout_run_name])
        if args.wandb:
            cmd.append("--wandb")
        if args.extra_eval_arg:
            cmd.extend(args.extra_eval_arg)

        print("Running rollout command:")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], env=env, check=True)

        rollout_root = (
            Path(env["DATA_DIR_RAW"])
            / "raw"
            / "diffik"
            / "sim"
            / task
            / "rollout"
            / args.randomness
            / "rgbd-only-skill"
        )
        if args.rollout_run_name:
            rollout_root = rollout_root / args.rollout_run_name
        rollout_roots.append(rollout_root / demo_outcome)

    return rollout_roots


def add_task_count_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-rollout",
        action="append",
        help="Task/count spec like one_leg=100. Can be repeated.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task names. Use with --rollouts-per-task.",
    )
    parser.add_argument(
        "--rollouts-per-task",
        type=int,
        default=None,
        help="Count applied to every task in --tasks.",
    )


def add_rollout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wt-path", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=3)
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument("--action-type", type=str, default="pos", choices=["delta", "pos", "relative"])
    parser.add_argument("--if-exists", type=str, default="append", choices=["skip", "overwrite", "append", "error"])
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--target-successes", action="store_true", help="Treat task counts as target successful rollouts.")
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--full-length-rollout", action="store_true")
    parser.add_argument("--rollout-after-success", type=int, default=None)
    parser.add_argument("--rollout-run-name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--extra-eval-arg",
        action="append",
        default=[],
        help="Extra single argument forwarded to src.eval.evaluate_model. Repeat for multiple args/values.",
    )


def add_convert_args(parser: argparse.ArgumentParser, include_source_args: bool = True) -> None:
    parser.add_argument("--input-dir", action="append", default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    if include_source_args:
        parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument("--suffix", type=str, default="rgbd-only-skill")
    parser.add_argument("--demo-outcome", type=str, default="success", choices=["success", "failure", "partial_success"])
    parser.add_argument("--format", type=str, default="both", choices=["both", "messages-jsonl", "sharegpt-json"])
    parser.add_argument(
        "--output-mode",
        type=str,
        default="error",
        choices=["error", "append", "overwrite"],
        help="How to handle an existing output dataset.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-rollout", type=int, default=0)
    parser.add_argument(
        "--allow-legacy-eepose",
        action="store_true",
        help=(
            "Allow converting old pickles without robot_state.ee_pos_sim. "
            "Without this flag, samples missing sim-local EE pose are skipped."
        ),
    )
    parser.add_argument("--no-save-depth-npy", dest="save_depth_npy", action="store_false")
    parser.set_defaults(save_depth_npy=True)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override the built-in task-specific system prompt for all samples.",
    )
    parser.add_argument("--user-prompt", type=str, default=DEFAULT_USER_PROMPT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VLM data generator",
        description="Generate rgbd-only-skill rollouts and convert them to VLM SFT data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollouts = subparsers.add_parser("rollouts", help="Batch-generate raw rollout pickles.")
    add_task_count_args(rollouts)
    add_rollout_args(rollouts)

    convert = subparsers.add_parser("convert", help="Convert rollout pickles to VLM SFT files.")
    add_task_count_args(convert)
    add_convert_args(convert)

    generate = subparsers.add_parser("generate", help="Run rollouts, then convert the new pickles.")
    add_task_count_args(generate)
    add_rollout_args(generate)
    add_convert_args(generate, include_source_args=False)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "rollouts":
        task_counts = _parse_task_counts(args.task_rollout or args.tasks, args.rollouts_per_task)
        roots = run_rollouts(args, task_counts=task_counts)
        print("Rollout roots:")
        for root in roots:
            print(root)
        return

    if args.command == "convert":
        manifest = convert_pickles_to_vlm_sft(args)
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
        return

    if args.command == "generate":
        task_counts = _parse_task_counts(args.task_rollout or args.tasks, args.rollouts_per_task)
        if not args.rollout_run_name:
            args.rollout_run_name = "vlm-data-generator-" + datetime.now().strftime("%Y%m%dT%H%M%S")
        roots = run_rollouts(args, task_counts=task_counts)
        args.input_dir = [str(root) for root in roots]
        manifest = convert_pickles_to_vlm_sft(args)
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
