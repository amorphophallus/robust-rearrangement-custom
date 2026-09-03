from __future__ import annotations

import argparse
import gzip
import hashlib
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

VALID_SKILLS = frozenset({"push", "pick", "place", "insert", "screw"})

STATE_INFO_BASE_KEYS = (
    "ee_pos",
    "ee_quat",
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
    "target_point_3d": "robot-base position in meters, same frame as state_info.base.ee_pos",
    "state_info.base.ee_pos": "robot-base position in meters",
    "state_info.base.ee_quat": "robot-base end-effector orientation quaternion [x, y, z, w]",
    "state_info.base.ee_pos_vel": "end-effector linear velocity from rollout robot_state",
    "state_info.base.ee_ori_vel": "end-effector angular velocity from rollout robot_state",
}
ROTATION_6D_KEY = "target_rotation_6d"
ROTATION_6D_DESCRIPTION = (
    "sim_local target orientation using the first two rows of the 3x3 rotation "
    "matrix flattened row-major as [r00, r01, r02, r10, r11, r12]; recover a "
    "proper SO(3) matrix with Gram--Schmidt orthogonalization"
)
POSE_COORDINATE_FRAMES = {
    **COORDINATE_FRAMES,
    ROTATION_6D_KEY: ROTATION_6D_DESCRIPTION,
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
    "target_point_3d and state_info.base.ee_pos are expressed in the same "
    "robot-base coordinate frame. The target point is the position component of the "
    "target end-effector pose for the current skill. Skill semantics: for push, the "
    "target point is the goal location where the object or part should be pushed; "
    "for pick, the target point is the grasp "
    "point on the object to be picked; for place, the target point is the desired "
    "release or placement point for the held object; for insert, the target point "
    "is the insertion target at the socket, opening, or mating location where the "
    "held part should be inserted; for screw, the target point is the grasp point "
    "on the object or part that should be rotated and tightened."
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
    '"target_point_3d": [0.460508, 0.000166, 0.015685]}'
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

POSE_BASE_SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT.replace(
        "predict the current skill and the next target point.",
        "predict the current skill, the next target point, and the target orientation.",
    )
    + " target_rotation_6d is the target end-effector orientation in the same "
    "sim_local frame, encoded by the first two rows of its rotation matrix in "
    "row-major order: [r00, r01, r02, r10, r11, r12]."
)
POSE_TASK_SYSTEM_PROMPTS = {
    task: prompt.replace(BASE_SYSTEM_PROMPT, POSE_BASE_SYSTEM_PROMPT, 1)
    for task, prompt in TASK_SYSTEM_PROMPTS.items()
}
POSE_OUTPUT_JSON_EXAMPLE = (
    '{"skill": "pick", "target_point_2d": [160.0, 153.0], '
    '"target_point_3d": [0.160508, 0.000166, 0.430685], '
    '"target_rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}'
)
POSE_DEFAULT_USER_PROMPT = (
    "This is the front camera image:\n"
    "<image>\n"
    "This is the wrist camera image:\n"
    "<image>\n"
    "This is the robot proprioceptive state information:\n"
    f"{STATE_INFO_PLACEHOLDER}\n"
    "Please analyze the images and state information, then provide the current skill, "
    "target point, and target orientation. Return the answer in JSON format exactly "
    f"like this example: {POSE_OUTPUT_JSON_EXAMPLE}"
)

# Ver2 text-SFT contract. Unlike the enrichment wire format above, the
# deployed assistant response intentionally omits target_point_3d: production
# reconstructs translation from the predicted front pixel and sensor depth.
POSE_VER2_BASE_SYSTEM_PROMPT = POSE_BASE_SYSTEM_PROMPT.replace(
    "target_point_3d and state_info.base.ee_pos are expressed in the same "
    "robot-base coordinate frame. ",
    "",
)
POSE_VER2_TASK_SYSTEM_PROMPTS = {
    task: prompt.replace(POSE_BASE_SYSTEM_PROMPT, POSE_VER2_BASE_SYSTEM_PROMPT, 1)
    for task, prompt in POSE_TASK_SYSTEM_PROMPTS.items()
}
POSE_VER2_OUTPUT_JSON_EXAMPLE = (
    '{"skill": "pick", "target_point_2d": [160.0, 153.0], '
    '"target_rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}'
)
POSE_VER2_DEFAULT_USER_PROMPT = (
    "This is the front camera image:\n"
    "<image>\n"
    "This is the wrist camera image:\n"
    "<image>\n"
    "This is the robot proprioceptive state information:\n"
    f"{STATE_INFO_PLACEHOLDER}\n"
    "Please analyze the images and state information, then provide the current skill, "
    "target point, and target orientation. Return the answer in JSON format exactly "
    f"like this example: {POSE_VER2_OUTPUT_JSON_EXAMPLE}"
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


def _pose_system_prompt_for_task(task: Optional[str]) -> str:
    return POSE_TASK_SYSTEM_PROMPTS.get(str(task), POSE_BASE_SYSTEM_PROMPT)


def _has_robot_base_eepose(robot_state: Any) -> bool:
    return (
        isinstance(robot_state, dict)
        and robot_state.get("ee_pos") is not None
        and robot_state.get("ee_quat") is not None
    )


def _target_point_2d_payload(guidance_point_2d: Any) -> Any:
    if not isinstance(guidance_point_2d, dict):
        return None
    return _jsonify(guidance_point_2d.get("color_image2"))


def _numeric_vector_error(value: Any, *, field: str, length: int) -> Optional[str]:
    if value is None:
        return f"{field}_null"
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return f"{field}_shape"
    if any(
        isinstance(component, bool)
        or not isinstance(component, (int, float))
        or not np.isfinite(component)
        for component in value
    ):
        return f"{field}_non_finite"
    return None


def _rotation_6d_to_matrix(rotation_6d: Any) -> np.ndarray:
    values = np.asarray(rotation_6d, dtype=np.float64).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("rotation_6d must contain six finite values")
    first = values[:3]
    second = values[3:]
    first_norm = np.linalg.norm(first)
    if first_norm <= 1e-12:
        raise ValueError("rotation_6d first row is degenerate")
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if second_norm <= 1e-12:
        raise ValueError("rotation_6d second row is degenerate")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=0)


def _target_rotation_6d_payload(
    guidance_pose: Any,
    *,
    target_point_3d: Any,
    clean_guidance_pose: Any = None,
    orthogonality_atol: float = 1e-4,
    translation_atol: float = 1e-5,
) -> tuple[Optional[list[float]], Optional[str], dict[str, float]]:
    """Validate a scripted guidance pose and return its Zhou-style 6D rotation."""
    if guidance_pose is None:
        return None, "guidance_pose_null", {}
    try:
        pose = np.asarray(guidance_pose, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "guidance_pose_non_numeric", {}
    if pose.shape != (4, 4):
        return None, "guidance_pose_shape", {}
    if not np.isfinite(pose).all():
        return None, "guidance_pose_non_finite", {}
    if not np.allclose(
        pose[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=orthogonality_atol,
        rtol=0.0,
    ):
        return None, "guidance_pose_last_row", {}

    rotation = pose[:3, :3]
    orthogonality_error = float(
        np.max(np.abs(rotation @ rotation.T - np.eye(3)))
    )
    determinant = float(np.linalg.det(rotation))
    metrics = {
        "orthogonality_max_error": orthogonality_error,
        "determinant": determinant,
    }
    if orthogonality_error > orthogonality_atol:
        return None, "guidance_pose_rotation_not_orthogonal", metrics
    if abs(determinant - 1.0) > orthogonality_atol:
        return None, "guidance_pose_rotation_bad_determinant", metrics

    try:
        point = np.asarray(target_point_3d, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None, "target_point_3d_non_numeric", metrics
    if point.shape != (3,) or not np.isfinite(point).all():
        return None, "target_point_3d_invalid", metrics
    translation_error = float(np.max(np.abs(pose[:3, 3] - point)))
    metrics["translation_max_error_m"] = translation_error
    if translation_error > translation_atol:
        return None, "guidance_pose_translation_mismatch", metrics

    if clean_guidance_pose is not None:
        clean = np.asarray(clean_guidance_pose, dtype=np.float64)
        if clean.shape != (4, 4) or not np.isfinite(clean).all():
            return None, "guidance_pose_clean_invalid", metrics
        clean_error = float(np.max(np.abs(clean - pose)))
        metrics["clean_pose_max_error"] = clean_error
        if clean_error > translation_atol:
            return None, "guidance_pose_not_clean_scripted_gt", metrics

    rotation_6d = rotation[:2, :].reshape(6)
    reconstructed = _rotation_6d_to_matrix(rotation_6d)
    reconstruction_error = float(np.max(np.abs(reconstructed - rotation)))
    metrics["roundtrip_max_error"] = reconstruction_error
    if reconstruction_error > orthogonality_atol:
        return None, "rotation_6d_roundtrip_mismatch", metrics
    return _jsonify(rotation_6d), None, metrics


def _supervision_error(
    payload: Any,
    *,
    front_image_shape: Optional[tuple[int, ...]] = None,
) -> Optional[str]:
    """Return a stable reason when an assistant label must not enter SFT data."""
    if not isinstance(payload, dict):
        return "assistant_not_object"

    skill = payload.get("skill")
    if skill is None:
        return "skill_null"
    if not isinstance(skill, str) or skill not in VALID_SKILLS:
        return "skill_invalid"

    error = _numeric_vector_error(
        payload.get("target_point_2d"),
        field="target_point_2d",
        length=2,
    )
    if error:
        return error

    point_2d = payload["target_point_2d"]
    if front_image_shape is not None:
        if len(front_image_shape) < 2:
            return "front_image_shape"
        height, width = front_image_shape[:2]
        u, v = point_2d
        if not (0 <= u < width and 0 <= v < height):
            return "target_point_2d_out_of_frame"

    return _numeric_vector_error(
        payload.get("target_point_3d"),
        field="target_point_3d",
        length=3,
    ) or (
        _numeric_vector_error(
            payload.get(ROTATION_6D_KEY),
            field=ROTATION_6D_KEY,
            length=6,
        )
        if ROTATION_6D_KEY in payload
        else None
    )


def _assistant_json_from_text(text: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not isinstance(text, str) or not text.strip():
        return None, "assistant_text_missing"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "assistant_json_invalid"
    error = _supervision_error(payload)
    if error:
        return None, error
    return payload, None


def _record_supervision_error(record: dict[str, Any]) -> Optional[str]:
    _, _, assistant_text = _message_content(record)
    _, error = _assistant_json_from_text(assistant_text)
    return error


def _filter_invalid_records(
    records: Iterable[dict[str, Any]],
    *,
    skipped: Counter,
    prefix: str,
) -> list[dict[str, Any]]:
    valid_records = []
    for record in records:
        error = _record_supervision_error(record)
        if error:
            skipped[f"{prefix}:{error}"] += 1
            continue
        valid_records.append(record)
    return valid_records


def _validate_output_records(records: Iterable[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Output record {index} has no valid id.")
        if sample_id in seen_ids:
            raise ValueError(f"Output dataset contains duplicate id: {sample_id}")
        seen_ids.add(sample_id)

        images = record.get("image")
        if (
            not isinstance(images, list)
            or len(images) != 2
            or any(not isinstance(path, str) or not path for path in images)
        ):
            raise ValueError(f"Output record {sample_id} must reference two images.")

        error = _record_supervision_error(record)
        if error:
            raise ValueError(
                f"Output record {sample_id} has invalid supervision: {error}"
            )


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
    if base["ee_pos"] is None and robot_state.get("ee_pos_sim") is not None:
        base["ee_pos"] = _jsonify(robot_state.get("ee_pos_sim"))
    if base["ee_quat"] is None and robot_state.get("ee_quat_sim") is not None:
        base["ee_quat"] = _jsonify(robot_state.get("ee_quat_sim"))
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


def _guidance_frame_for_trajectory(data: dict[str, Any]) -> str:
    value = data.get("guidance_frame")
    if value is None:
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            annotation = metadata.get("real_skill_annotation")
            if isinstance(annotation, dict):
                value = annotation.get("target_point_frame")
    if value is not None:
        normalized = str(value).strip().lower().replace("_", "-")
        if normalized in {"robot-base", "sim-local"}:
            return normalized
    observations = data.get("observations")
    sample = observations[0] if isinstance(observations, list) and observations else {}
    state = sample.get("robot_state", {}) if isinstance(sample, dict) else {}
    return "sim-local" if isinstance(state, dict) and "ee_pos_sim" in state else "robot-base"


def _guidance_point_robot_base(obs: dict[str, Any], guidance_frame: str):
    point = obs.get("guidance_point")
    if point is None or guidance_frame == "robot-base":
        return point
    state = obs.get("robot_state")
    if not isinstance(state, dict) or "ee_pos" not in state or "ee_pos_sim" not in state:
        return None
    offset = np.asarray(state["ee_pos_sim"], dtype=np.float32) - np.asarray(
        state["ee_pos"], dtype=np.float32
    )
    return np.asarray(point, dtype=np.float32) - offset


def _assistant_payload(
    obs: dict[str, Any],
    task: Optional[str],
    *,
    guidance_frame: str = "robot-base",
) -> dict[str, Any]:
    return {
        "skill": _jsonify(obs.get("skill")),
        "target_point_2d": _target_point_2d_payload(obs.get("guidance_point_2d")),
        "target_point_3d": _jsonify(
            _guidance_point_robot_base(obs, guidance_frame)
        ),
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


def _message_content(record: dict[str, Any]) -> tuple[str, str, str]:
    system_text = ""
    user_text = ""
    assistant_text = ""
    for message in record.get("messages", []):
        role = message.get("role")
        if role == "system" and not system_text:
            system_text = message.get("content", "")
        elif role == "user" and not user_text:
            user_text = message.get("content", "")
        elif role == "assistant" and not assistant_text:
            assistant_text = message.get("content", "")
    return system_text, user_text, assistant_text


def _state_info_for_prompt(state_info: Any, mode: str) -> Any:
    if mode == "placeholder":
        return STATE_INFO_PLACEHOLDER
    if not isinstance(state_info, dict):
        return None
    if mode == "base":
        return {"base": state_info.get("base")}
    if mode == "base-extra":
        return {
            "base": state_info.get("base"),
            "extra": state_info.get("extra"),
        }
    raise ValueError(f"Unsupported state info mode: {mode}")


def _replace_state_info_placeholder(user_text: str, state_info: Any, mode: str) -> str:
    if mode == "placeholder":
        return user_text
    prompt_state = _state_info_for_prompt(state_info, mode)
    prompt_state_text = json.dumps(
        prompt_state,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return user_text.replace(STATE_INFO_PLACEHOLDER, prompt_state_text)


def _llamafactory_item_from_record(
    record: dict[str, Any],
    *,
    state_mode: str,
    system_prompt_override: Optional[str] = None,
) -> dict[str, Any]:
    system_text, user_text, assistant_text = _message_content(record)
    task = record.get("metadata", {}).get("task")
    if system_prompt_override is not None:
        system_text = _system_prompt_for_task(task, system_prompt_override)
    elif not system_text:
        system_text = _system_prompt_for_task(task, None)
    return {
        "id": record["id"],
        "images": record["image"],
        "state_info": record.get("state_info"),
        "system": system_text,
        "conversations": [
            {
                "from": "human",
                "value": _replace_state_info_placeholder(
                    user_text,
                    record.get("state_info"),
                    state_mode,
                ),
            },
            {"from": "gpt", "value": assistant_text},
        ],
        "metadata": record["metadata"],
    }


def _llamafactory_item_from_sharegpt(
    item: dict[str, Any],
    *,
    state_mode: str,
    system_prompt_override: Optional[str] = None,
) -> dict[str, Any]:
    task = item.get("metadata", {}).get("task")
    conversations = item.get("conversations", [])
    converted_conversations = []
    for turn in conversations:
        converted_turn = dict(turn)
        if converted_turn.get("from") == "human":
            converted_turn["value"] = _replace_state_info_placeholder(
                str(converted_turn.get("value", "")),
                item.get("state_info"),
                state_mode,
            )
        converted_conversations.append(converted_turn)

    output = {
        "id": item.get("id"),
        "images": item.get("images", item.get("image", [])),
        "state_info": item.get("state_info"),
        "system": _system_prompt_for_task(task, system_prompt_override),
        "conversations": converted_conversations,
        "metadata": item.get("metadata", {}),
    }
    return output


def _write_json(path: Path, payload: Any, *, pretty: bool = False) -> None:
    with path.open("w") as f:
        json.dump(
            payload,
            f,
            indent=2 if pretty else None,
            ensure_ascii=False,
            separators=None if pretty else (",", ":"),
        )


def _write_records(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    formats: str,
    llamafactory_state_mode: str,
) -> dict[str, str]:
    _validate_output_records(records)
    write_messages = formats in ("all", "both", "messages-jsonl")
    write_sharegpt = formats in ("all", "both", "sharegpt-json")
    write_llamafactory = formats in ("all", "llamafactory-json")

    outputs: dict[str, str] = {}
    if write_messages:
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

    if write_sharegpt:
        sharegpt_path = output_dir / "qwen_llava_sharegpt.json"
        sharegpt_records = []
        for record in records:
            _, user_text, assistant_text = _message_content(record)
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
        _write_json(sharegpt_path, sharegpt_records, pretty=True)
        outputs["sharegpt_json"] = str(sharegpt_path)

    if write_llamafactory:
        suffix = llamafactory_state_mode.replace("-", "_")
        llamafactory_path = output_dir / f"llamafactory_{suffix}.json"
        llamafactory_records = [
            _llamafactory_item_from_record(
                record,
                state_mode=llamafactory_state_mode,
            )
            for record in records
        ]
        _write_json(llamafactory_path, llamafactory_records)
        outputs["llamafactory_json"] = str(llamafactory_path)
        dataset_info_path = output_dir / f"llamafactory_{suffix}_dataset_info.json"
        dataset_info = _llamafactory_dataset_info(
            llamafactory_path.name,
            f"rr_vlm_{suffix}",
        )
        _write_json(dataset_info_path, dataset_info, pretty=True)
        outputs["llamafactory_dataset_info"] = str(dataset_info_path)

    return outputs


def _llamafactory_dataset_info(file_name: str, dataset_name: str) -> dict[str, Any]:
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
                "system": "system",
            },
        }
    }


def convert_sharegpt_to_llamafactory(args: argparse.Namespace) -> dict[str, Any]:
    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    dataset_info_file = (
        Path(args.dataset_info_file).expanduser().resolve()
        if args.dataset_info_file
        else output_file.with_name(output_file.stem + "_dataset_info.json")
    )
    payload = json.loads(input_file.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {input_file}.")

    converted = []
    skipped = Counter()
    for item in payload:
        conversations = item.get("conversations", [])
        assistant_text = next(
            (
                turn.get("value", "")
                for turn in conversations
                if turn.get("from") == "gpt"
            ),
            "",
        )
        _, error = _assistant_json_from_text(assistant_text)
        if error:
            skipped[f"invalid_supervision:{error}"] += 1
            continue
        converted.append(
            _llamafactory_item_from_sharegpt(
                item,
                state_mode=args.llamafactory_state_mode,
                system_prompt_override=args.system_prompt,
            )
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_file, converted)

    dataset_info = _llamafactory_dataset_info(
        output_file.name,
        args.dataset_name,
    )
    _write_json(dataset_info_file, dataset_info, pretty=True)
    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "dataset_info_file": str(dataset_info_file),
        "num_input_samples": len(payload),
        "num_samples": len(converted),
        "num_skipped_invalid": sum(skipped.values()),
        "skipped": dict(skipped),
        "llamafactory_state_mode": args.llamafactory_state_mode,
        "dataset_info": dataset_info,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_pickle_contract(data: Any) -> tuple[Optional[str], bool]:
    if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
        return "invalid_pickle_structure", False
    annotation_source = data.get("annotation_source")
    if annotation_source not in (None, "scripted"):
        return f"unexpected_annotation_source:{annotation_source}", False
    if any(str(key).lower().startswith("vlm") for key in data):
        return "vlm_metadata_present", False
    observations = data["observations"]
    for observation_index, observation in enumerate(observations):
        if isinstance(observation, dict) and any(
            str(key).lower().startswith("vlm") for key in observation
        ):
            return f"vlm_observation_metadata_present:{observation_index}", False
    return None, annotation_source is None


def _pose_record_from_existing(
    source_record: dict[str, Any],
    *,
    observation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    metadata = dict(source_record.get("metadata", {}))
    task = _normalize_task(str(metadata.get("task", "")))
    if not task:
        raise ValueError(f"Record {source_record.get('id')} has no task metadata")

    _, old_error = _assistant_json_from_text(
        _message_content(
            {
                "messages": source_record.get("messages", []),
            }
        )[2]
    )
    if old_error:
        raise ValueError(
            f"Record {source_record.get('id')} has invalid source supervision: {old_error}"
        )
    old_assistant = json.loads(source_record["messages"][-1]["content"])
    local_assistant = _assistant_payload(observation, task=task)
    if old_assistant != local_assistant:
        raise ValueError(
            f"Record {source_record.get('id')} does not match its source pickle assistant label"
        )
    local_state = _state_info_payload(observation)
    if source_record.get("state_info") != local_state:
        raise ValueError(
            f"Record {source_record.get('id')} does not match its source pickle state_info"
        )

    rotation_6d, pose_error, metrics = _target_rotation_6d_payload(
        observation.get("guidance_pose"),
        target_point_3d=observation.get("guidance_point"),
        clean_guidance_pose=observation.get("guidance_pose_clean"),
    )
    if pose_error:
        raise ValueError(
            f"Record {source_record.get('id')} has invalid scripted target pose: {pose_error}"
        )

    enriched_assistant = dict(old_assistant)
    enriched_assistant[ROTATION_6D_KEY] = rotation_6d
    supervision_error = _supervision_error(enriched_assistant)
    if supervision_error:
        raise ValueError(
            f"Record {source_record.get('id')} has invalid enriched supervision: "
            f"{supervision_error}"
        )

    metadata["annotation_source"] = "scripted"
    metadata["coordinate_frames"] = POSE_COORDINATE_FRAMES
    metadata["target_rotation_representation"] = {
        "name": "rotation_6d",
        "order": ["r00", "r01", "r02", "r10", "r11", "r12"],
        "decode": "Gram--Schmidt rows, then cross product for the third row",
    }
    images = source_record.get("images", source_record.get("image", []))
    return (
        {
            "id": source_record["id"],
            "image": images,
            "state_info": source_record.get("state_info"),
            "messages": [
                {
                    "role": "system",
                    "content": _pose_system_prompt_for_task(task),
                },
                {"role": "user", "content": POSE_DEFAULT_USER_PROMPT},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        enriched_assistant,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "metadata": metadata,
        },
        metrics,
    )


def enrich_existing_dataset_with_rotation_6d(args: argparse.Namespace) -> dict[str, Any]:
    """Add scripted Rotation6D labels to an existing messages JSONL index.

    The source media are intentionally not read or rewritten. Existing sample IDs,
    image/depth references, state, skill, and point labels must match the local raw
    pickle before an enriched record is emitted.
    """
    input_messages = Path(args.input_messages).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_manifest_path = (
        Path(args.source_manifest).expanduser().resolve()
        if args.source_manifest
        else input_messages.with_name("manifest.json")
    )
    if not input_messages.is_file():
        raise FileNotFoundError(input_messages)
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    if args.annotation_source != "scripted":
        raise ValueError("Rotation6D enrichment currently requires --annotation-source scripted")

    output_paths = {
        "messages_jsonl": output_dir / "messages.jsonl",
        "sharegpt_json": output_dir / "qwen_llava_sharegpt.json",
        "llamafactory_json": output_dir / "llamafactory_base.json",
        "preview_jsonl": output_dir / "preview" / "llamafactory_preview.jsonl",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and args.output_mode == "error":
        raise FileExistsError(
            "Rotation6D output already exists:\n"
            + "\n".join(f"  {path}" for path in existing)
        )
    if args.output_mode == "overwrite":
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths["preview_jsonl"].parent.mkdir(parents=True, exist_ok=True)

    temporary_paths = {
        name: path.with_name(path.name + ".tmp")
        for name, path in output_paths.items()
    }
    for path in temporary_paths.values():
        if path.exists():
            path.unlink()

    source_manifest = json.loads(source_manifest_path.read_text())
    expected_task_counts = source_manifest.get("samples_per_task", {})
    source_pickles: set[Path] = set()
    closed_pickles: set[Path] = set()
    task_counts: Counter[str] = Counter()
    legacy_missing_annotation_source = 0
    max_metrics = {
        "orthogonality_max_error": 0.0,
        "translation_max_error_m": 0.0,
        "clean_pose_max_error": 0.0,
        "roundtrip_max_error": 0.0,
    }
    determinant_min = float("inf")
    determinant_max = float("-inf")
    num_records = 0
    seen_ids: set[str] = set()
    current_pickle: Optional[Path] = None
    current_data: Optional[dict[str, Any]] = None
    num_loaded_pickles = 0
    source_files = {}

    try:
        source_files["messages_jsonl"] = temporary_paths["messages_jsonl"].open("w")
        source_files["sharegpt_json"] = temporary_paths["sharegpt_json"].open("w")
        source_files["llamafactory_json"] = temporary_paths["llamafactory_json"].open("w")
        source_files["preview_jsonl"] = temporary_paths["preview_jsonl"].open("w")
        source_files["sharegpt_json"].write("[")
        source_files["llamafactory_json"].write("[")

        with input_messages.open() as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    source_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {input_messages}:{line_no}: {exc}"
                    ) from exc
                metadata = source_record.get("metadata", {})
                source_pickle_value = metadata.get("source_pickle")
                if not isinstance(source_pickle_value, str) or not source_pickle_value:
                    raise ValueError(
                        f"Record {source_record.get('id')} has no source_pickle metadata"
                    )
                source_pickle = Path(source_pickle_value).expanduser().resolve()
                if source_pickle != current_pickle:
                    if source_pickle in closed_pickles:
                        raise ValueError(
                            f"Source pickle records are not contiguous: {source_pickle}"
                        )
                    if current_pickle is not None:
                        closed_pickles.add(current_pickle)
                    if not source_pickle.is_file():
                        raise FileNotFoundError(source_pickle)
                    current_pickle = source_pickle
                    source_pickles.add(source_pickle)
                    loaded = _load_pickle(source_pickle)
                    contract_error, missing_source = _source_pickle_contract(loaded)
                    if contract_error:
                        raise ValueError(f"{source_pickle}: {contract_error}")
                    if missing_source:
                        legacy_missing_annotation_source += 1
                    current_data = loaded
                    num_loaded_pickles += 1
                    if num_loaded_pickles == 1 or num_loaded_pickles % 10 == 0:
                        print(
                            "[enrich-rotation6d] "
                            f"loaded {num_loaded_pickles} source pickles; "
                            f"emitted {num_records} samples",
                            flush=True,
                        )

                assert current_data is not None
                frame_index = metadata.get("frame_index")
                if not isinstance(frame_index, int):
                    raise ValueError(
                        f"Record {source_record.get('id')} has invalid frame_index"
                    )
                observations = current_data["observations"]
                if not 0 <= frame_index < len(observations):
                    raise IndexError(
                        f"Record {source_record.get('id')} frame {frame_index} is out of range"
                    )
                observation = observations[frame_index]
                if not isinstance(observation, dict):
                    raise ValueError(
                        f"Record {source_record.get('id')} observation is not a mapping"
                    )
                enriched_record, pose_metrics = _pose_record_from_existing(
                    source_record,
                    observation=observation,
                )
                if enriched_record["id"] in seen_ids:
                    raise ValueError(f"Duplicate sample ID: {enriched_record['id']}")
                seen_ids.add(enriched_record["id"])
                _validate_output_records([enriched_record])
                for key in max_metrics:
                    max_metrics[key] = max(
                        max_metrics[key],
                        float(pose_metrics.get(key, 0.0)),
                    )
                determinant = float(pose_metrics["determinant"])
                determinant_min = min(determinant_min, determinant)
                determinant_max = max(determinant_max, determinant)

                task = str(enriched_record["metadata"]["task"])
                task_counts[task] += 1
                messages_item = {
                    "id": enriched_record["id"],
                    "images": enriched_record["image"],
                    "state_info": enriched_record.get("state_info"),
                    "messages": enriched_record["messages"],
                    "metadata": enriched_record["metadata"],
                }
                _, user_text, assistant_text = _message_content(enriched_record)
                sharegpt_item = {
                    "id": enriched_record["id"],
                    "image": enriched_record["image"],
                    "state_info": enriched_record.get("state_info"),
                    "conversations": [
                        {"from": "human", "value": user_text},
                        {"from": "gpt", "value": assistant_text},
                    ],
                    "metadata": enriched_record["metadata"],
                }
                llamafactory_item = _llamafactory_item_from_record(
                    enriched_record,
                    state_mode="base",
                )

                source_files["messages_jsonl"].write(
                    json.dumps(messages_item, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                if num_records:
                    source_files["sharegpt_json"].write(",")
                    source_files["llamafactory_json"].write(",")
                source_files["sharegpt_json"].write(
                    json.dumps(sharegpt_item, ensure_ascii=False, separators=(",", ":"))
                )
                source_files["llamafactory_json"].write(
                    json.dumps(llamafactory_item, ensure_ascii=False, separators=(",", ":"))
                )
                if num_records < args.preview_samples:
                    source_files["preview_jsonl"].write(
                        json.dumps(
                            llamafactory_item,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                num_records += 1
                if num_records % 5000 == 0:
                    print(
                        f"[enrich-rotation6d] emitted {num_records} samples",
                        flush=True,
                    )

        source_files["sharegpt_json"].write("]")
        source_files["llamafactory_json"].write("]")
    finally:
        for stream in source_files.values():
            stream.close()

    if args.expected_samples is not None and num_records != args.expected_samples:
        raise ValueError(
            f"Enriched {num_records} records; expected {args.expected_samples}"
        )
    if expected_task_counts and dict(task_counts) != expected_task_counts:
        raise ValueError(
            f"Task counts changed: enriched={dict(task_counts)} "
            f"source_manifest={expected_task_counts}"
        )

    expected_pickle_paths: set[Path] = set()
    for directory in args.expected_pickle_dir or []:
        expected_pickle_paths.update(
            path.resolve()
            for path in Path(directory).expanduser().resolve().glob("*.pkl*")
        )
    if expected_pickle_paths and source_pickles != expected_pickle_paths:
        missing = sorted(str(path) for path in source_pickles - expected_pickle_paths)
        extra = sorted(str(path) for path in expected_pickle_paths - source_pickles)
        raise ValueError(
            "Source pickle set differs from expected campaign directories: "
            f"missing_expected={missing[:5]} extra_expected={extra[:5]}"
        )

    for name, path in output_paths.items():
        temporary_paths[name].replace(path)

    dataset_info_path = output_dir / "llamafactory_base_dataset_info.json"
    _write_json(
        dataset_info_path,
        _llamafactory_dataset_info("llamafactory_base.json", "rr_vlm_base"),
        pretty=True,
    )
    output_hashes = {
        path.relative_to(output_dir).as_posix(): _sha256_file(path)
        for path in [*output_paths.values(), dataset_info_path]
    }
    source_path_digest = hashlib.sha256(
        "\n".join(sorted(str(path) for path in source_pickles)).encode()
    ).hexdigest()
    audit = {
        "created_at": datetime.now().isoformat(),
        "annotation_source": "scripted",
        "source_revision": args.source_revision,
        "source_messages": str(input_messages),
        "num_samples": num_records,
        "samples_per_task": dict(task_counts),
        "num_source_pickles": len(source_pickles),
        "source_pickle_paths_sha256": source_path_digest,
        "source_pickle_set_matches_expected_campaigns": bool(expected_pickle_paths),
        "legacy_pickles_missing_top_level_annotation_source": (
            legacy_missing_annotation_source
        ),
        "old_fields_verified_unchanged": [
            "sample_id",
            "images",
            "state_info",
            "skill",
            "target_point_2d",
            "target_point_3d",
        ],
        "rotation_6d": {
            "field": ROTATION_6D_KEY,
            "description": ROTATION_6D_DESCRIPTION,
            "max_errors": max_metrics,
            "determinant_min": determinant_min,
            "determinant_max": determinant_max,
        },
        "media_policy": "Existing image/depth tar archives are not rewritten or uploaded.",
        "output_sha256": output_hashes,
    }
    audit_path = output_dir / "rotation6d_enrichment_audit_20260831.json"
    _write_json(audit_path, audit, pretty=True)
    manifest = {
        **source_manifest,
        "updated_at": datetime.now().isoformat(),
        "annotation_source": "scripted",
        "source_revision": args.source_revision,
        "num_samples": num_records,
        "samples_per_task": dict(task_counts),
        "schema": {
            **source_manifest.get("schema", {}),
            "assistant_json_keys": [
                "skill",
                "target_point_2d",
                "target_point_3d",
                ROTATION_6D_KEY,
            ],
            "coordinate_frames": POSE_COORDINATE_FRAMES,
        },
        "rotation6d_enrichment": {
            "audit": audit_path.name,
            "representation": ROTATION_6D_DESCRIPTION,
            "media_archives_unchanged": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest, pretty=True)
    return {
        "output_dir": str(output_dir),
        "num_samples": num_records,
        "samples_per_task": dict(task_counts),
        "num_source_pickles": len(source_pickles),
        "output_sha256": output_hashes,
        "manifest": str(manifest_path),
        "audit": str(audit_path),
    }


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


def _read_existing_llamafactory_json(path: Path) -> list[dict[str, Any]]:
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
        messages = []
        if item.get("system"):
            messages.append({"role": "system", "content": item.get("system", "")})
        messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        records.append(
            {
                "id": item["id"],
                "image": item.get("images", item.get("image", [])),
                "state_info": item.get("state_info"),
                "messages": messages,
                "metadata": item.get("metadata", {}),
            }
        )
    return records


def _read_existing_records(output_dir: Path) -> list[dict[str, Any]]:
    messages_path = output_dir / "messages.jsonl"
    if messages_path.exists():
        return _read_existing_messages_jsonl(messages_path)
    sharegpt_path = output_dir / "qwen_llava_sharegpt.json"
    if sharegpt_path.exists():
        return _read_existing_sharegpt_json(sharegpt_path)
    for llamafactory_path in sorted(output_dir.glob("llamafactory_*.json")):
        if llamafactory_path.name.endswith("_dataset_info.json"):
            continue
        return _read_existing_llamafactory_json(llamafactory_path)
    return []


def _prepare_output_dir(output_dir: Path, output_mode: str) -> list[dict[str, Any]]:
    generated_paths = [
        output_dir / "messages.jsonl",
        output_dir / "qwen_llava_sharegpt.json",
        output_dir / "manifest.json",
        output_dir / "images",
        output_dir / "depth",
    ]
    generated_paths.extend(output_dir.glob("llamafactory_*.json"))
    generated_paths.extend(output_dir.glob("llamafactory_*_dataset_info.json"))
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
    skipped = Counter()
    existing_input_records = _prepare_output_dir(output_dir, args.output_mode)
    existing_records = _filter_invalid_records(
        existing_input_records,
        skipped=skipped,
        prefix="invalid_existing_supervision",
    )
    existing_ids = {str(record["id"]) for record in existing_records}

    pickle_paths = gather_pickle_paths(args, task_counts)
    selected_per_task: dict[str, int] = defaultdict(int)
    new_records: list[dict[str, Any]] = []

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
        guidance_frame = _guidance_frame_for_trajectory(data)

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
            if not args.allow_legacy_eepose and not _has_robot_base_eepose(
                obs.get("robot_state")
            ):
                skipped["missing_robot_base_eepose"] += 1
                continue

            try:
                front_image = _as_image_uint8(obs["color_image2"])
                wrist_image = _as_image_uint8(obs["color_image1"])
            except Exception as exc:
                skipped[f"image_error:{type(exc).__name__}"] += 1
                continue

            assistant_obj = _assistant_payload(
                obs,
                task=task,
                guidance_frame=guidance_frame,
            )
            supervision_error = _supervision_error(
                assistant_obj,
                front_image_shape=front_image.shape,
            )
            if supervision_error:
                skipped[f"invalid_supervision:{supervision_error}"] += 1
                continue

            sample_id = f"{task}_{rollout_index:05d}_{rollout_stem}_frame_{frame_idx:05d}"
            if sample_id in existing_ids:
                skipped["duplicate_sample_id"] += 1
                continue
            front_path = image_dir / task / f"{sample_id}_front.png"
            wrist_path = image_dir / task / f"{sample_id}_wrist.png"
            try:
                _save_png(front_image, front_path)
                _save_png(wrist_image, wrist_path)
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
    outputs = _write_records(
        output_dir=output_dir,
        records=records,
        formats=args.format,
        llamafactory_state_mode=args.llamafactory_state_mode,
    )
    manifest = {
        "name": "VLM data generator",
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "output_mode": args.output_mode,
        "formats": outputs,
        "num_samples": len(records),
        "num_existing_samples": len(existing_records),
        "num_existing_input_samples": len(existing_input_records),
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
        "validation": {
            "invalid_supervision_policy": "skip",
            "valid_skills": sorted(VALID_SKILLS),
            "required_non_null_fields": [
                "skill",
                "target_point_2d",
                "target_point_3d",
            ],
            "target_point_2d_checked_against_front_image_bounds": True,
            "final_output_validation": "strict",
        },
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
            "llamafactory_state_mode": args.llamafactory_state_mode,
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
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=[
            "both",
            "all",
            "messages-jsonl",
            "sharegpt-json",
            "llamafactory-json",
        ],
        help=(
            "Output dataset format. 'both' keeps the historical messages.jsonl + "
            "qwen_llava_sharegpt.json outputs; 'all' also writes LLaMAFactory JSON."
        ),
    )
    parser.add_argument(
        "--llamafactory-state-mode",
        type=str,
        default="base",
        choices=["placeholder", "base", "base-extra"],
        help=(
            "How to fill <state_info> when writing LLaMAFactory JSON. "
            "'base' is directly trainable with the core proprioceptive state."
        ),
    )
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
            "Allow converting old pickles without canonical robot_state.ee_pos/ee_quat. "
            "Without this flag, samples missing robot-base EE pose are skipped."
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


def add_llamafactory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-file",
        type=str,
        default="/data/hy/robust-rearrangement/data/processed/vlm/qwen_llava_sharegpt.json",
        help="Existing qwen/LLaVA ShareGPT JSON file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/data/hy/robust-rearrangement/data/processed/vlm/llamafactory_base.json",
        help="Output LLaMAFactory JSON file.",
    )
    parser.add_argument(
        "--dataset-info-file",
        type=str,
        default=None,
        help="Output dataset_info JSON snippet. Defaults next to --output-file.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="rr_vlm_base",
        help="Dataset key to use in the generated dataset_info JSON snippet.",
    )
    parser.add_argument(
        "--llamafactory-state-mode",
        type=str,
        default="base",
        choices=["placeholder", "base", "base-extra"],
        help="How to replace <state_info> in the human prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override built-in task-specific system prompts for all samples.",
    )


def add_rotation6d_enrichment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-messages",
        type=str,
        required=True,
        help="Current dataset messages.jsonl whose IDs and media references are preserved.",
    )
    parser.add_argument("--source-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--source-revision",
        type=str,
        required=True,
        help="Immutable ModelScope revision being enriched.",
    )
    parser.add_argument(
        "--annotation-source",
        type=str,
        required=True,
        choices=["scripted"],
    )
    parser.add_argument(
        "--expected-pickle-dir",
        action="append",
        default=[],
        help="Campaign success directory; repeat to require exact source-pickle set equality.",
    )
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--preview-samples", type=int, default=100)
    parser.add_argument(
        "--output-mode",
        type=str,
        default="error",
        choices=["error", "overwrite"],
    )


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

    llamafactory = subparsers.add_parser(
        "to-llamafactory",
        help="Convert qwen/LLaVA ShareGPT JSON to directly trainable LLaMAFactory JSON.",
    )
    add_llamafactory_args(llamafactory)

    enrichment = subparsers.add_parser(
        "enrich-rotation6d",
        help=(
            "Add scripted target_rotation_6d to an existing dataset index without "
            "rewriting RGB-D media."
        ),
    )
    add_rotation6d_enrichment_args(enrichment)
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

    if args.command == "to-llamafactory":
        result = convert_sharegpt_to_llamafactory(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return

    if args.command == "enrich-rotation6d":
        result = enrich_existing_dataset_with_rotation_6d(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
