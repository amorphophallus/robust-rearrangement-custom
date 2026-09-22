"""Single-window inspection dashboard for real-world policy inputs."""

from __future__ import annotations

import queue
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np


CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 900
_BACKGROUND = (24, 24, 24)
_PANEL_BACKGROUND = (34, 34, 34)
_TEXT = (230, 230, 230)
_MUTED = (165, 165, 165)
_ACCENT = (80, 210, 255)
VIDEO_FRAME_WIDTH = 1280
VIDEO_FRAME_HEIGHT = 960
_TABLETOP_PART_INDEX = 0
_TABLETOP_MESH_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "furniture"
    / "mesh"
    / "square_table"
    / "square_table_top.obj"
)
_TABLETOP_AXIS_LENGTH_M = 0.08


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _format_vector(value: Any, precision: int = 4) -> str:
    if value is None:
        return "none"
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return "[]"
    return "[" + ", ".join(f"{float(item):.{precision}f}" for item in array) + "]"


def _format_uv(value: Any) -> str:
    if value is None:
        return "none"
    array = np.asarray(value).reshape(-1)
    if array.size < 2:
        return _format_vector(array, precision=1)
    return f"({float(array[0]):.1f}, {float(array[1]):.1f})"


def _shape_and_dtype(value: Any) -> str:
    if value is None:
        return "missing"
    array = _to_numpy(value)
    return f"{tuple(array.shape)} {array.dtype}"


def _depth_stats(depth: Any) -> tuple[str, Optional[tuple[float, float]]]:
    if depth is None:
        return "missing", None
    array = _to_numpy(depth).astype(np.float32, copy=False)
    valid = array[np.isfinite(array) & (array > 0)]
    if valid.size == 0:
        return "no positive finite pixels", None
    p02, median, p98 = np.percentile(valid, [2, 50, 98])
    return (
        f"min/med/max={valid.min():.3f}/{median:.3f}/{valid.max():.3f} m",
        (float(p02), float(p98)),
    )


def _rgb_to_bgr(image: Any) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"RGB image must have HxWx3 shape, got {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if array.size and float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _depth_to_bgr(depth: Any) -> tuple[np.ndarray, str]:
    array = _to_numpy(depth).astype(np.float32, copy=False)
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"depth image must be 2-D, got {array.shape}")
    stats, limits = _depth_stats(array)
    normalized = np.zeros(array.shape, dtype=np.uint8)
    if limits is not None:
        low, high = limits
        if high <= low:
            high = low + 1e-6
        valid = np.isfinite(array) & (array > 0)
        scaled = (array - low) * (255.0 / (high - low))
        normalized[valid] = np.clip(scaled[valid], 0, 255).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(normalized, colormap)
    colored[~(np.isfinite(array) & (array > 0))] = 0
    return colored, stats


@lru_cache(maxsize=1)
def _tabletop_mesh_vertices() -> np.ndarray:
    vertices = []
    with _TABLETOP_MESH_PATH.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                values = line.split()
                vertices.append(
                    [float(values[1]), float(values[2]), float(values[3])]
                )
    if not vertices:
        raise ValueError(f"No CAD vertices found in {_TABLETOP_MESH_PATH}")
    return np.asarray(vertices, dtype=np.float64)


def _pose_vector_to_matrix(pose: Any) -> np.ndarray:
    vector = np.asarray(pose, dtype=np.float64).reshape(-1)
    if vector.shape != (7,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected finite xyz+xyzw pose, got {vector.shape}")
    x, y, z, w = vector[3:]
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-9:
        raise ValueError("Pose quaternion has zero norm")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = vector[:3]
    return matrix


def _front_projection(
    points_local: np.ndarray,
    observation: Mapping[str, Any],
    camera_info: Mapping[str, Any],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    debug = observation.get("real_annotation_debug") or {}
    effective_poses = debug.get("effective_part_poses_april") or {}
    if "square_table_top" in effective_poses:
        tabletop_pose = effective_poses["square_table_top"]
    else:
        parts_poses = np.asarray(
            observation.get("parts_poses"), dtype=np.float64
        ).reshape(-1, 7)
        if parts_poses.shape[0] <= _TABLETOP_PART_INDEX:
            raise ValueError("Observation has no tabletop pose")
        tabletop_pose = parts_poses[_TABLETOP_PART_INDEX]
    tabletop_pose_april = _pose_vector_to_matrix(tabletop_pose)
    camera_to_april = np.asarray(
        observation.get("camera_to_april"), dtype=np.float64
    )
    if camera_to_april.shape != (4, 4):
        raise ValueError("Observation camera_to_april must have shape (4, 4)")

    front = camera_info.get("front")
    if not isinstance(front, Mapping):
        raise ValueError("Camera metadata has no front camera")
    values = front.get("record_intrinsics", front.get("intrinsics"))
    if not isinstance(values, Mapping):
        raise ValueError("Front camera metadata has no intrinsics")
    intrinsics = np.asarray(
        [
            [values["fx"], 0.0, values.get("ppx", values.get("cx"))],
            [0.0, values["fy"], values.get("ppy", values.get("cy"))],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    record_width = float(values["width"])
    record_height = float(values["height"])

    points_h = np.concatenate(
        [points_local, np.ones((points_local.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_camera = (
        np.linalg.inv(camera_to_april) @ tabletop_pose_april @ points_h.T
    ).T[:, :3]
    visible = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-4)
    projected = np.full((points_local.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(visible):
        pixels_h = (intrinsics @ points_camera[visible].T).T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        height, width = image_shape
        pixels[:, 0] *= width / record_width
        pixels[:, 1] *= height / record_height
        projected[visible] = pixels
    return projected, visible


def _draw_tabletop_cad_overlay(
    image: np.ndarray,
    observation: Mapping[str, Any],
    camera_info: Optional[Mapping[str, Any]],
) -> np.ndarray:
    """Draw the tabletop pose used by the FSM without mutating policy input."""

    output = image.copy()
    if camera_info is None:
        return output
    try:
        mesh_pixels, visible = _front_projection(
            _tabletop_mesh_vertices(),
            observation,
            camera_info,
            output.shape[:2],
        )
        finite_pixels = mesh_pixels[visible]
        if finite_pixels.shape[0] < 3:
            return output
        hull = cv2.convexHull(
            np.round(finite_pixels).astype(np.int32).reshape(-1, 1, 2)
        )
        debug = observation.get("real_annotation_debug") or {}
        source = (debug.get("part_pose_sources") or {}).get(
            "square_table_top", "unknown"
        )
        color = (
            (80, 220, 80)
            if source in {"detected", "relocalized_detection"}
            else (0, 180, 255)
        )
        cv2.polylines(output, [hull], True, color, 2, cv2.LINE_AA)

        axes_local = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [_TABLETOP_AXIS_LENGTH_M, 0.0, 0.0],
                [0.0, _TABLETOP_AXIS_LENGTH_M, 0.0],
                [0.0, 0.0, _TABLETOP_AXIS_LENGTH_M],
            ],
            dtype=np.float64,
        )
        axes_pixels, axes_visible = _front_projection(
            axes_local, observation, camera_info, output.shape[:2]
        )
        if np.all(axes_visible):
            origin = tuple(np.round(axes_pixels[0]).astype(int))
            for label, endpoint, axis_color in zip(
                ("X", "Y", "Z"),
                axes_pixels[1:],
                ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
            ):
                end = tuple(np.round(endpoint).astype(int))
                cv2.arrowedLine(
                    output,
                    origin,
                    end,
                    axis_color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.15,
                )
                cv2.putText(
                    output,
                    label,
                    end,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    axis_color,
                    1,
                    cv2.LINE_AA,
                )
        text_anchor = tuple(np.min(hull[:, 0, :], axis=0).astype(int))
        text_y = max(16, int(text_anchor[1]) - 7)
        cv2.putText(
            output,
            f"tabletop CAD: {source}",
            (max(2, int(text_anchor[0])), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return output
    return output


def _policy_camera_arrays(
    policy_inputs: Mapping[str, Any], camera: str
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Split one captured post-transform encoder input into RGB and depth."""

    if camera not in policy_inputs:
        raise ValueError(f"missing captured {camera} policy input")
    array = _to_numpy(policy_inputs[camera])
    if array.ndim == 4:
        array = array[-1]
    if array.ndim != 3 or array.shape[0] not in (3, 4):
        raise ValueError(
            f"captured {camera} policy input must be CHW/BCHW RGB(D), got {array.shape}"
        )
    rgb = np.moveaxis(array[:3], 0, -1)
    depth = array[3] if array.shape[0] == 4 else None
    return rgb, depth


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    output = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA,
    )
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    output[y : y + resized_height, x : x + resized_width] = resized
    return output


def _draw_image_panel(
    canvas: np.ndarray,
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    subtitle: str,
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), _PANEL_BACKGROUND, -1)
    cv2.putText(
        canvas,
        title,
        (x + 10, y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        _ACCENT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (x + 10, y + height - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        _MUTED,
        1,
        cv2.LINE_AA,
    )
    image_area = _letterbox(image, width - 20, height - 62)
    canvas[y + 32 : y + height - 30, x + 10 : x + width - 10] = image_area


def render_rgbd_video_frame(policy_inputs: Mapping[str, Any]) -> np.ndarray:
    """Render the exact post-transform front/wrist RGB-D encoder inputs."""

    front_rgb, front_depth = _policy_camera_arrays(policy_inputs, "front")
    wrist_rgb, wrist_depth = _policy_camera_arrays(policy_inputs, "wrist")
    if front_depth is None or wrist_depth is None:
        raise ValueError("RGB-D video requires four-channel policy inputs")
    sources = (
        ("FRONT RGB - FINAL POLICY INPUT", _rgb_to_bgr(front_rgb)),
        ("WRIST RGB - FINAL POLICY INPUT", _rgb_to_bgr(wrist_rgb)),
        ("FRONT DEPTH - FINAL POLICY INPUT", _depth_to_bgr(front_depth)[0]),
        ("WRIST DEPTH - FINAL POLICY INPUT", _depth_to_bgr(wrist_depth)[0]),
    )
    tile_width = VIDEO_FRAME_WIDTH // 2
    tile_height = VIDEO_FRAME_HEIGHT // 2
    tiles = []
    for title, image in sources:
        tile = _letterbox(image, tile_width, tile_height)
        cv2.rectangle(tile, (0, 0), (tile_width, 34), _PANEL_BACKGROUND, -1)
        cv2.putText(
            tile,
            title,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            _ACCENT,
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


class EvalInputVideoRecorder:
    """Asynchronously save every successful query's RGB-D inputs."""

    _STOP = object()

    def __init__(self, *, enabled: bool, fps: float):
        self.enabled = bool(enabled)
        self.fps = float(fps)
        if self.enabled and self.fps <= 0:
            raise ValueError("input video fps must be positive")
        self.path: Optional[Path] = None
        self.frame_count = 0
        self.error: Optional[str] = None
        self._queue = None
        self._thread = None
        self._writer = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, path: Path) -> Optional[Path]:
        if not self.enabled:
            return None
        if self.active:
            raise RuntimeError("input video recorder is already active")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.error = None
        self._queue = queue.SimpleQueue()
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT),
        )
        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise RuntimeError(f"failed to open input video writer: {self.path}")
        self._thread = threading.Thread(
            target=self._run,
            name="rr-input-video",
            daemon=True,
        )
        self._thread.start()
        return self.path

    def submit(self, policy_inputs: Mapping[str, Any]) -> bool:
        if not self.active or self.error is not None:
            return False
        frame_inputs = {}
        for camera in ("front", "wrist"):
            if camera not in policy_inputs:
                self.error = f"missing captured {camera} policy input"
                return False
            frame_inputs[camera] = _to_numpy(policy_inputs[camera]).copy()
        self._queue.put(frame_inputs)
        return True

    def close(self) -> Mapping[str, Any]:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(self._STOP)
            self._thread.join(timeout=30.0)
            if self._thread.is_alive() and self.error is None:
                self.error = "video writer did not stop within 30 seconds"
        if self._writer is not None:
            self._writer.release()
        result = {
            "path": None if self.path is None else str(self.path),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "error": self.error,
        }
        self._thread = None
        self._queue = None
        self._writer = None
        self.path = None
        return result

    def _run(self) -> None:
        try:
            while True:
                observation = self._queue.get()
                if observation is self._STOP:
                    return
                frame = render_rgbd_video_frame(observation)
                self._writer.write(frame)
                self.frame_count += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"


def _dashboard_lines(
    observation: Mapping[str, Any],
    policy_inputs: Mapping[str, Any],
    *,
    query_id: int,
    annotation_mode: str,
    timing: Mapping[str, Any],
    inference_latency_ms: float,
    action_chunk: np.ndarray,
    actions_accepted: int,
    common_stale_prefix: int,
    query_interval_steps: int,
    scheduled: bool,
    checkpoint: Optional[Path],
    arm_queue_pending: int = 0,
    gripper_queue_pending: int = 0,
    immutable_coverage_ms: float = 0.0,
    occupied_preserved: int = 0,
    warmstart_mapped_count: int = 0,
) -> list[str]:
    state = observation.get("robot_state") or {}
    projections = observation.get("guidance_point_2d") or {}
    debug = observation.get("real_annotation_debug") or {}
    actions = np.asarray(action_chunk)
    gripper = actions[:, -1] if actions.ndim == 2 and actions.shape[1] else np.asarray([])
    live_start = min(int(common_stale_prefix), gripper.size)
    live_end = min(live_start + int(query_interval_steps), gripper.size)
    near_gripper = gripper[live_start:live_end]
    _, front_depth = _policy_camera_arrays(policy_inputs, "front")
    _, wrist_depth = _policy_camera_arrays(policy_inputs, "wrist")
    front_depth_stats, _ = _depth_stats(front_depth)
    wrist_depth_stats, _ = _depth_stats(wrist_depth)
    lines = [
        f"query: {query_id}    scheduled: {scheduled}",
        f"checkpoint: {Path(checkpoint).name if checkpoint else 'unknown'}",
        f"annotation mode: {annotation_mode}",
        "panels show exact post-transform tensors sent to the encoders",
        "",
        "ONLINE ANNOTATION RESULT",
        f"skill: {observation.get('skill')}",
        f"skill_state: {observation.get('skill_state')}",
        f"assembly_step: {observation.get('assembly_step')}",
        f"guidance xyz(base): {_format_vector(observation.get('guidance_point'))}",
        f"guidance uv front: {_format_uv(projections.get('color_image2'))}",
        f"guidance uv wrist: {_format_uv(projections.get('color_image1'))}",
        f"parts found: {_format_vector(observation.get('parts_founds'), 0)}",
        f"parts valid: {_format_vector(observation.get('parts_pose_valid'), 0)}",
        f"active/attached: {debug.get('active_part')} / {debug.get('attached_part')}",
        f"gripper event: {debug.get('gripper_event')}",
        f"pose sources: {debug.get('part_pose_sources')}",
        "",
        "POLICY PROPRIOCEPTION (raw 14D)",
        f"ee pos: {_format_vector(state.get('ee_pos'))}",
        f"ee quat xyzw: {_format_vector(state.get('ee_quat'))}",
        f"ee linear vel: {_format_vector(state.get('ee_pos_vel'))}",
        f"ee angular vel: {_format_vector(state.get('ee_ori_vel'))}",
        f"gripper width: {state.get('gripper_width')}",
        "ROBOT DIAGNOSTICS (not policy tensor)",
        f"joint q: {_format_vector(state.get('joint_positions'))}",
        f"joint dq: {_format_vector(state.get('joint_velocities'))}",
        f"joint tau: {_format_vector(state.get('joint_torques'))}",
        "",
        "FINAL POLICY IMAGE INPUTS (post actor transform)",
        f"front RGB-D: {_shape_and_dtype(policy_inputs.get('front'))}",
        f"  {front_depth_stats}",
        f"wrist RGB-D: {_shape_and_dtype(policy_inputs.get('wrist'))}",
        f"  {wrist_depth_stats}",
        "",
        "TIMING / OUTPUT",
        "frames front/wrist: "
        f"{timing.get('front_frame_number')} / {timing.get('wrist_frame_number')}",
        f"front age: {float(timing.get('front_age_ms_at_build', 0)):.1f} ms",
        f"wrist residual: {float(timing.get('wrist_residual_ms', 0)):.1f} ms",
        f"PromptDA latency: {float(timing.get('prompt_depth_latency_ms', 0)):.1f} ms",
        f"policy inference: {float(inference_latency_ms):.1f} ms",
        f"action chunk: {tuple(actions.shape)} "
        f"accepted/stale={actions_accepted}/{common_stale_prefix}",
        "queues arm/gripper: "
        f"{arm_queue_pending}/{gripper_queue_pending}",
        f"immutable coverage ahead: {float(immutable_coverage_ms):.1f} ms",
        f"occupied preserved: {occupied_preserved}",
        f"warm-start queue mappings: {warmstart_mapped_count}",
    ]
    if gripper.size:
        lines.extend(
            [
                "gripper min/max: "
                f"{float(gripper.min()):.3f}/{float(gripper.max()):.3f}",
                "gripper open(<0)/close(>=0): "
                f"{int(np.sum(gripper < 0))}/{int(np.sum(gripper >= 0))}",
                f"next-{query_interval_steps} close predictions: "
                f"{int(np.sum(near_gripper >= 0))}",
                f"gripper sequence: {_format_vector(gripper, 2)}",
            ]
        )
    return lines


def render_input_dashboard(
    observation: Mapping[str, Any],
    *,
    policy_inputs: Mapping[str, Any],
    query_id: int,
    annotation_mode: str,
    timing: Mapping[str, Any],
    inference_latency_ms: float,
    action_chunk: np.ndarray,
    actions_accepted: int,
    common_stale_prefix: int,
    query_interval_steps: int,
    scheduled: bool,
    checkpoint: Optional[Path] = None,
    arm_queue_pending: int = 0,
    gripper_queue_pending: int = 0,
    immutable_coverage_ms: float = 0.0,
    occupied_preserved: int = 0,
    warmstart_mapped_count: int = 0,
    camera_info: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """Render one query using the exact post-transform encoder inputs."""

    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), _BACKGROUND, dtype=np.uint8)
    cv2.putText(
        canvas,
        "RR REAL EVAL - EXACT QUERY INPUT INSPECTION",
        (12, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        _TEXT,
        2,
        cv2.LINE_AA,
    )
    skill = str(observation.get("skill") or "unknown")
    skill_state = str(observation.get("skill_state") or "unknown")
    assembly_step = str(observation.get("assembly_step") or "unknown")
    skill_banner = f"SKILL: {skill.upper()}   STATE: {skill_state.upper()}"
    banner_color = {
        "pick": (80, 210, 255),
        "place": (120, 230, 120),
        "insert": (255, 190, 80),
        "screw": (220, 140, 255),
    }.get(skill.lower(), _ACCENT)
    cv2.rectangle(canvas, (1030, 6), (1590, 41), _PANEL_BACKGROUND, -1)
    cv2.rectangle(canvas, (1030, 6), (1040, 41), banner_color, -1)
    cv2.putText(
        canvas,
        skill_banner[:55],
        (1050, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        banner_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"ASSEMBLY: {assembly_step}"[:28],
        (1335, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        _TEXT,
        1,
        cv2.LINE_AA,
    )
    front_rgb_array, front_depth_array = _policy_camera_arrays(
        policy_inputs, "front"
    )
    front_size = f"{front_rgb_array.shape[1]}x{front_rgb_array.shape[0]}"
    wrist_rgb_array, wrist_depth_array = _policy_camera_arrays(
        policy_inputs, "wrist"
    )
    wrist_size = f"{wrist_rgb_array.shape[1]}x{wrist_rgb_array.shape[0]}"
    front_rgb = _draw_tabletop_cad_overlay(
        _rgb_to_bgr(front_rgb_array), observation, camera_info
    )
    wrist_rgb = _rgb_to_bgr(wrist_rgb_array)
    if front_depth_array is None or wrist_depth_array is None:
        raise ValueError("RGB-D dashboard requires four-channel policy inputs")
    front_depth, front_stats = _depth_to_bgr(front_depth_array)
    wrist_depth, wrist_stats = _depth_to_bgr(wrist_depth_array)
    panel_width = 500
    panel_height = 405
    _draw_image_panel(
        canvas,
        front_rgb,
        x=10,
        y=48,
        width=panel_width,
        height=panel_height,
        title=f"FRONT RGB - FINAL {front_size} ({annotation_mode})",
        subtitle="exact post-transform RGB sent to front encoder",
    )
    _draw_image_panel(
        canvas,
        wrist_rgb,
        x=520,
        y=48,
        width=panel_width,
        height=panel_height,
        title=f"WRIST RGB - FINAL {wrist_size}",
        subtitle="exact post-transform RGB sent to wrist encoder",
    )
    _draw_image_panel(
        canvas,
        front_depth,
        x=10,
        y=465,
        width=panel_width,
        height=panel_height,
        title=f"FRONT DEPTH - FINAL {front_size} (meters)",
        subtitle=f"exact post-transform depth; TURBO p02-p98; {front_stats}",
    )
    _draw_image_panel(
        canvas,
        wrist_depth,
        x=520,
        y=465,
        width=panel_width,
        height=panel_height,
        title=f"WRIST DEPTH - FINAL {wrist_size} (meters)",
        subtitle=f"exact post-transform depth; TURBO p02-p98; {wrist_stats}",
    )

    cv2.rectangle(canvas, (1030, 48), (1590, 870), _PANEL_BACKGROUND, -1)
    lines = _dashboard_lines(
        observation,
        policy_inputs,
        query_id=query_id,
        annotation_mode=annotation_mode,
        timing=timing,
        inference_latency_ms=inference_latency_ms,
        action_chunk=action_chunk,
        actions_accepted=actions_accepted,
        common_stale_prefix=common_stale_prefix,
        query_interval_steps=query_interval_steps,
        scheduled=scheduled,
        checkpoint=checkpoint,
        arm_queue_pending=arm_queue_pending,
        gripper_queue_pending=gripper_queue_pending,
        immutable_coverage_ms=immutable_coverage_ms,
        occupied_preserved=occupied_preserved,
        warmstart_mapped_count=warmstart_mapped_count,
    )
    text_y = 69
    for line in lines:
        color = _ACCENT if line in {
            "ONLINE ANNOTATION RESULT",
            "POLICY PROPRIOCEPTION (raw 14D)",
            "ROBOT DIAGNOSTICS (not policy tensor)",
            "FINAL POLICY IMAGE INPUTS (post actor transform)",
            "TIMING / OUTPUT",
        } else _TEXT
        cv2.putText(
            canvas,
            str(line)[:82],
            (1042, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
        text_y += 18
        if text_y > 858:
            break
    return canvas


class EvalInputDashboard:
    """Display the latest query while keeping GUI failures out of control flow."""

    def __init__(
        self,
        *,
        enabled: bool,
        checkpoint: Optional[Path] = None,
        actor: Any = None,
        capture_policy_inputs: bool = False,
    ):
        self.enabled = bool(enabled)
        self.capture_policy_inputs = bool(enabled or capture_policy_inputs)
        self.checkpoint = checkpoint
        self.window_name = "RR real eval query inputs"
        self.camera_info: Optional[Mapping[str, Any]] = None
        self._policy_inputs: dict[str, Any] = {}
        self._hooks = []
        if self.capture_policy_inputs:
            if actor is None:
                raise ValueError("policy input capture requires actor")
            self._hooks = [
                actor.camera1_transform.register_forward_hook(
                    self._capture_policy_input("wrist")
                ),
                actor.camera2_transform.register_forward_hook(
                    self._capture_policy_input("front")
                ),
            ]

    def policy_inputs_snapshot(self) -> dict[str, Any]:
        """Copy the most recent exact transform outputs after inference."""

        return {
            camera: value.detach().cpu().clone()
            for camera, value in self._policy_inputs.items()
        }

    def _capture_policy_input(self, camera: str):
        def capture(_module, _inputs, output):
            # Keep the exact transform output.  CPU transfer happens after
            # inference, when the dashboard is rendered, so the hook does not
            # stall the encoder's CUDA stream before its forward pass.
            self._policy_inputs[camera] = output.detach()

        return capture

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def set_camera_info(self, camera_info: Mapping[str, Any]) -> None:
        self.camera_info = camera_info

    def show(self, observation: Mapping[str, Any], **query: Any) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            frame = render_input_dashboard(
                observation,
                policy_inputs=self._policy_inputs,
                checkpoint=self.checkpoint,
                camera_info=self.camera_info,
                **query,
            )
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, CANVAS_WIDTH, CANVAS_HEIGHT)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
        except Exception as exc:  # A preview must never reject a robot action.
            self.enabled = False
            self._remove_hooks()
            return f"{type(exc).__name__}: {exc}"
        return None

    def close(self) -> None:
        self._remove_hooks()
        if not self.enabled:
            return
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
