"""Single-window inspection dashboard for real-world policy inputs."""

from __future__ import annotations

import queue
import threading
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


def render_rgbd_video_frame(observation: Mapping[str, Any]) -> np.ndarray:
    """Render front/wrist RGB-D query inputs as one 2x2 BGR frame."""

    sources = (
        ("FRONT RGB", _rgb_to_bgr(observation["color_image2"])),
        ("WRIST RGB", _rgb_to_bgr(observation["color_image1"])),
        ("FRONT DEPTH", _depth_to_bgr(observation["depth_image2"])[0]),
        ("WRIST DEPTH", _depth_to_bgr(observation["depth_image1"])[0]),
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

    def submit(self, observation: Mapping[str, Any]) -> bool:
        if not self.active or self.error is not None:
            return False
        frame_inputs = {
            key: np.asarray(observation[key]).copy()
            for key in (
                "color_image1",
                "color_image2",
                "depth_image1",
                "depth_image2",
            )
        }
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
    front_rgb = _rgb_to_bgr(front_rgb_array)
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
    ):
        self.enabled = bool(enabled)
        self.checkpoint = checkpoint
        self.window_name = "RR real eval query inputs"
        self._policy_inputs: dict[str, Any] = {}
        self._hooks = []
        if self.enabled:
            if actor is None:
                raise ValueError("enabled policy dashboard requires actor")
            self._hooks = [
                actor.camera1_transform.register_forward_hook(
                    self._capture_policy_input("wrist")
                ),
                actor.camera2_transform.register_forward_hook(
                    self._capture_policy_input("front")
                ),
            ]

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

    def show(self, observation: Mapping[str, Any], **query: Any) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            frame = render_input_dashboard(
                observation,
                policy_inputs=self._policy_inputs,
                checkpoint=self.checkpoint,
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
