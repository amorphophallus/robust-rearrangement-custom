"""Render an annotated trajectory with the standard simulator visuals.

This renderer intentionally reuses the same drawing primitives and layout as
the simulator rollout path: wrist RGB on the left, front RGB on the right,
the default guidance point on the front image, and the skill label on the
front image.  It contains no real-robot-specific diagnostic overlays.
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

from src.common.image_annotations import (
    draw_guidance_point_on_image,
    resize_guidance_point_for_image,
)
from src.common.guidance import camera_info_for_image
from src.eval.skill_annotation_util import draw_skill_on_image, project_3d_to_2d


DEFAULT_FPS = 20
WRIST_IMAGE_KEY = "color_image1"
FRONT_IMAGE_KEY = "color_image2"


def _load_annotated_pickle(path: Path) -> Dict[str, Any]:
    if path.suffix == ".gz":
        opener = gzip.open
    elif path.suffix == ".xz":
        opener = lzma.open
    else:
        opener = open
    with opener(path, "rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise ValueError("Annotated pickle root must be a mapping")
    return data


def _camera_annotation(
    observation: Mapping[str, Any], key: str, camera: str
):
    value = observation.get(key)
    return value.get(camera) if isinstance(value, Mapping) else None


def _recorded_image_size(
    camera_info: Optional[Mapping[str, Any]], image_key: str
):
    """Return the annotation projection size as ``(width, height)``."""
    if not isinstance(camera_info, Mapping):
        return None

    saved_projection = camera_info_for_image(camera_info, image_key)
    if isinstance(saved_projection, Mapping) and saved_projection.get("image_size") is not None:
        size = np.asarray(saved_projection["image_size"]).reshape(-1)
        if size.shape == (2,):
            return int(size[0]), int(size[1])

    mapping = camera_info.get("camera_key_mapping", {})
    default_camera = "wrist" if image_key == WRIST_IMAGE_KEY else "front"
    camera_name = (
        mapping.get(image_key, default_camera)
        if isinstance(mapping, Mapping)
        else default_camera
    )
    details = camera_info.get(camera_name)
    if isinstance(details, Mapping):
        intrinsics = details.get("record_intrinsics")
        if isinstance(intrinsics, Mapping):
            width = intrinsics.get("width")
            height = intrinsics.get("height")
            if width is not None and height is not None:
                return int(width), int(height)
        transform = details.get("record_transform")
        if isinstance(transform, Mapping):
            width = transform.get("output_width")
            height = transform.get("output_height")
            if width is not None and height is not None:
                return int(width), int(height)

    width = camera_info.get("record_width")
    height = camera_info.get("record_height")
    if width is not None and height is not None:
        return int(width), int(height)
    return None


def _front_guidance_point(
    observation: Mapping[str, Any],
    camera_info: Optional[Mapping[str, Any]],
):
    front_camera = camera_info_for_image(camera_info, FRONT_IMAGE_KEY)
    point = observation.get("guidance_point")
    if (
        point is not None
        and isinstance(front_camera, Mapping)
        and "robot_base_to_camera" in front_camera
    ):
        return project_3d_to_2d(point, front_camera)
    return _camera_annotation(observation, "guidance_point_2d", FRONT_IMAGE_KEY)


def _draw_robot_base_axes(
    image: np.ndarray,
    camera_info: Optional[Mapping[str, Any]],
    *,
    axis_length_m: float = 0.15,
) -> np.ndarray:
    """Draw the projected robot-base origin and +X/+Y/+Z axes when available."""

    camera = camera_info_for_image(camera_info, FRONT_IMAGE_KEY)
    if not isinstance(camera, Mapping) or "robot_base_to_camera" not in camera:
        return image
    points = {
        "O": np.zeros(3, dtype=np.float32),
        "X": np.array([axis_length_m, 0.0, 0.0], dtype=np.float32),
        "Y": np.array([0.0, axis_length_m, 0.0], dtype=np.float32),
        "Z": np.array([0.0, 0.0, axis_length_m], dtype=np.float32),
    }
    pixels = {label: project_3d_to_2d(point, camera) for label, point in points.items()}
    if pixels["O"] is None:
        return image
    output = image.copy()
    origin = tuple(np.asarray(pixels["O"], dtype=int))
    colors = {"X": (255, 64, 64), "Y": (64, 255, 64), "Z": (64, 128, 255)}
    for label in ("X", "Y", "Z"):
        if pixels[label] is None:
            continue
        endpoint = tuple(np.asarray(pixels[label], dtype=int))
        cv2.arrowedLine(output, origin, endpoint, colors[label], 2, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(
            output,
            label,
            (endpoint[0] + 3, endpoint[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colors[label],
            1,
            cv2.LINE_AA,
        )
    cv2.circle(output, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        output,
        "robot base",
        (origin[0] + 5, origin[1] + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def standard_annotation_frame(
    observation: Mapping[str, Any],
    camera_info: Optional[Mapping[str, Any]] = None,
    *,
    show_robot_base_axes: bool = False,
) -> np.ndarray:
    """Return one frame using the simulator rollout annotation settings."""

    wrist = np.asarray(observation[WRIST_IMAGE_KEY], dtype=np.uint8).copy()
    front = np.asarray(observation[FRONT_IMAGE_KEY], dtype=np.uint8).copy()
    if show_robot_base_axes:
        front = _draw_robot_base_axes(front, camera_info)
    guidance = resize_guidance_point_for_image(
        _front_guidance_point(observation, camera_info),
        image_key=FRONT_IMAGE_KEY,
        source_image_size=_recorded_image_size(camera_info, FRONT_IMAGE_KEY),
        image_shape=front.shape,
    )
    front = draw_guidance_point_on_image(
        front,
        guidance,
        skill=observation.get("skill"),
        use_skill_color=False,
    )
    skill = observation.get("skill")
    if isinstance(skill, bytes):
        skill = skill.decode("utf-8")
    if skill is not None:
        front = draw_skill_on_image(front, str(skill))
    return np.concatenate([wrist, front], axis=1)


def _open_encoder(
    output_path: Path, width: int, height: int, fps: int
) -> subprocess.Popen:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )


def render_skill_annotation_video(
    annotated_pickle: Path,
    output_path: Path,
    *,
    fps: int = DEFAULT_FPS,
    show_robot_base_axes: bool = False,
) -> Dict[str, Any]:
    data = _load_annotated_pickle(annotated_pickle)
    observations = data.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Annotated pickle must contain non-empty observations")

    camera_info = data.get("camera_info")
    sample = standard_annotation_frame(
        observations[0], camera_info, show_robot_base_axes=show_robot_base_axes
    )
    height, width = sample.shape[:2]
    encoder = _open_encoder(output_path, width, height, fps)
    preview_idx = next(
        (
            idx
            for idx, observation in enumerate(observations)
            if observation.get("skill") == "push"
        ),
        len(observations) // 2,
    )
    preview_path = output_path.with_suffix(".preview.jpg")
    skill_counts: Dict[str, int] = {}
    visible_guidance_frames = 0
    try:
        for frame_idx, observation in enumerate(observations):
            frame = standard_annotation_frame(
                observation,
                camera_info,
                show_robot_base_axes=show_robot_base_axes,
            )
            skill = str(observation.get("skill", "none"))
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
            if (
                _front_guidance_point(observation, camera_info) is not None
            ):
                visible_guidance_frames += 1
            if frame_idx == preview_idx:
                preview_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(preview_path), preview_bgr):
                    raise OSError(f"Failed to write preview {preview_path}")
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

    report = {
        "input": str(Path(annotated_pickle).resolve()),
        "video": str(Path(output_path).resolve()),
        "preview": str(preview_path.resolve()),
        "frame_count": len(observations),
        "fps": fps,
        "layout": [WRIST_IMAGE_KEY, FRONT_IMAGE_KEY],
        "front_annotations": ["guidance_point", "skill"],
        "guidance_point_colored": False,
        "guidance_frame": data.get("guidance_frame"),
        "robot_base_axes": show_robot_base_axes,
        "image_channel_order": "RGB",
        "front_projection_size": _recorded_image_size(
            camera_info, FRONT_IMAGE_KEY
        ),
        "front_render_size": [
            int(observations[0][FRONT_IMAGE_KEY].shape[1]),
            int(observations[0][FRONT_IMAGE_KEY].shape[0]),
        ],
        "skill_counts": skill_counts,
        "visible_front_guidance_frames": visible_guidance_frames,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotated_pickle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--show-robot-base-axes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = render_skill_annotation_video(
        args.annotated_pickle.resolve(),
        args.output.resolve(),
        fps=args.fps,
        show_robot_base_axes=args.show_robot_base_axes,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
