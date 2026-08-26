from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch


GUIDANCE_POINT_COLOR_MAP = {
    "pick": (0, 255, 255),
    "screw": (0, 255, 255),
    "place": (255, 0, 0),
    "push": (255, 0, 0),
    "insert": (255, 0, 0),
}
DEFAULT_GUIDANCE_POINT_COLOR_RGB = (255, 0, 0)
GUIDANCE_POINT_ALPHA = 0.5
GUIDANCE_POINT_RADIUS_PX = 2
GRASP_COLOR_GROUP_A = {"pick", "screw"}
GRASP_COLOR_GROUP_B = {"place", "push", "insert"}

# Colors are written in the image channel order used by the existing OpenCV
# drawing path. Each rectangle edge gets a distinct color, and the two skill
# groups use separate high-contrast palettes.
GRASP_COLORED_PALETTE_A = {
    "main_a": (255, 0, 255),
    "main_b": (0, 255, 255),
    "side_a": (255, 255, 0),
    "side_b": (0, 255, 0),
}
GRASP_COLORED_PALETTE_B = {
    "main_a": (0, 165, 255),
    "main_b": (0, 0, 255),
    "side_a": (255, 0, 0),
    "side_b": (255, 0, 128),
}


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def resize_guidance_point_for_image(
    guidance_point_2d,
    *,
    image_key: str,
    source_image_size,
    image_shape,
):
    """Map a projected guidance point to the rendered image resolution.

    This is the shared transform used by simulator rollouts and offline
    real-robot videos. Wrist images are resized directly. Front images follow
    the simulator's resize-to-height then horizontal-center-crop path.
    ``source_image_size`` is ``(width, height)``.
    """
    if guidance_point_2d is None or source_image_size is None:
        return guidance_point_2d

    source_width, source_height = [int(v) for v in source_image_size]
    target_height, target_width = [int(v) for v in image_shape[:2]]
    if source_width == target_width and source_height == target_height:
        return guidance_point_2d

    uv = np.asarray(guidance_point_2d, dtype=np.float32)
    if image_key == "color_image1":
        sx = target_width / max(source_width, 1)
        sy = target_height / max(source_height, 1)
        uv = np.array([uv[0] * sx, uv[1] * sy], dtype=np.float32)
    elif image_key == "color_image2":
        aspect_ratio = source_width / max(source_height, 1)
        resized_width = int(target_height * aspect_ratio)
        crop_size = max(0, (resized_width - target_width) // 2)
        sx = resized_width / max(source_width, 1)
        sy = target_height / max(source_height, 1)
        uv = np.array(
            [uv[0] * sx - crop_size, uv[1] * sy], dtype=np.float32
        )
    else:
        return guidance_point_2d

    if (
        uv[0] < 0
        or uv[0] >= target_width
        or uv[1] < 0
        or uv[1] >= target_height
    ):
        return None
    return uv.astype(np.float32)


def draw_guidance_point_on_image(
    image: np.ndarray,
    guidance_point_2d,
    skill: Optional[str] = None,
    use_skill_color: bool = False,
) -> np.ndarray:
    if guidance_point_2d is None:
        return image

    uv = _to_numpy(guidance_point_2d).astype(np.int32)
    annotated = image.copy()
    if use_skill_color and skill and skill in GUIDANCE_POINT_COLOR_MAP:
        point_color = GUIDANCE_POINT_COLOR_MAP[skill]
    else:
        point_color = DEFAULT_GUIDANCE_POINT_COLOR_RGB

    def _draw_point(frame: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        overlay = frame.copy()
        cv2.circle(
            overlay,
            center,
            GUIDANCE_POINT_RADIUS_PX,
            point_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        return cv2.addWeighted(
            overlay,
            GUIDANCE_POINT_ALPHA,
            frame,
            1.0 - GUIDANCE_POINT_ALPHA,
            0.0,
        )

    if annotated.ndim == 4:
        if annotated.shape[0] != 1:
            return annotated
        frame = annotated[0]
        height, width = frame.shape[:2]
        center = (int(uv[0]), int(uv[1]))
        if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
            return annotated
        annotated[0] = _draw_point(frame, center)
        return annotated

    height, width = annotated.shape[:2]
    center = (int(uv[0]), int(uv[1]))
    if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
        return annotated
    return _draw_point(annotated, center)


def draw_grasp_annotation_on_image(
    image: np.ndarray,
    grasp_annotation_2d,
    skill: Optional[str] = None,
    use_skill_color: bool = False,
) -> np.ndarray:
    if not grasp_annotation_2d or grasp_annotation_2d.get("style") != "grasp_rect":
        return image

    corners = _to_numpy(grasp_annotation_2d.get("corners"))
    center = _to_numpy(grasp_annotation_2d.get("center"))
    if corners is None or center is None or corners.shape != (4, 2):
        return image

    main_a_color = (255, 0, 0)
    main_b_color = (0, 0, 255)
    side_a_color = (0, 255, 255)
    side_b_color = (0, 255, 0)
    if use_skill_color and skill in GRASP_COLOR_GROUP_A:
        main_a_color = GRASP_COLORED_PALETTE_A["main_a"]
        main_b_color = GRASP_COLORED_PALETTE_A["main_b"]
        side_a_color = GRASP_COLORED_PALETTE_A["side_a"]
        side_b_color = GRASP_COLORED_PALETTE_A["side_b"]
    elif use_skill_color and skill in GRASP_COLOR_GROUP_B:
        main_a_color = GRASP_COLORED_PALETTE_B["main_a"]
        main_b_color = GRASP_COLORED_PALETTE_B["main_b"]
        side_a_color = GRASP_COLORED_PALETTE_B["side_a"]
        side_b_color = GRASP_COLORED_PALETTE_B["side_b"]
    points = np.round(corners).astype(np.int32)
    center = np.round(center).astype(np.int32)

    def _draw_rect(frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.line(overlay, tuple(points[0]), tuple(points[1]), main_a_color, 2, cv2.LINE_AA)
        cv2.line(overlay, tuple(points[2]), tuple(points[3]), main_b_color, 2, cv2.LINE_AA)
        cv2.line(overlay, tuple(points[1]), tuple(points[2]), side_b_color, 1, cv2.LINE_AA)
        cv2.line(overlay, tuple(points[3]), tuple(points[0]), side_a_color, 1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(center), 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        return overlay

    if image.ndim == 4:
        if image.shape[0] != 1:
            return image
        annotated = image.copy()
        annotated[0] = _draw_rect(annotated[0])
        return annotated

    return _draw_rect(image.copy())
