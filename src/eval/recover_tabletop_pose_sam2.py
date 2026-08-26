"""Recover the final real-robot tabletop pose with SAM 2 and aligned RGB-D.

This script intentionally runs outside the legacy ``rr`` environment.  It only
depends on the trajectory pickle schema and the optional Meta SAM 2 package;
the resulting JSON manifest is consumed by ``real_skill_annotation_util``.

The recovery is split into two observable stages:

1. Project the known tabletop mesh at a reliable AprilTag frame and use the
   resulting image box to prompt SAM 2 video tracking.
2. Recover a planar tabletop pose by robustly registering the complete
   SAM-mask RGB-D surface to the CAD top footprint.

All diagnostic masks and overlays are written below the requested output
directory.  The input pickle is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


TABLETOP_NAME = "square_table_top"
TABLETOP_PART_INDEX = 0
TABLETOP_HOLE_OFFSET_M = 0.05625
APRIL_TO_ROBOT = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.3015],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class RecoveryResult:
    trajectory: str
    part_name: str
    prompt_frame: int
    push_end_frame: int
    place_start_frame: int
    keyframe: Optional[int]
    start_frame: Optional[int]
    pose_april: Optional[list]
    confidence: float
    source: str
    diagnostics_dir: str
    fit_details: Optional[dict]


def _load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def _quat_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-9:
        raise ValueError("Quaternion has zero norm")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_vector_to_matrix(pose: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_xyzw_to_matrix(pose[3:])
    matrix[:3, 3] = pose[:3]
    return matrix


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized quaternion in xyzw order."""

    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def _intrinsics(camera_info: Mapping[str, Any]) -> np.ndarray:
    front = camera_info["front"]
    values = front.get("record_intrinsics", front.get("intrinsics"))
    return np.asarray(
        [
            [values["fx"], 0.0, values.get("ppx", values.get("cx"))],
            [0.0, values["fy"], values.get("ppy", values.get("cy"))],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _project_mesh_box(
    vertices: np.ndarray,
    pose_april: np.ndarray,
    camera_to_april: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    margin_px: int = 4,
) -> np.ndarray:
    vertex_h = np.concatenate(
        [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1
    )
    points_camera = (
        np.linalg.inv(camera_to_april) @ pose_april @ vertex_h.T
    ).T[:, :3]
    points_camera = points_camera[points_camera[:, 2] > 1e-4]
    if not len(points_camera):
        raise ValueError("Projected tabletop mesh is behind the front camera")
    pixels_h = (intrinsics @ points_camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    height, width = image_shape
    low = np.floor(np.nanmin(pixels, axis=0) - margin_px)
    high = np.ceil(np.nanmax(pixels, axis=0) + margin_px)
    low = np.maximum(low, [0, 0])
    high = np.minimum(high, [width - 1, height - 1])
    box = np.concatenate([low, high]).astype(np.float32)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        raise ValueError(f"Degenerate projected tabletop box: {box.tolist()}")
    return box


def _phase_boundaries(observations: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
    skills = [observation.get("skill") for observation in observations]
    push_indices = [idx for idx, skill in enumerate(skills) if skill == "push"]
    place_indices = [idx for idx, skill in enumerate(skills) if skill == "place"]
    if not push_indices or not place_indices:
        raise ValueError("Input must already contain push/place skill annotations")
    return push_indices[-1] + 1, place_indices[0]


def _flat_tag_frames(
    observations: Sequence[Mapping[str, Any]],
    april_to_robot: np.ndarray,
    push_end: int,
) -> Tuple[int, int]:
    """Return clean-prompt and latest-flat AprilTag frames before push end."""

    candidates = []
    for idx, observation in enumerate(observations[:push_end]):
        founds = np.asarray(observation.get("parts_founds", []), dtype=bool)
        if founds.size <= TABLETOP_PART_INDEX or not founds[TABLETOP_PART_INDEX]:
            continue
        pose = np.asarray(observation["parts_poses"][:7], dtype=np.float64)
        pose_robot = april_to_robot @ _pose_vector_to_matrix(pose)
        # The tabletop mesh lies in local XZ; local Y is its surface normal.
        flatness = abs(float(pose_robot[2, 1]))
        if flatness >= 0.90:
            candidates.append(idx)
    if not candidates:
        raise ValueError("No reliable flat tabletop AprilTag pose before push end")
    # The opening frame is unobstructed in the teleoperation trajectories.  A
    # clean prompt is more useful to SAM than the last tag frame, where the arm
    # is already covering most of the tabletop.
    return candidates[0], candidates[-1]


def _write_video_frames(
    observations: Sequence[Mapping[str, Any]], frame_dir: Path, end_frame: int
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(end_frame + 1):
        output = frame_dir / f"{idx:05d}.jpg"
        if output.exists():
            continue
        image = np.asarray(observations[idx]["color_image2"], dtype=np.uint8)
        if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Failed to write {output}")


def _draw_prompt(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    output = image.copy()
    x0, y0, x1, y1 = np.round(box).astype(int)
    cv2.rectangle(output, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(
        output,
        "SAM2 tabletop prompt",
        (max(0, x0), max(18, y0 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _run_sam_video(
    *,
    frame_dir: Path,
    prompt_frame: int,
    prompt_box: np.ndarray,
    checkpoint: Path,
    model_config: str,
    device: str,
) -> Dict[int, np.ndarray]:
    from sam2.build_sam import build_sam2_video_predictor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM2 CUDA inference requested but CUDA is unavailable")
    predictor = build_sam2_video_predictor(
        model_config, str(checkpoint), device=device
    )
    state = predictor.init_state(
        str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=False
    )
    predictor.add_new_points_or_box(
        state,
        frame_idx=prompt_frame,
        obj_id=1,
        box=prompt_box,
    )
    masks: Dict[int, np.ndarray] = {}
    for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
        state,
        start_frame_idx=prompt_frame,
        max_frame_num_to_track=len(list(frame_dir.glob("*.jpg"))),
    ):
        object_index = list(object_ids).index(1)
        masks[int(frame_idx)] = (
            mask_logits[object_index, 0].detach().cpu().numpy() > 0.0
        )
    return masks


def _load_saved_masks(path: Path) -> Dict[int, np.ndarray]:
    saved = np.load(path)
    return {
        int(frame_idx): np.asarray(mask, dtype=bool)
        for frame_idx, mask in zip(saved["frame_indices"], saved["masks"])
    }


def _depth_points_robot(
    observation: Mapping[str, Any],
    mask: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(observation["depth_image2"], dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0.10) & (depth < 1.50)
    rows, columns = np.nonzero(valid)
    values = depth[rows, columns]
    if values.size < 100:
        raise ValueError("Too few valid front-depth pixels inside the SAM mask")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    points_camera = np.stack(
        [
            (columns - cx) * values / fx,
            (rows - cy) * values / fy,
            values,
            np.ones_like(values),
        ],
        axis=1,
    )
    camera_to_robot = APRIL_TO_ROBOT @ np.asarray(
        observation["camera_to_april"], dtype=np.float64
    )
    return (camera_to_robot @ points_camera.T).T[:, :3]


def _dominant_surface_height(
    points_robot: np.ndarray,
    preferred_height: Optional[float] = None,
) -> float:
    heights = np.asarray(points_robot, dtype=np.float64)[:, 2]
    counts, edges = np.histogram(heights, bins=240, range=(-0.08, 0.10))
    smoothed = np.convolve(
        counts.astype(np.float64),
        np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]),
        mode="same",
    )
    candidates = np.flatnonzero(
        (smoothed >= np.roll(smoothed, 1))
        & (smoothed >= np.roll(smoothed, -1))
        & (smoothed >= float(np.max(smoothed)) * 0.10)
    )
    candidates = candidates[(candidates > 0) & (candidates < len(counts) - 1)]
    if preferred_height is None or not len(candidates):
        index = int(np.argmax(smoothed))
    else:
        centers = (edges[candidates] + edges[candidates + 1]) * 0.5
        index = int(candidates[np.argmin(np.abs(centers - preferred_height))])
    coarse = float((edges[index] + edges[index + 1]) * 0.5)
    near = heights[np.abs(heights - coarse) <= 0.004]
    if near.size < 100:
        raise ValueError("Could not identify tabletop surface in masked depth")
    return float(np.median(near))


def _pixels_on_horizontal_plane(
    pixels: np.ndarray,
    plane_height_robot: float,
    observation: Mapping[str, Any],
    intrinsics: np.ndarray,
) -> np.ndarray:
    camera_to_robot = APRIL_TO_ROBOT @ np.asarray(
        observation["camera_to_april"], dtype=np.float64
    )
    inverse_intrinsics = np.linalg.inv(intrinsics)
    output = []
    for column, row in np.asarray(pixels, dtype=np.float64):
        ray_camera = inverse_intrinsics @ np.asarray([column, row, 1.0])
        ray_robot = camera_to_robot[:3, :3] @ ray_camera
        if abs(ray_robot[2]) < 1e-8:
            continue
        distance = (plane_height_robot - camera_to_robot[2, 3]) / ray_robot[2]
        if distance <= 0:
            continue
        point_robot = camera_to_robot[:3, 3] + distance * ray_robot
        output.append(point_robot[:2])
    return np.asarray(output, dtype=np.float64)


def _pose_robot_from_planar(
    center_xy: np.ndarray, center_height: float, yaw: float, normal_sign: float
) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    normal_sign = 1.0 if normal_sign >= 0.0 else -1.0
    local_x = np.asarray([cosine, sine, 0.0])
    local_y = np.asarray([0.0, 0.0, normal_sign])
    local_z = np.cross(local_x, local_y)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack([local_x, local_y, local_z], axis=1)
    pose[:3, 3] = [center_xy[0], center_xy[1], center_height]
    return pose


def _project_points(
    points_local: np.ndarray,
    pose_april: np.ndarray,
    observation: Mapping[str, Any],
    intrinsics: np.ndarray,
) -> np.ndarray:
    homogeneous = np.concatenate(
        [points_local, np.ones((len(points_local), 1), dtype=np.float64)], axis=1
    )
    points_camera = (
        np.linalg.inv(np.asarray(observation["camera_to_april"], dtype=np.float64))
        @ pose_april
        @ homogeneous.T
    ).T[:, :3]
    pixels_h = (intrinsics @ points_camera.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def _voxel_downsample_xy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return points
    _, indices = np.unique(
        np.floor(points / voxel_size).astype(np.int32),
        axis=0,
        return_index=True,
    )
    return points[np.sort(indices)]


def _trimmed_chamfer_score(
    observed_xy: np.ndarray,
    model_xy: np.ndarray,
    *,
    keep_fraction: float = 0.70,
) -> Tuple[float, float, float]:
    """Return a robust symmetric Chamfer score for two planar point clouds."""

    from scipy.spatial import cKDTree

    observed_tree = cKDTree(observed_xy)
    model_tree = cKDTree(model_xy)
    observed_distances = model_tree.query(observed_xy, k=1)[0]
    model_distances = observed_tree.query(model_xy, k=1)[0]

    def clipped_rms(distances: np.ndarray) -> float:
        cap = float(np.quantile(distances, keep_fraction))
        return float(np.sqrt(np.mean(np.minimum(distances, cap) ** 2)))

    observed_to_model = clipped_rms(observed_distances)
    model_to_observed = clipped_rms(model_distances)
    return (
        observed_to_model + model_to_observed,
        observed_to_model,
        model_to_observed,
    )


def _recover_pointcloud_pose(
    observations: Sequence[Mapping[str, Any]],
    masks: Mapping[int, np.ndarray],
    intrinsics: np.ndarray,
    mesh_vertices: np.ndarray,
    *,
    prompt_frame: int,
    prior_frame: int,
    push_end: int,
    place_start: int,
    output_dir: Path,
) -> Tuple[int, np.ndarray, float, Dict[str, Any]]:
    """Register the complete tabletop RGB-D surface to its CAD footprint.

    RealSense depth around the white tabletop boundary contains mixed pixels.
    Each mask is therefore completed on its robust horizontal depth plane,
    and all post-push mask pixels are accumulated into one surface cloud.  A
    dense CAD top footprint is matched with a symmetric trimmed Chamfer score.
    Socket detections are not used anywhere in this estimator.
    """

    prior_pose_april = _pose_vector_to_matrix(
        observations[prior_frame]["parts_poses"][:7]
    )
    prior_pose_robot = APRIL_TO_ROBOT @ prior_pose_april
    prior_yaw = float(
        math.atan2(prior_pose_robot[1, 0], prior_pose_robot[0, 0])
    )
    normal_sign = float(np.sign(prior_pose_robot[2, 1]))
    normal_sign = 1.0 if normal_sign >= 0.0 else -1.0
    surface_local_y = (
        float(np.max(mesh_vertices[:, 1]))
        if normal_sign > 0.0
        else float(np.min(mesh_vertices[:, 1]))
    )
    preferred_surface_height = float(
        prior_pose_robot[2, 3] + normal_sign * surface_local_y
    )

    # The tabletop is static as soon as the annotated push phase ends.  Keep
    # the complete post-push interval: early frames often expose an edge that
    # is occluded again while the robot reaches for the leg.
    start_frame = min(place_start, push_end + 1)
    sampled_frames = list(range(start_frame, place_start + 1, 3))
    if sampled_frames[-1] != place_start:
        sampled_frames.append(place_start)

    surface_clouds = []
    frame_details = []
    for frame_idx in sampled_frames:
        mask = masks.get(frame_idx)
        if mask is None:
            continue
        try:
            raw_points = _depth_points_robot(
                observations[frame_idx], mask, intrinsics
            )
            surface_height = _dominant_surface_height(
                raw_points,
                preferred_height=preferred_surface_height,
            )
        except ValueError:
            continue

        rows_columns = np.argwhere(np.asarray(mask, dtype=bool))[::5]
        if len(rows_columns) < 100:
            continue
        pixels = rows_columns[:, [1, 0]]
        surface_xy = _pixels_on_horizontal_plane(
            pixels,
            surface_height,
            observations[frame_idx],
            intrinsics,
        )
        if len(surface_xy) < 100:
            continue
        surface_clouds.append(surface_xy)
        depth_inliers = int(
            np.count_nonzero(
                np.abs(raw_points[:, 2] - surface_height) <= 0.007
            )
        )
        frame_details.append(
            {
                "frame_idx": frame_idx,
                "surface_height": surface_height,
                "mask_area": int(np.asarray(mask, dtype=bool).sum()),
                "depth_inliers": depth_inliers,
            }
        )

    if not surface_clouds:
        raise ValueError("No post-push RGB-D tabletop surface cloud was recovered")

    observed_xy = _voxel_downsample_xy(
        np.concatenate(surface_clouds, axis=0), 0.002
    )
    if len(observed_xy) < 500:
        raise ValueError("Tabletop surface cloud is too small for CAD registration")

    minimum = np.min(mesh_vertices[:, [0, 2]], axis=0)
    maximum = np.max(mesh_vertices[:, [0, 2]], axis=0)
    grid_x = np.linspace(minimum[0], maximum[0], 55)
    grid_z = np.linspace(minimum[1], maximum[1], 55)
    model_x, model_z = np.meshgrid(grid_x, grid_z)
    # _pose_robot_from_planar maps [local_x, -normal_sign * local_z]
    # through a conventional planar yaw rotation.
    model_planar = np.stack(
        [model_x.ravel(), -normal_sign * model_z.ravel()], axis=1
    )

    def candidate(yaw: float) -> Dict[str, Any]:
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=np.float64
        )
        observed_local = observed_xy @ rotation
        low = np.quantile(observed_local, 0.005, axis=0)
        high = np.quantile(observed_local, 0.995, axis=0)
        center_local = (low + high) * 0.5
        center_xy = center_local @ rotation.T
        model_xy = model_planar @ rotation.T + center_xy
        score, observed_to_model, model_to_observed = _trimmed_chamfer_score(
            observed_xy, model_xy
        )
        return {
            "yaw": yaw,
            "center": center_xy,
            "score": score,
            "observed_to_model": observed_to_model,
            "model_to_observed": model_to_observed,
            "observed_span": high - low,
        }

    coarse_yaws = np.linspace(
        prior_yaw - math.radians(15.0),
        prior_yaw + math.radians(15.0),
        61,
    )
    coarse_best = min((candidate(yaw) for yaw in coarse_yaws), key=lambda fit: fit["score"])
    fine_yaws = np.linspace(
        coarse_best["yaw"] - math.radians(0.75),
        coarse_best["yaw"] + math.radians(0.75),
        31,
    )
    best = min((candidate(yaw) for yaw in fine_yaws), key=lambda fit: fit["score"])

    surface_heights = np.asarray(
        [detail["surface_height"] for detail in frame_details], dtype=np.float64
    )
    surface_height = float(np.median(surface_heights))
    center_height = surface_height - normal_sign * surface_local_y
    pose_robot = _pose_robot_from_planar(
        best["center"], center_height, best["yaw"], normal_sign
    )
    pose_april = np.linalg.inv(APRIL_TO_ROBOT) @ pose_robot

    keyframe_detail = max(
        frame_details,
        key=lambda detail: (
            detail["depth_inliers"],
            detail["mask_area"],
            detail["frame_idx"],
        ),
    )
    keyframe = int(keyframe_detail["frame_idx"])
    height_spread = float(
        np.median(np.abs(surface_heights - surface_height))
    )
    confidence = float(
        np.clip(
            math.exp(-best["score"] / 0.008)
            * math.exp(-height_spread / 0.006)
            * min(1.0, len(frame_details) / 12.0),
            0.0,
            1.0,
        )
    )

    diagnostic = np.asarray(
        observations[keyframe]["color_image2"], dtype=np.uint8
    ).copy()
    outline_local = np.asarray(
        [
            [minimum[0], surface_local_y, minimum[1]],
            [maximum[0], surface_local_y, minimum[1]],
            [maximum[0], surface_local_y, maximum[1]],
            [minimum[0], surface_local_y, maximum[1]],
        ],
        dtype=np.float64,
    )
    outline_pixels = _project_points(
        outline_local, pose_april, observations[keyframe], intrinsics
    )
    cv2.polylines(
        diagnostic,
        [np.round(outline_pixels).astype(np.int32)],
        True,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    mask = np.asarray(masks[keyframe], dtype=np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(diagnostic, contours, -1, (0, 255, 255), 1)
    cv2.putText(
        diagnostic,
        (
            f"full-cloud frame={keyframe} chamfer={best['score'] * 1000:.1f}mm "
            f"frames={len(frame_details)} conf={confidence:.2f}"
        ),
        (4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "pointcloud_pose_keyframe.jpg"), diagnostic)

    details = {
        "method": "sam2_rgbd_full_tabletop_cad_chamfer",
        "uses_socket_detections": False,
        "prior_frame": prior_frame,
        "prior_yaw_deg": math.degrees(prior_yaw),
        "recovered_yaw_deg": math.degrees(best["yaw"]),
        "sampled_frame_count": len(frame_details),
        "surface_cloud_point_count": int(len(observed_xy)),
        "chamfer_m": float(best["score"]),
        "observed_to_model_m": float(best["observed_to_model"]),
        "model_to_observed_m": float(best["model_to_observed"]),
        "observed_span_m": best["observed_span"].tolist(),
        "cad_span_m": (maximum - minimum).tolist(),
        "surface_height_robot_m": surface_height,
        "surface_height_spread_m": height_spread,
        "cad_surface_local_y_m": surface_local_y,
        "preferred_surface_height_robot_m": preferred_surface_height,
        "center_robot": pose_robot[:3, 3].tolist(),
        "pose_robot": pose_robot.tolist(),
        "frame_details": frame_details,
    }
    return keyframe, pose_april, confidence, details


def _save_masks_and_overlay(
    observations: Sequence[Mapping[str, Any]],
    masks: Mapping[int, np.ndarray],
    output_dir: Path,
    prompt_frame: int,
    push_end: int,
    place_start: int,
) -> None:
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    first_image = np.asarray(observations[0]["color_image2"])
    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(
        str(output_dir / "sam2_mask_overlay.mpeg4.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError("Could not open SAM mask overlay video writer")
    try:
        for frame_idx in sorted(masks):
            mask = np.asarray(masks[frame_idx], dtype=bool)
            cv2.imwrite(str(mask_dir / f"{frame_idx:05d}.png"), mask.astype(np.uint8) * 255)
            image = np.asarray(observations[frame_idx]["color_image2"], dtype=np.uint8)
            overlay = image.copy()
            green = np.zeros_like(overlay)
            green[..., 1] = 255
            overlay[mask] = cv2.addWeighted(image[mask], 0.45, green[mask], 0.55, 0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
            label = (
                f"step={frame_idx} prompt={prompt_frame} "
                f"push_end={push_end} place={place_start} area={int(mask.sum())}"
            )
            cv2.rectangle(overlay, (0, 0), (width, 22), (0, 0, 0), -1)
            cv2.putText(
                overlay,
                label,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(overlay)
    finally:
        writer.release()


def run_mask_recovery(
    *,
    annotated_pickle: Path,
    output_dir: Path,
    mesh_path: Path,
    checkpoint: Path,
    model_config: str,
    device: str = "auto",
    output_json_name: str = "recovery.json",
) -> RecoveryResult:
    data = _load_pickle(annotated_pickle)
    observations = data["observations"]
    push_end, place_start = _phase_boundaries(observations)
    camera_info = data["camera_info"]
    prompt_frame, prior_frame = _flat_tag_frames(
        observations, APRIL_TO_ROBOT, push_end
    )
    prompt_observation = observations[prompt_frame]
    vertices = _load_obj_vertices(mesh_path)
    prompt_pose = _pose_vector_to_matrix(prompt_observation["parts_poses"][:7])
    prompt_image = np.asarray(prompt_observation["color_image2"], dtype=np.uint8)
    prompt_box = _project_mesh_box(
        vertices,
        prompt_pose,
        np.asarray(prompt_observation["camera_to_april"], dtype=np.float64),
        _intrinsics(camera_info),
        prompt_image.shape[:2],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    _write_video_frames(observations, frame_dir, place_start)
    cv2.imwrite(str(output_dir / "prompt_box.jpg"), _draw_prompt(prompt_image, prompt_box))
    masks_path = output_dir / "sam2_masks.npz"
    if masks_path.exists():
        masks = _load_saved_masks(masks_path)
    else:
        masks = _run_sam_video(
            frame_dir=frame_dir,
            prompt_frame=prompt_frame,
            prompt_box=prompt_box,
            checkpoint=checkpoint,
            model_config=model_config,
            device=device,
        )
        _save_masks_and_overlay(
            observations, masks, output_dir, prompt_frame, push_end, place_start
        )
        np.savez_compressed(
            masks_path,
            frame_indices=np.asarray(sorted(masks), dtype=np.int32),
            masks=np.stack([masks[idx] for idx in sorted(masks)]),
        )

    keyframe, pose_april_matrix, confidence, fit_details = (
        _recover_pointcloud_pose(
            observations,
            masks,
            _intrinsics(camera_info),
            vertices,
            prompt_frame=prompt_frame,
            prior_frame=prior_frame,
            push_end=push_end,
            place_start=place_start,
            output_dir=output_dir,
        )
    )
    pose_april = np.concatenate(
        [pose_april_matrix[:3, 3], _matrix_to_quat_xyzw(pose_april_matrix)]
    )
    result = RecoveryResult(
        trajectory=annotated_pickle.name,
        part_name=TABLETOP_NAME,
        prompt_frame=prompt_frame,
        push_end_frame=push_end,
        place_start_frame=place_start,
        keyframe=keyframe,
        start_frame=push_end + 1,
        pose_april=pose_april.astype(float).tolist(),
        confidence=confidence,
        source="sam2_rgbd_full_tabletop_cad_chamfer",
        diagnostics_dir=str(output_dir),
        fit_details=fit_details,
    )
    with (output_dir / output_json_name).open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotated_pickle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="SAM2 inference device; auto uses CUDA when available.",
    )
    parser.add_argument(
        "--output-json-name",
        default="recovery.json",
        help="Recovery JSON filename below --output-dir.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_mask_recovery(
        annotated_pickle=args.annotated_pickle.resolve(),
        output_dir=args.output_dir.resolve(),
        mesh_path=args.mesh.resolve(),
        checkpoint=args.checkpoint.resolve(),
        model_config=args.model_config,
        device=args.device,
        output_json_name=args.output_json_name,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
