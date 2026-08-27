from __future__ import annotations

from datetime import datetime
from typing import List
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.visualization.render_mp4 import pickle_data
from src.common.types import Trajectory, Observation
from src.common.geometry import np_action_6d_to_quat
from src.common.eepose import ROBOT_BASE, SIM_LOCAL

from ipdb import set_trace as bp
from src.visualization.render_mp4 import (
    create_in_memory_mp4,
    depth2heatmap,
    analyze_depth_smoothness,
)
from src.eval.skill_annotation_util import draw_skill_on_image


def _front_point(point_mapping) -> np.ndarray | None:
    """Return a finite front-camera point from an annotation mapping."""
    if not isinstance(point_mapping, dict):
        return None
    point = point_mapping.get("color_image2")
    if point is None:
        return None
    point = np.asarray(point, dtype=np.float32)
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return point


def draw_vlm_rollout_debug_frames(
    frames: np.ndarray,
    vlm_skills: List[str],
    vlm_points_2d: List[dict],
    oracle_skills: List[str],
    oracle_points_2d: List[dict],
    vlm_annotations: List[dict],
) -> np.ndarray:
    """Overlay VLM/GT points and their pixel error on front-camera frames.

    Images in this project are RGB arrays.  OpenCV is only used as a drawing
    primitive here, so the color tuples below intentionally use RGB ordering.
    """
    annotated_frames = np.asarray(frames).copy()
    n_frames = min(
        len(annotated_frames),
        len(vlm_skills),
        len(vlm_points_2d),
        len(oracle_skills),
        len(oracle_points_2d),
        len(vlm_annotations),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    vlm_color = (255, 32, 32)
    gt_color = (32, 255, 32)
    link_color = (255, 255, 32)

    for frame_idx in range(n_frames):
        frame = annotated_frames[frame_idx]
        if frame.ndim != 3 or frame.shape[-1] != 3:
            continue
        height, width = frame.shape[:2]
        vlm_point = _front_point(vlm_points_2d[frame_idx])
        gt_point = _front_point(oracle_points_2d[frame_idx])

        def visible(point: np.ndarray | None) -> bool:
            return bool(
                point is not None
                and 0 <= point[0] < width
                and 0 <= point[1] < height
            )

        vlm_xy = tuple(np.rint(vlm_point).astype(int)) if visible(vlm_point) else None
        gt_xy = tuple(np.rint(gt_point).astype(int)) if visible(gt_point) else None
        error_px = (
            float(np.linalg.norm(vlm_point - gt_point))
            if vlm_point is not None and gt_point is not None
            else None
        )

        if vlm_xy is not None and gt_xy is not None:
            cv2.line(frame, vlm_xy, gt_xy, link_color, 1, cv2.LINE_AA)
        if vlm_xy is not None:
            cv2.circle(frame, vlm_xy, 6, vlm_color, 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "V",
                (min(vlm_xy[0] + 7, width - 12), max(vlm_xy[1] - 5, 12)),
                font,
                0.38,
                vlm_color,
                1,
                cv2.LINE_AA,
            )
        if gt_xy is not None:
            cv2.drawMarker(
                frame,
                gt_xy,
                gt_color,
                cv2.MARKER_CROSS,
                13,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "G",
                (min(gt_xy[0] + 7, width - 12), max(gt_xy[1] - 5, 12)),
                font,
                0.38,
                gt_color,
                1,
                cv2.LINE_AA,
            )

        metadata = vlm_annotations[frame_idx]
        metadata = metadata if isinstance(metadata, dict) else {}
        confidence = metadata.get("skill_confidence")
        age = metadata.get("cache_age_steps")
        confidence_text = (
            f"{float(confidence):.2f}" if confidence is not None else "n/a"
        )
        error_text = f"{error_px:.1f}px" if error_px is not None else "n/a"
        vlm_skill = vlm_skills[frame_idx] or "n/a"
        oracle_skill = oracle_skills[frame_idx] or "n/a"
        age_text = str(age) if age is not None else "n/a"
        cv2.rectangle(frame, (0, 0), (width - 1, 40), (0, 0, 0), -1)
        cv2.putText(
            frame,
            (
                f"step {frame_idx:04d}  VLM={vlm_skill}({confidence_text})  "
                f"GT={oracle_skill}"
            ),
            (5, 15),
            font,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"error={error_text}  cache_age={age_text}  V=red G=green",
            (5, 33),
            font,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated_frames


def _write_depth_smoothness_report(report_path: Path, camera_name: str, smoothness: dict):
    lines = [
        f"camera={camera_name}",
        f"depth_sign_mode={smoothness.get('depth_sign_mode', 'as_is')}",
        f"valid_pixel_ratio_global={smoothness.get('valid_pixel_ratio_global', 0.0):.6f}",
        f"global_min_p1={smoothness['global_min']:.6f}",
        f"global_max_p99={smoothness['global_max']:.6f}",
        f"jump_threshold_p95={smoothness['threshold']:.6f}",
        f"jump_frames={smoothness['n_jumps']}/{max(smoothness['n_frames'] - 1, 0)}",
        "frame_idx,valid_ratio,depth_mean,depth_p95,delta_mean,delta_p95,delta_max,status",
    ]
    for row in smoothness["per_frame"]:
        status = "JUMP" if row["is_jump"] else "OK"
        lines.append(
            f"{row['frame']},{row['valid_ratio']:.6f},{row['depth_mean']:.6f},"
            f"{row['depth_p95']:.6f},{row['delta_mean']:.6f},{row['delta_p95']:.6f},"
            f"{row['delta_max']:.6f},{status}"
        )
    report_path.write_text("\n".join(lines) + "\n")


def save_raw_rollout(
    robot_states: np.ndarray,
    imgs1: np.ndarray,
    imgs2: np.ndarray,
    depth_image1: np.ndarray,
    depth_image2: np.ndarray,
    skills: List[str],
    guidance_points: List[np.ndarray],
    guidance_poses: List[np.ndarray],
    guidance_gripper_widths: List[float],
    guidance_points_2d: List[dict],
    grasp_annotations_2d: List[dict],
    camera_infos: List[dict],
    actions: np.ndarray,
    rewards: np.ndarray,
    parts_poses: np.ndarray,
    success: bool,
    task: str,
    action_type: str,
    rollout_save_dir: Path,
    compress_pickles: bool = False,
    have_img_obs:  bool = False,
    have_depth_obs: bool = False,
    pcs: List[np.ndarray] = None,
    skill_on_image: bool = False,
    output_only_pickle: bool = False,
    output_only_video: bool = False,
    guidance_points_clean: List[np.ndarray] = None,
    guidance_poses_clean: List[np.ndarray] = None,
    oracle_skills: List[str] = None,
    oracle_guidance_points_2d: List[dict] = None,
    vlm_annotations: List[dict] = None,
    vlm_point_error_records: List[dict] = None,
    annotation_source: str = "scripted",
    vlm_model_revision: str = None,
    eepose_frame: str = ROBOT_BASE,
    eepose_original_frame: str = SIM_LOCAL,
    policy_eepose_frame: str = ROBOT_BASE,
):
    observations: List[Observation] = list()
    include_vlm_metadata = any(
        value is not None
        for value in (
            oracle_skills,
            oracle_guidance_points_2d,
            vlm_annotations,
            vlm_point_error_records,
        )
    )

    # If pcs is None, create a list of Nones with the same length as robot_states
    if pcs is None:
        pcs = [None] * len(robot_states)

    if skills is None:
        skills = [None] * len(robot_states)
    if guidance_points is None:
        guidance_points = [None] * len(robot_states)
    if guidance_points_clean is None:
        guidance_points_clean = [None] * len(robot_states)
    if guidance_poses is None:
        guidance_poses = [None] * len(robot_states)
    if guidance_poses_clean is None:
        guidance_poses_clean = [None] * len(robot_states)
    if guidance_gripper_widths is None:
        guidance_gripper_widths = [None] * len(robot_states)
    if guidance_points_2d is None:
        guidance_points_2d = [None] * len(robot_states)
    if grasp_annotations_2d is None:
        grasp_annotations_2d = [None] * len(robot_states)
    if camera_infos is None:
        camera_infos = [None] * len(robot_states)
    if oracle_skills is None:
        oracle_skills = [None] * len(robot_states)
    if oracle_guidance_points_2d is None:
        oracle_guidance_points_2d = [None] * len(robot_states)
    if vlm_annotations is None:
        vlm_annotations = [None] * len(robot_states)
    error_by_step = {
        int(record["step_idx"]): record
        for record in (vlm_point_error_records or [])
    }

    rows = zip(
        robot_states,
        imgs1,
        imgs2,
        depth_image1,
        depth_image2,
        parts_poses,
        pcs,
        skills,
        guidance_points,
        guidance_points_clean,
        guidance_poses,
        guidance_poses_clean,
        guidance_gripper_widths,
        guidance_points_2d,
        grasp_annotations_2d,
        oracle_skills,
        oracle_guidance_points_2d,
        vlm_annotations,
    )
    for frame_idx, row in enumerate(rows):
        (
            robot_state,
            image1,
            image2,
            depth1,
            depth2,
            parts_pose,
            pc,
            skill,
            guidance_point,
            guidance_point_clean,
            guidance_pose,
            guidance_pose_clean,
            guidance_gripper_width,
            guidance_point_2d,
            grasp_annotation_2d,
            oracle_skill,
            oracle_guidance_point_2d,
            vlm_annotation,
        ) = row
        observation = {
            "robot_state": robot_state,
            "color_image1": image1,
            "color_image2": image2,
            "depth_image1": depth1,
            "depth_image2": depth2,
            "parts_poses": parts_pose,
            "point_cloud": pc,
            "skill": skill,
            "guidance_point": guidance_point,
            "guidance_point_clean": guidance_point_clean,
            "guidance_pose": guidance_pose,
            "guidance_pose_clean": guidance_pose_clean,
            "guidance_gripper_width": guidance_gripper_width,
            "guidance_point_2d": guidance_point_2d,
            "grasp_annotation_2d": grasp_annotation_2d,
        }
        if include_vlm_metadata:
            observation.update(
                {
                    "oracle_skill": oracle_skill,
                    "oracle_guidance_point_2d": oracle_guidance_point_2d,
                    "vlm_annotation": vlm_annotation,
                    "vlm_point_error": error_by_step.get(frame_idx),
                }
            )
        observations.append(observation)

    front_camera_info = None
    for camera_info in camera_infos:
        if not isinstance(camera_info, dict):
            continue

        if front_camera_info is None and "color_image2" in camera_info:
            front_camera_info = camera_info["color_image2"]

    if action_type == "pos":

        if actions.shape[1] == 10:
            # If we've used rot_6d convert to quat
            actions = np_action_6d_to_quat(actions)
        
        assert actions.shape[1] == 8

        # Get the action quat
        pos_action_quat = R.from_quat(actions[:, 3:7])

        # Get the position quat from the robot state
        pos_quat = R.from_quat([rs["ee_quat"] for rs in robot_states[:-1]])

        # The action quat was calculated as pos_quat * action_quat
        # Calculate the delta quat between the pos_quat and the action_quat
        delta_action_quat = pos_quat.inv() * pos_action_quat

        # Also calculate the delta position
        delta_action_pos = actions[:, :3] - np.array(
            [rs["ee_pos"] for rs in robot_states[:-1]]
        )

        # Insert the delta quat into the actions
        actions = np.concatenate(
            [delta_action_pos, delta_action_quat.as_quat(), actions[:, -1:]], axis=1
        )

    data: Trajectory = {
        "env": "FurnitureBench",
        "observations": observations,
        "actions": actions.tolist(),
        "rewards": rewards.tolist(),
        "camera_info": {
            "front_camera": front_camera_info,
        },
        "success": success,
        "task": task,
        "action_type": action_type,
        "annotation_source": annotation_source,
        "vlm_model_revision": vlm_model_revision,
        "eepose_frame": eepose_frame,
        "eepose_original_frame": eepose_original_frame,
        "policy_eepose_frame": policy_eepose_frame,
        "eepose_schema_version": 2,
    }

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    output_path = rollout_save_dir / ("success" if success else "failure")
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"{timestamp}.pkl"

    if compress_pickles:
        output_path = output_path.with_suffix(".pkl.xz")

    if not output_only_video:
        pickle_data(data, output_path)

    if output_only_pickle:
        return

    # Additionally save MP4 videos for video1 and video2
    if have_img_obs:
        # Ensure output directory exists (with success/failure subdirectory)
        status_dir = Path(rollout_save_dir) / ("success" if success else "failure")
        status_dir.mkdir(parents=True, exist_ok=True)

        imgs2_for_video = imgs2.copy()
        if skill_on_image:
            n_annotated = min(len(imgs2_for_video), len(skills))
            for frame_idx in range(n_annotated):
                skill = skills[frame_idx]
                if skill is None:
                    continue
                imgs2_for_video[frame_idx] = draw_skill_on_image(
                    imgs2_for_video[frame_idx], skill
                )

        # Create MP4 bytes for each camera stream
        mp4_cam1 = create_in_memory_mp4(imgs1, fps=20)
        mp4_cam2 = create_in_memory_mp4(imgs2_for_video, fps=20)
        mp4_cam2_vlm_debug = None
        if annotation_source == "vlm":
            vlm_debug_frames = draw_vlm_rollout_debug_frames(
                imgs2,
                skills,
                guidance_points_2d,
                oracle_skills,
                oracle_guidance_points_2d,
                vlm_annotations,
            )
            mp4_cam2_vlm_debug = create_in_memory_mp4(vlm_debug_frames, fps=20)

        # Build filenames
        cam1_path = status_dir / f"{timestamp}_cam1.mp4"
        cam2_path = status_dir / f"{timestamp}_cam2.mp4"
        cam2_vlm_debug_path = status_dir / f"{timestamp}_cam2_vlm_debug.mp4"

        # Write files
        with open(cam1_path, "wb") as f1:
            f1.write(mp4_cam1.getvalue() if hasattr(mp4_cam1, "getvalue") else mp4_cam1)
        with open(cam2_path, "wb") as f2:
            f2.write(mp4_cam2.getvalue() if hasattr(mp4_cam2, "getvalue") else mp4_cam2)
        if mp4_cam2_vlm_debug is not None:
            with open(cam2_vlm_debug_path, "wb") as f2_debug:
                f2_debug.write(
                    mp4_cam2_vlm_debug.getvalue()
                    if hasattr(mp4_cam2_vlm_debug, "getvalue")
                    else mp4_cam2_vlm_debug
                )

    # Additionally save depth videos as MP4 for video1 and video2
    if have_depth_obs:
        # Ensure output directory exists (with success/failure subdirectory)
        status_dir = Path(rollout_save_dir) / ("success" if success else "failure")
        status_dir.mkdir(parents=True, exist_ok=True)

        smooth1 = analyze_depth_smoothness(depth_image1)
        smooth2 = analyze_depth_smoothness(depth_image2)
        report1_path = status_dir / f"{timestamp}_dep1_smoothness.txt"
        report2_path = status_dir / f"{timestamp}_dep2_smoothness.txt"
        _write_depth_smoothness_report(report1_path, "depth_image1", smooth1)
        _write_depth_smoothness_report(report2_path, "depth_image2", smooth2)
        print(
            f"[DepthSmoothness] dep1 jump_frames={smooth1['n_jumps']}/"
            f"{max(smooth1['n_frames'] - 1, 0)}, threshold={smooth1['threshold']:.6f}, "
            f"report={report1_path}"
        )
        print(
            f"[DepthSmoothness] dep2 jump_frames={smooth2['n_jumps']}/"
            f"{max(smooth2['n_frames'] - 1, 0)}, threshold={smooth2['threshold']:.6f}, "
            f"report={report2_path}"
        )

        # Create MP4 bytes for each camera stream
        depth1_heatmap_frames = depth2heatmap(depth_image1)
        depth2_heatmap_frames = depth2heatmap(depth_image2)

        mp4_dep1 = create_in_memory_mp4(depth1_heatmap_frames, fps=20)
        mp4_dep2 = create_in_memory_mp4(depth2_heatmap_frames, fps=20)

        # Build filenames
        dep1_path = status_dir / f"{timestamp}_dep1.mp4"
        dep2_path = status_dir / f"{timestamp}_dep2.mp4"

        # Write files
        with open(dep1_path, "wb") as f1:
            f1.write(mp4_dep1.getvalue() if hasattr(mp4_dep1, "getvalue") else mp4_dep1)
        with open(dep2_path, "wb") as f2:
            f2.write(mp4_dep2.getvalue() if hasattr(mp4_dep2, "getvalue") else mp4_dep2)
