"""Offline skill/target-point annotation for real-robot trajectory pickles.

The simulator annotator consumes a live Isaac Gym environment whose rigid-body
poses and contact forces are complete on every step.  Real demonstrations have
neither guarantee: AprilTag poses disappear under occlusion and the saved
robot state does not contain simulator contact forces.  This module keeps those
real-only policies out of :mod:`src.eval.skill_annotation_util` while reusing
its furniture skill definitions through ``SkillAnnotator`` inheritance.

The first supported real-data schema is ``deoxys_furniturebench_raw_v2`` for
``one_leg``.  Each output observation receives the standard fields consumed by
the offline image-annotation pipeline, including ``skill``,
``guidance_point`` and ``guidance_point_2d``.
"""

from __future__ import annotations

import argparse
import gzip
import lzma
import os
import pickle
import tempfile
from collections import Counter
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import furniture_bench.controllers.control_utils as C
import furniture_bench.utils.transform as T
from furniture_bench.config import config as furniture_bench_config

from src.eval.skill_annotation_util import (
    SkillAnnotator,
    _build_grasp_rect_points_3d,
    _pose_to_numpy,
    _to_numpy,
)
from src.eval.real_pose_provider import PartPoseEstimate, PartPoseProvider


ANNOTATION_SOURCE = "real_skill_annotation_util"
ANNOTATION_VERSION = 8
DEFAULT_POSE_TRACKING_POLICY = "april_tag_then_ee_rigid"
SUPPORTED_FURNITURE = {"one_leg"}
_OBSTACLE_NAMES = ("obstacle_front", "obstacle_right", "obstacle_left")
PLACE_TARGET_POLICY_TABLETOP = "tabletop_max_xy_socket_aligned"
_TABLETOP_SOCKET_SIGNS = (
    (1, 1.0, 1.0),
    (2, -1.0, 1.0),
    (3, 1.0, -1.0),
    (4, -1.0, -1.0),
)
# A live leg pose is unreliable in the real demonstrations.  Approximate the
# leg-center-to-EE separation with one quarter of the leg's longest dimension,
# applied along robot/world +Z with no lateral leg-pose term.
LEG_TO_EE_LENGTH_FRACTION = 0.25


def _pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"Expected finite xyzw pose with shape (7,), got {pose.shape}")
    quat_norm = float(np.linalg.norm(pose[3:]))
    if quat_norm < 1e-6:
        raise ValueError("Pose quaternion has zero norm")
    normalized = pose.copy()
    normalized[3:] /= quat_norm
    return T.pose2mat(normalized).astype(np.float32)


def _matrix_to_pose_vector(pose: np.ndarray) -> np.ndarray:
    pos, quat = T.mat2pose(np.asarray(pose, dtype=np.float32))
    return np.concatenate([pos, quat]).astype(np.float32)


def _ee_pose_robot(observation: Mapping[str, Any]) -> np.ndarray:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, Mapping):
        raise ValueError("Real annotation requires observation['robot_state'] to be a mapping")

    ee_pose = robot_state.get("ee_pose")
    if ee_pose is not None:
        ee_pose = np.asarray(ee_pose, dtype=np.float32)
        if ee_pose.shape == (4, 4) and np.isfinite(ee_pose).all():
            return ee_pose

    if "ee_pos" not in robot_state or "ee_quat" not in robot_state:
        raise ValueError("robot_state must contain ee_pose or both ee_pos and ee_quat")
    return _pose_vector_to_matrix(
        np.concatenate([robot_state["ee_pos"], robot_state["ee_quat"]])
    )


def _intrinsics_matrix(camera_config: Mapping[str, Any]) -> np.ndarray:
    values = camera_config.get("record_intrinsics", camera_config.get("intrinsics"))
    if not isinstance(values, Mapping):
        raise ValueError("Camera metadata is missing record intrinsics")
    return np.asarray(
        [
            [values["fx"], 0.0, values["ppx"]],
            [0.0, values["fy"], values["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _camera_image_size(camera_config: Mapping[str, Any]) -> Tuple[int, int]:
    values = camera_config.get("record_intrinsics", camera_config.get("intrinsics"))
    if not isinstance(values, Mapping):
        raise ValueError("Camera metadata is missing image dimensions")
    return int(values["width"]), int(values["height"])


def _project_point(
    point: np.ndarray,
    *,
    point_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    point_h = np.ones(4, dtype=np.float32)
    point_h[:3] = np.asarray(point, dtype=np.float32)
    point_camera = np.asarray(point_to_camera, dtype=np.float32) @ point_h
    if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-6:
        return None

    pixel_h = intrinsics @ point_camera[:3]
    pixel = pixel_h[:2] / pixel_h[2]
    if not np.isfinite(pixel).all():
        return None
    width, height = image_size
    if pixel[0] < 0 or pixel[0] >= width or pixel[1] < 0 or pixel[1] >= height:
        return None
    return pixel.astype(np.float32)


def _project_polygon(
    points: np.ndarray,
    *,
    point_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    pixels = []
    for point in np.asarray(points, dtype=np.float32):
        pixel = _project_point(
            point,
            point_to_camera=point_to_camera,
            intrinsics=intrinsics,
            image_size=image_size,
        )
        if pixel is None:
            return None
        pixels.append(pixel)
    return np.stack(pixels, axis=0) if pixels else None


@dataclass
class _TrackedPart:
    pose_april: Optional[np.ndarray] = None
    ee_to_part_robot: Optional[np.ndarray] = None
    attached: bool = False
    source: str = "uninitialized"
    pending_ee_to_part_robot: Optional[np.ndarray] = None
    pending_detection_count: int = 0


@dataclass
class RealAnnotationStats:
    frame_count: int = 0
    skill_counts: Counter = field(default_factory=Counter)
    pose_source_counts: Counter = field(default_factory=Counter)
    missing_detection_counts: Counter = field(default_factory=Counter)
    projected_front_count: int = 0
    projected_wrist_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "skill_counts": dict(sorted(self.skill_counts.items())),
            "pose_source_counts": dict(sorted(self.pose_source_counts.items())),
            "missing_detection_counts": dict(
                sorted(self.missing_detection_counts.items())
            ),
            "projected_front_count": self.projected_front_count,
            "projected_wrist_count": self.projected_wrist_count,
        }


@dataclass
class RealSkillAnnotator(SkillAnnotator):
    """Real-pickle annotator with occlusion-aware part tracking.

    ``parts_founds`` is authoritative.  A raw pose from a not-found part is
    treated as a cached/bootstrap value, never as a fresh detection.  Once the
    gripper closes on the active part, its pose is propagated using a fixed
    EE-to-part transform until the gripper opens; otherwise the last reliable
    pose is held.
    """

    close_width_m: float = 0.055
    open_width_m: float = 0.060
    attachment_distance_m: float = 0.25
    attached_detection_gate_m: float = 0.040
    relocalization_gate_m: float = 0.030
    relocalization_frames: int = 3
    table_release_displacement_m: float = 0.10

    def __post_init__(self):
        if self.furniture_name not in SUPPORTED_FURNITURE:
            raise ValueError(
                f"Real annotation currently supports {sorted(SUPPORTED_FURNITURE)}, "
                f"got {self.furniture_name!r}"
            )
        super().__post_init__()
        self.april_to_robot = np.asarray(
            furniture_bench_config["robot"]["tag_base_from_robot_base"],
            dtype=np.float32,
        )
        self.robot_to_april = np.linalg.inv(self.april_to_robot).astype(np.float32)
        self._tracked_parts = {
            part.name: _TrackedPart() for part in self.furniture.parts
        }
        self._gripper_closed = False
        self._attached_part_name: Optional[str] = None
        self._initial_part_pose_robot: Dict[str, np.ndarray] = {}
        self._current_gripper_event: Optional[str] = None
        self._placed_part_names = set()
        self._frame_idx = -1
        self.stats = RealAnnotationStats()

    def reset(self):
        super().reset()
        self._tracked_parts = {
            part.name: _TrackedPart() for part in self.furniture.parts
        }
        self._gripper_closed = False
        self._attached_part_name = None
        self._initial_part_pose_robot = {}
        self._current_gripper_event = None
        self._placed_part_names = set()
        self._frame_idx = -1
        self.stats = RealAnnotationStats()

    def _gripper_event(self, width: float) -> Optional[str]:
        if not self._gripper_closed and width <= self.close_width_m:
            self._gripper_closed = True
            return "closed"
        if self._gripper_closed and width >= self.open_width_m:
            self._gripper_closed = False
            return "opened"
        return None

    def _active_part(self):
        if self.assemble_idx >= len(self.furniture.should_be_assembled):
            return None
        part1_idx, part2_idx = self.furniture.should_be_assembled[self.assemble_idx]
        part1 = self.furniture.parts[part1_idx]
        part2 = self.furniture.parts[part2_idx]
        part1_active = (
            not getattr(part1, "pre_assemble_done", True)
            and getattr(part1, "skill_state", None) != "done"
        )
        return part1 if part1_active else part2

    def _raw_pose_status(
        self, observation: Mapping[str, Any], part_idx: int
    ) -> Tuple[np.ndarray, bool, bool]:
        poses = np.asarray(observation.get("parts_poses"), dtype=np.float32).reshape(-1)
        required = (part_idx + 1) * 7
        if poses.size < required:
            raise ValueError(
                f"parts_poses has {poses.size} values; part {part_idx} requires {required}"
            )
        pose = poses[part_idx * 7 : required].copy()

        founds = observation.get("parts_founds")
        found = True if founds is None else bool(np.asarray(founds)[part_idx])
        validity = observation.get("parts_pose_valid")
        valid_flag = True if validity is None else bool(np.asarray(validity)[part_idx])
        numerically_valid = (
            np.isfinite(pose).all()
            and not np.allclose(pose, 0.0)
            and float(np.linalg.norm(pose[3:])) > 1e-6
        )
        return pose, found and valid_flag and numerically_valid, valid_flag and numerically_valid

    def _tracked_pose(
        self,
        observation: Mapping[str, Any],
        part,
        ee_pose_robot: np.ndarray,
    ) -> np.ndarray:
        raw_pose, detected, usable_cached_pose = self._raw_pose_status(
            observation, part.part_idx
        )
        tracker = self._tracked_parts[part.name]
        override_sources = observation.get("_real_pose_override_sources", {})
        detected_source = str(override_sources.get(part.name, "detected"))

        if detected and part.name in override_sources:
            tracker.pose_april = raw_pose
            tracker.source = detected_source
            tracker.pending_ee_to_part_robot = None
            tracker.pending_detection_count = 0
            if tracker.attached:
                detected_pose_robot = (
                    self.april_to_robot @ _pose_vector_to_matrix(raw_pose)
                )
                tracker.ee_to_part_robot = (
                    np.linalg.inv(ee_pose_robot) @ detected_pose_robot
                ).astype(np.float32)
        elif detected:
            detected_pose_robot = self.april_to_robot @ _pose_vector_to_matrix(raw_pose)
            if tracker.attached and tracker.ee_to_part_robot is not None:
                predicted_pose_robot = ee_pose_robot @ tracker.ee_to_part_robot
                detection_error = float(
                    np.linalg.norm(
                        detected_pose_robot[:3, 3] - predicted_pose_robot[:3, 3]
                    )
                )
                if detection_error > self.attached_detection_gate_m:
                    candidate_ee_to_part = (
                        np.linalg.inv(ee_pose_robot) @ detected_pose_robot
                    ).astype(np.float32)
                    pending = tracker.pending_ee_to_part_robot
                    if pending is not None and np.linalg.norm(
                        candidate_ee_to_part[:3, 3] - pending[:3, 3]
                    ) <= self.relocalization_gate_m:
                        tracker.pending_detection_count += 1
                    else:
                        tracker.pending_detection_count = 1
                    tracker.pending_ee_to_part_robot = candidate_ee_to_part
                    if tracker.pending_detection_count >= self.relocalization_frames:
                        tracker.pose_april = raw_pose
                        tracker.ee_to_part_robot = candidate_ee_to_part
                        tracker.pending_ee_to_part_robot = None
                        tracker.pending_detection_count = 0
                        tracker.source = "relocalized_detection"
                    else:
                        tracker.pose_april = _matrix_to_pose_vector(
                            self.robot_to_april @ predicted_pose_robot
                        )
                        tracker.source = "rejected_detection_ee"
                else:
                    tracker.pose_april = raw_pose
                    tracker.source = detected_source
                    tracker.ee_to_part_robot = (
                        np.linalg.inv(ee_pose_robot) @ detected_pose_robot
                    ).astype(np.float32)
                    tracker.pending_ee_to_part_robot = None
                    tracker.pending_detection_count = 0
            else:
                tracker.pose_april = raw_pose
                tracker.source = detected_source
        elif tracker.attached and tracker.ee_to_part_robot is not None:
            part_pose_robot = ee_pose_robot @ tracker.ee_to_part_robot
            tracker.pose_april = _matrix_to_pose_vector(
                self.robot_to_april @ part_pose_robot
            )
            tracker.source = "ee_propagated"
        elif tracker.pose_april is not None:
            tracker.source = "held_last"
        elif usable_cached_pose:
            tracker.pose_april = raw_pose
            tracker.source = "bootstrap_cached"
        else:
            raise ValueError(
                f"No usable pose for part {part.name!r} at frame {self._frame_idx}"
            )

        if not detected:
            tracker.pending_ee_to_part_robot = None
            tracker.pending_detection_count = 0
            self.stats.missing_detection_counts[part.name] += 1
        self.stats.pose_source_counts[f"{part.name}:{tracker.source}"] += 1
        if part.name not in self._initial_part_pose_robot:
            self._initial_part_pose_robot[part.name] = (
                self.april_to_robot @ _pose_vector_to_matrix(tracker.pose_april)
            ).astype(np.float32)
        return tracker.pose_april.copy()

    def _part_displacement_from_start(
        self, part_name: str, annotation_inputs: Mapping[str, Any]
    ) -> float:
        initial_pose = self._initial_part_pose_robot.get(part_name)
        if initial_pose is None:
            return 0.0
        idx = annotation_inputs["part_idxs"][part_name][0]
        pose_april = annotation_inputs["rb_states"][idx]
        current_pose_april = C.to_homogeneous(
            pose_april[:3], C.quat2mat(pose_april[3:7])
        )
        current_pose_robot = annotation_inputs["april_to_robot_mat"] @ current_pose_april
        return float(
            torch.linalg.norm(
                current_pose_robot[:2, 3]
                - torch.as_tensor(initial_pose[:2, 3], dtype=torch.float32)
            ).item()
        )

    @staticmethod
    def _longest_part_length(operated_part) -> float:
        dimensions = [
            float(getattr(operated_part, attribute))
            for attribute in ("reset_x_len", "reset_y_len", "reset_z_len")
            if hasattr(operated_part, attribute)
            and getattr(operated_part, attribute) is not None
        ]
        if not dimensions:
            raise ValueError(
                f"Part {operated_part.name!r} has no reset dimensions"
            )
        return max(dimensions)

    def _tabletop_place_target_details(
        self,
        operated_part,
        assemble_to: str,
        annotation_inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return max-XY socket, target leg center, and vertical EE target.

        The four sockets come from the CAD layout.  Their robot-frame ``x+y``
        scores select the physically operated corner (socket 3 in the current
        three demonstrations).  Half the leg's longest dimension moves from
        the socket to the target leg center.  A separate nominal leg-to-EE
        offset then gives the EE guidance point.  Both offsets are strictly
        along robot/world +Z and do not depend on a live leg pose.
        """

        table_idx = annotation_inputs["part_idxs"][assemble_to][0]
        table_pose = annotation_inputs["rb_states"][table_idx]
        table_pose_april = C.to_homogeneous(
            table_pose[:3], C.quat2mat(table_pose[3:7])
        )
        table_pose_robot = (
            annotation_inputs["april_to_robot_mat"]
            @ annotation_inputs["sim_to_april_mat"]
            @ table_pose_april
        )
        socket_offset_x = abs(
            float(operated_part.default_assembled_pose[0, 3])
        )
        socket_offset_z = abs(
            float(operated_part.default_assembled_pose[2, 3])
        )
        socket_locals = torch.as_tensor(
            [
                [sign_x * socket_offset_x, 0.0, sign_z * socket_offset_z, 1.0]
                for _, sign_x, sign_z in _TABLETOP_SOCKET_SIGNS
            ],
            dtype=table_pose_robot.dtype,
            device=table_pose_robot.device,
        )
        socket_robots = (table_pose_robot @ socket_locals.T).T[:, :3]
        socket_scores = socket_robots[:, 0] + socket_robots[:, 1]
        selected_index = int(torch.argmax(socket_scores).item())
        socket_label = _TABLETOP_SOCKET_SIGNS[selected_index][0]
        socket_local = socket_locals[selected_index, :3].clone()

        longest_leg_length = self._longest_part_length(operated_part)
        half_leg_length = longest_leg_length * 0.5
        leg_to_ee_z_offset = (
            longest_leg_length * LEG_TO_EE_LENGTH_FRACTION
        )
        robot_z = torch.as_tensor(
            [0.0, 0.0, 1.0],
            dtype=table_pose_robot.dtype,
            device=table_pose_robot.device,
        )
        socket_robot = socket_robots[selected_index]
        leg_center_robot = socket_robot + robot_z * half_leg_length
        guidance_robot = leg_center_robot + robot_z * leg_to_ee_z_offset

        def inverse_transform(robot_point: torch.Tensor) -> torch.Tensor:
            homogeneous = torch.ones(
                4,
                dtype=table_pose_robot.dtype,
                device=table_pose_robot.device,
            )
            homogeneous[:3] = robot_point
            return (torch.linalg.inv(table_pose_robot) @ homogeneous)[:3]

        leg_center_local = inverse_transform(leg_center_robot)
        guidance_local = inverse_transform(guidance_robot)

        return {
            "socket_label": socket_label,
            "socket_local": socket_local,
            "socket_robot": socket_robot,
            "socket_scores": socket_scores,
            "longest_leg_length_m": longest_leg_length,
            "half_leg_length_m": half_leg_length,
            "leg_center_local": leg_center_local,
            "leg_center_robot": leg_center_robot,
            "leg_to_ee_z_offset_m": leg_to_ee_z_offset,
            "total_z_offset_m": half_leg_length + leg_to_ee_z_offset,
            "guidance_local": guidance_local,
            "guidance_robot": guidance_robot,
        }

    def _obstacle_poses(self, observation: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        poses = np.asarray(observation.get("parts_poses"), dtype=np.float32).reshape(-1)
        furniture_pose_size = len(self.furniture.parts) * 7
        if poses.size < furniture_pose_size + 7:
            raise ValueError(
                "one_leg real annotation requires the appended obstacle-front pose"
            )
        front = poses[furniture_pose_size : furniture_pose_size + 7].copy()
        if not np.isfinite(front).all() or np.linalg.norm(front[3:]) < 1e-6:
            raise ValueError("Invalid obstacle-front pose")

        # Simulator side-obstacle offsets are expressed in the robot/sim frame.
        # Convert the offsets into April coordinates before feeding the inherited
        # furniture skill functions with sim_to_april=identity.
        offsets_robot = (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-0.075, -0.175, 0.0], dtype=np.float32),
            np.array([-0.075, 0.175, 0.0], dtype=np.float32),
        )
        rotation_robot_to_april = self.robot_to_april[:3, :3]
        obstacle_poses = {}
        for name, offset_robot in zip(_OBSTACLE_NAMES, offsets_robot):
            pose = front.copy()
            pose[:3] += rotation_robot_to_april @ offset_robot
            obstacle_poses[name] = pose
        return obstacle_poses

    def _attach_active_part(
        self,
        active_part,
        ee_pose_robot: np.ndarray,
        effective_poses: Mapping[str, np.ndarray],
    ) -> bool:
        if active_part is None:
            return False
        part_pose_robot = self.april_to_robot @ _pose_vector_to_matrix(
            effective_poses[active_part.name]
        )
        center_distance = float(
            np.linalg.norm(ee_pose_robot[:3, 3] - part_pose_robot[:3, 3])
        )
        guidance_distance = float("inf")
        if self.previous_guidance_point_robot is not None:
            guidance_distance = float(
                np.linalg.norm(
                    ee_pose_robot[:3, 3]
                    - np.asarray(self.previous_guidance_point_robot, dtype=np.float32)
                )
            )
        if min(center_distance, guidance_distance) > self.attachment_distance_m:
            return False

        if self._attached_part_name is not None:
            previous = self._tracked_parts[self._attached_part_name]
            previous.attached = False
            previous.ee_to_part_robot = None

        tracker = self._tracked_parts[active_part.name]
        tracker.attached = True
        tracker.ee_to_part_robot = (
            np.linalg.inv(ee_pose_robot) @ part_pose_robot
        ).astype(np.float32)
        tracker.pending_ee_to_part_robot = None
        tracker.pending_detection_count = 0
        self._attached_part_name = active_part.name
        return True

    def _detach_part(self):
        if self._attached_part_name is None:
            return
        tracker = self._tracked_parts[self._attached_part_name]
        tracker.attached = False
        tracker.ee_to_part_robot = None
        tracker.pending_ee_to_part_robot = None
        tracker.pending_detection_count = 0
        self._attached_part_name = None

    def _annotation_inputs(
        self,
        observation: Mapping[str, Any],
        ee_pose_robot: np.ndarray,
        effective_poses: Mapping[str, np.ndarray],
    ) -> Dict[str, Any]:
        ordered_names = [part.name for part in self.furniture.parts]
        ordered_poses = [effective_poses[name] for name in ordered_names]
        obstacles = self._obstacle_poses(observation)
        ordered_names.extend(_OBSTACLE_NAMES)
        ordered_poses.extend(obstacles[name] for name in _OBSTACLE_NAMES)

        rb_states = torch.as_tensor(
            np.stack(ordered_poses, axis=0), dtype=torch.float32
        )
        part_idxs = {name: [idx] for idx, name in enumerate(ordered_names)}
        robot_state = observation["robot_state"]
        ee_pos = torch.as_tensor(ee_pose_robot[:3, 3], dtype=torch.float32)
        ee_quat = torch.as_tensor(T.mat2quat(ee_pose_robot[:3, :3]), dtype=torch.float32)
        gripper_width = torch.as_tensor(
            float(robot_state["gripper_width"]), dtype=torch.float32
        )

        active_part = self._active_part()
        active_name = None if active_part is None else active_part.name
        if active_name is not None:
            active_pos = rb_states[part_idxs[active_name][0], :3]
        else:
            active_pos = torch.zeros(3, dtype=torch.float32)
        finger_offset = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float32)
        attached_and_closed = (
            active_name is not None
            and active_name == self._attached_part_name
            and self._gripper_closed
        )
        active_force = (
            torch.tensor([0.01, 0.01, 0.0], dtype=torch.float32)
            if attached_and_closed
            else torch.zeros(3, dtype=torch.float32)
        )
        contact_forces = {name: None for name in ordered_names}
        if active_name is not None:
            contact_forces[active_name] = active_force

        return {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "gripper_width": gripper_width,
            "rb_states": rb_states,
            "part_idxs": part_idxs,
            "sim_to_april_mat": torch.eye(4, dtype=torch.float32),
            "april_to_robot_mat": torch.as_tensor(
                self.april_to_robot, dtype=torch.float32
            ),
            "left_finger_pos": active_pos - finger_offset,
            "right_finger_pos": active_pos + finger_offset,
            "left_finger_force": active_force,
            "right_finger_force": active_force,
            "part_contact_forces": contact_forces,
        }

    @staticmethod
    def _push_guidance_at_tabletop_height(
        part,
        annotation_inputs: Mapping[str, Any],
        guidance_point_robot: Optional[torch.Tensor],
        guidance_pose_robot: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
        """Keep inherited push XY while placing the real GP on tabletop z.

        The inherited simulator target copies the EE z at the pick-to-push
        transition.  Real trajectories record a converted tool-tip pose about
        10 cm below the tracked tabletop center, so the real-only policy
        replaces just z and leaves the obstacle-corner XY unchanged.
        """

        if guidance_point_robot is None:
            return guidance_point_robot, guidance_pose_robot, None

        table_idx = annotation_inputs["part_idxs"][part.name][0]
        table_pose_april = annotation_inputs["rb_states"][table_idx]
        table_center_robot = annotation_inputs["april_to_robot_mat"] @ torch.cat(
            [
                table_pose_april[:3],
                torch.ones(
                    1,
                    dtype=table_pose_april.dtype,
                    device=table_pose_april.device,
                ),
            ]
        )
        tabletop_z = table_center_robot[2]

        aligned_point = guidance_point_robot.clone()
        aligned_point[2] = tabletop_z.to(
            dtype=aligned_point.dtype, device=aligned_point.device
        )
        aligned_pose = guidance_pose_robot
        if guidance_pose_robot is not None:
            aligned_pose = guidance_pose_robot.clone()
            aligned_pose[2, 3] = tabletop_z.to(
                dtype=aligned_pose.dtype, device=aligned_pose.device
            )

        # Keep getters coherent inside the real annotator without changing the
        # inherited simulator target or simulator implementation.
        part.skill_guidance_point = aligned_point.clone()
        if aligned_pose is not None:
            part.skill_guidance_pose_robot = aligned_pose.clone()
        return aligned_point, aligned_pose, float(tabletop_z.item())

    def _step_skill_state(self, annotation_inputs: Mapping[str, Any]) -> Dict[str, Any]:
        if self.assemble_idx >= len(self.furniture.should_be_assembled):
            return {
                "skill": self.previous_skill,
                "skill_state": self.previous_skill_state,
                "assembly_step": self.previous_assembly_step,
                "guidance_point": self.previous_guidance_point,
                "guidance_pose": self.previous_guidance_pose,
                "guidance_gripper_width": self.previous_guidance_gripper_width,
                "debug": {"phase": "complete"},
            }

        part1_idx, part2_idx = self.furniture.should_be_assembled[self.assemble_idx]
        part1 = self.furniture.parts[part1_idx]
        part2 = self.furniture.parts[part2_idx]
        assembly_step = self._assembly_step_label(part1, part2)
        skill_state = None
        skill = None
        guidance_point_robot = None
        guidance_pose_robot = None
        guidance_gripper_width = None
        debug = {
            "assemble_idx": self.assemble_idx,
            "active_part": None,
            "assembly_step": assembly_step,
            "phase": None,
        }

        part1_active = (
            not getattr(part1, "pre_assemble_done", True)
            and getattr(part1, "skill_state", None) != "done"
        )
        if part1_active:
            (
                skill_state,
                skill,
                guidance_point_robot,
                guidance_pose_robot,
                guidance_gripper_width,
            ) = self._update_part1_skill_state(part1, annotation_inputs)
            # A geometric target hit is not sufficient on the real robot: the
            # table is still being moved while the gripper remains closed.  Do
            # not hand control to the leg phase until the operator releases it.
            if (
                skill_state == "done"
                and self._gripper_closed
                and self._attached_part_name == part1.name
            ):
                part1.skill_state = "push"
                skill_state = "push"
                skill = "push"
                guidance_point_robot = part1.get_guidance_point()
                guidance_pose_robot = part1.get_guidance_pose()
                guidance_gripper_width = part1.get_guidance_gripper_width()
            if skill_state == "push":
                (
                    guidance_point_robot,
                    guidance_pose_robot,
                    push_tabletop_z,
                ) = self._push_guidance_at_tabletop_height(
                    part1,
                    annotation_inputs,
                    guidance_point_robot,
                    guidance_pose_robot,
                )
                debug["push_target_z_policy"] = "tracked_tabletop_center"
                debug["push_target_tabletop_z_robot"] = push_tabletop_z
            released_after_push = (
                skill_state == "push"
                and not self._gripper_closed
                and self._attached_part_name == part1.name
                and self._part_displacement_from_start(part1.name, annotation_inputs)
                >= self.table_release_displacement_m
            )
            if released_after_push:
                part1.skill_state = "done"
                skill_state = "done"
                skill = None
            if skill_state == "done":
                part1.pre_assemble_done = True
            debug["active_part"] = part1.name
            debug["phase"] = "pre_assemble"

        assembled = self._assembled(annotation_inputs, part1_idx, part2_idx)
        part1_complete = getattr(part1, "pre_assemble_done", True) or (
            getattr(part1, "skill_state", None) == "done"
        )
        if part1_complete:
            released_operated_part = (
                self._current_gripper_event == "opened"
                and self._attached_part_name == part2.name
                and getattr(part2, "skill_state", None) == "place"
            )
            if released_operated_part:
                # The real pickle has no contact force with which to certify a
                # seated part.  On a successful demonstration, releasing the
                # transported leg at the assembly site is the observable
                # place->insert boundary.  ``None`` keeps the inherited drop
                # guard conservative for this transition frame.
                part2.skill_state = "insert"
                annotation_inputs["part_contact_forces"][part2.name] = None
                self._placed_part_names.add(part2.name)
            elif (
                part2.name in self._placed_part_names
                and getattr(part2, "skill_state", None) in {"insert", "screw"}
            ):
                annotation_inputs["part_contact_forces"][part2.name] = None
            (
                skill_state,
                skill,
                guidance_point_robot,
                guidance_pose_robot,
                guidance_gripper_width,
            ) = self._update_operated_part(
                part2, annotation_inputs, part1.name, assembled
            )
            debug["active_part"] = part2.name
            debug["phase"] = "assemble"

        if part1_complete and skill == "place":
            place_target = self._tabletop_place_target_details(
                part2, part1.name, annotation_inputs
            )
            guidance_point_robot = place_target["guidance_robot"]
            if guidance_pose_robot is not None:
                guidance_pose_robot = guidance_pose_robot.clone()
                guidance_pose_robot[:3, 3] = guidance_point_robot
            debug["place_target_policy"] = PLACE_TARGET_POLICY_TABLETOP
            debug["place_target_socket_label"] = place_target["socket_label"]
            debug["place_target_socket_local"] = (
                place_target["socket_local"].detach().cpu().tolist()
            )
            debug["place_target_socket_robot"] = (
                place_target["socket_robot"].detach().cpu().tolist()
            )
            debug["place_target_socket_xy_scores"] = (
                place_target["socket_scores"].detach().cpu().tolist()
            )
            debug["place_target_longest_leg_length_m"] = place_target[
                "longest_leg_length_m"
            ]
            debug["place_target_half_leg_length_m"] = place_target[
                "half_leg_length_m"
            ]
            debug["place_target_leg_center_robot"] = (
                place_target["leg_center_robot"].detach().cpu().tolist()
            )
            debug["place_target_leg_to_ee_z_offset_m"] = place_target[
                "leg_to_ee_z_offset_m"
            ]
            debug["place_target_total_z_offset_m"] = place_target[
                "total_z_offset_m"
            ]
            debug["place_target_guidance_local"] = (
                place_target["guidance_local"].detach().cpu().tolist()
            )

        skill_state_label = self._skill_state_label(
            debug["active_part"],
            part1.name if debug["active_part"] == part2.name else part2.name,
            skill,
        )
        if skill_state == "done" and assembled:
            self.assemble_idx += 1
            self._reset_next_pair(self.assemble_idx)

        if skill is None or skill_state == "done":
            skill = self.previous_skill
            skill_state_label = self.previous_skill_state
            assembly_step = self.previous_assembly_step
            guidance_point_robot = self.previous_guidance_point_robot
            guidance_pose_robot = self.previous_guidance_pose_robot
            guidance_gripper_width = self.previous_guidance_gripper_width
        else:
            self.previous_skill = skill
            self.previous_skill_state = skill_state_label
            self.previous_assembly_step = assembly_step
            if guidance_point_robot is not None:
                self.previous_guidance_point_robot = _to_numpy(
                    guidance_point_robot
                ).astype(np.float32)
            if guidance_pose_robot is not None:
                self.previous_guidance_pose_robot = _pose_to_numpy(
                    guidance_pose_robot
                )
            self.previous_guidance_gripper_width = guidance_gripper_width

        guidance_point = (
            None
            if guidance_point_robot is None
            else _to_numpy(guidance_point_robot).astype(np.float32)
        )
        guidance_pose = (
            None if guidance_pose_robot is None else _pose_to_numpy(guidance_pose_robot)
        )
        self.previous_guidance_point = (
            None if guidance_point is None else guidance_point.copy()
        )
        self.previous_guidance_point_clean = (
            None if guidance_point is None else guidance_point.copy()
        )
        self.previous_guidance_pose = (
            None if guidance_pose is None else guidance_pose.copy()
        )
        self.previous_guidance_pose_clean = (
            None if guidance_pose is None else guidance_pose.copy()
        )
        return {
            "skill": skill,
            "skill_state": skill_state_label,
            "assembly_step": assembly_step,
            "guidance_point": guidance_point,
            "guidance_pose": guidance_pose,
            "guidance_gripper_width": guidance_gripper_width,
            "debug": debug,
        }

    def _camera_projections(
        self,
        observation: Mapping[str, Any],
        trajectory_camera_info: Mapping[str, Any],
        guidance_point_robot: Optional[np.ndarray],
        guidance_pose_robot: Optional[np.ndarray],
        guidance_gripper_width: Optional[float],
    ) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Optional[dict]]]:
        point_annotations: Dict[str, Optional[np.ndarray]] = {
            "color_image1": None,
            "color_image2": None,
        }
        grasp_annotations: Dict[str, Optional[dict]] = {
            "color_image1": None,
            "color_image2": None,
        }
        if guidance_point_robot is None:
            return point_annotations, grasp_annotations

        front_config = trajectory_camera_info.get("front")
        wrist_config = trajectory_camera_info.get("wrist")
        if not isinstance(front_config, Mapping) or not isinstance(wrist_config, Mapping):
            raise ValueError("Trajectory camera_info must contain front and wrist metadata")

        front_camera_to_april = np.asarray(
            observation.get("camera_to_april"), dtype=np.float32
        )
        if front_camera_to_april.shape != (4, 4):
            raise ValueError("Observation camera_to_april must have shape (4, 4)")
        front_point_to_camera = (
            np.linalg.inv(front_camera_to_april) @ self.robot_to_april
        ).astype(np.float32)

        wrist_pose = np.asarray(
            observation["robot_state"].get("wrist_pose"), dtype=np.float32
        )
        if wrist_pose.shape != (4, 4):
            raise ValueError("robot_state wrist_pose must have shape (4, 4)")
        wrist_point_to_camera = np.linalg.inv(wrist_pose).astype(np.float32)

        projection_specs = {
            "color_image2": (
                front_point_to_camera,
                _intrinsics_matrix(front_config),
                _camera_image_size(front_config),
            ),
            "color_image1": (
                wrist_point_to_camera,
                _intrinsics_matrix(wrist_config),
                _camera_image_size(wrist_config),
            ),
        }
        for image_key, (point_to_camera, intrinsics, image_size) in projection_specs.items():
            point_annotations[image_key] = _project_point(
                guidance_point_robot,
                point_to_camera=point_to_camera,
                intrinsics=intrinsics,
                image_size=image_size,
            )

        corners = _build_grasp_rect_points_3d(
            guidance_pose_robot,
            gripper_width=(
                None
                if guidance_gripper_width is None
                else float(_to_numpy(guidance_gripper_width).reshape(-1)[0])
            ),
        )
        if corners is not None:
            for image_key, (point_to_camera, intrinsics, image_size) in projection_specs.items():
                projected_corners = _project_polygon(
                    corners,
                    point_to_camera=point_to_camera,
                    intrinsics=intrinsics,
                    image_size=image_size,
                )
                center = point_annotations[image_key]
                if projected_corners is not None and center is not None:
                    grasp_annotations[image_key] = {
                        "style": "grasp_rect",
                        "center": center.copy(),
                        "corners": projected_corners,
                    }
        return point_annotations, grasp_annotations

    def annotate_observation(
        self,
        observation: Mapping[str, Any],
        trajectory_camera_info: Mapping[str, Any],
        *,
        frame_idx: int,
    ) -> Dict[str, Any]:
        self._frame_idx = frame_idx
        self.stats.frame_count += 1
        ee_pose = _ee_pose_robot(observation)
        width = float(observation["robot_state"]["gripper_width"])
        gripper_event = self._gripper_event(width)
        self._current_gripper_event = gripper_event
        active_part = self._active_part()

        effective_poses = {
            part.name: self._tracked_pose(observation, part, ee_pose)
            for part in self.furniture.parts
        }
        attached_on_this_frame = False
        if gripper_event == "closed":
            attached_on_this_frame = self._attach_active_part(
                active_part, ee_pose, effective_poses
            )

        annotation_inputs = self._annotation_inputs(
            observation, ee_pose, effective_poses
        )
        bundle = self._step_skill_state(annotation_inputs)
        if gripper_event == "opened":
            self._detach_part()

        guidance_point_2d, grasp_annotation_2d = self._camera_projections(
            observation,
            trajectory_camera_info,
            bundle["guidance_point"],
            bundle["guidance_pose"],
            bundle["guidance_gripper_width"],
        )
        if guidance_point_2d["color_image2"] is not None:
            self.stats.projected_front_count += 1
        if guidance_point_2d["color_image1"] is not None:
            self.stats.projected_wrist_count += 1
        if bundle["skill"] is not None:
            self.stats.skill_counts[bundle["skill"]] += 1

        debug = dict(bundle["debug"])
        debug.update(
            {
                "frame_idx": frame_idx,
                "gripper_event": gripper_event,
                "gripper_closed": self._gripper_closed,
                "attached_on_this_frame": attached_on_this_frame,
                "attached_part": self._attached_part_name,
                "part_pose_sources": {
                    name: tracker.source
                    for name, tracker in self._tracked_parts.items()
                },
            }
        )
        return {
            **bundle,
            "guidance_point_clean": (
                None
                if bundle["guidance_point"] is None
                else bundle["guidance_point"].copy()
            ),
            "guidance_pose_clean": (
                None
                if bundle["guidance_pose"] is None
                else bundle["guidance_pose"].copy()
            ),
            "guidance_point_2d": guidance_point_2d,
            "grasp_annotation_2d": grasp_annotation_2d,
            "debug": debug,
        }


def _observation_with_pose_estimate(
    observation: Mapping[str, Any],
    estimate: PartPoseEstimate,
    part_indices: Mapping[str, int],
) -> Dict[str, Any]:
    """Build an annotation-only view with one recovered pose inserted."""

    if estimate.part_name not in part_indices:
        raise ValueError(f"Unknown recovered part {estimate.part_name!r}")
    part_idx = part_indices[estimate.part_name]
    poses = np.asarray(observation.get("parts_poses"), dtype=np.float32).reshape(-1)
    required = (part_idx + 1) * 7
    if poses.size < required:
        raise ValueError(
            f"parts_poses has {poses.size} values; part {part_idx} requires {required}"
        )

    overlay = dict(observation)
    overlaid_poses = poses.copy()
    overlaid_poses[part_idx * 7 : required] = estimate.pose_april
    overlay["parts_poses"] = overlaid_poses

    founds = observation.get("parts_founds")
    if founds is not None:
        overlaid_founds = np.asarray(founds, dtype=bool).copy()
        overlaid_founds[part_idx] = True
        overlay["parts_founds"] = overlaid_founds
    validity = observation.get("parts_pose_valid")
    if validity is not None:
        overlaid_validity = np.asarray(validity, dtype=bool).copy()
        overlaid_validity[part_idx] = True
        overlay["parts_pose_valid"] = overlaid_validity
    overlay["_real_pose_override_sources"] = {
        estimate.part_name: estimate.source
    }
    return overlay


_ANNOTATION_FIELDS = (
    "skill",
    "skill_state",
    "assembly_step",
    "guidance_point",
    "guidance_point_clean",
    "guidance_pose",
    "guidance_pose_clean",
    "guidance_gripper_width",
    "guidance_point_2d",
    "grasp_annotation_2d",
)


def _apply_annotation_bundle(
    observation: MutableMapping[str, Any], bundle: Mapping[str, Any]
) -> None:
    for key in _ANNOTATION_FIELDS:
        observation[key] = bundle[key]
    observation["guidance"] = {
        "source": ANNOTATION_SOURCE,
        "target_point": bundle["guidance_point"],
        "target_point_frame": "robot_base",
        "target_point_2d": bundle["guidance_point_2d"],
        "skill": bundle["skill"],
        "skill_state": bundle["skill_state"],
    }
    observation["real_annotation_debug"] = bundle["debug"]


def _real_annotation_metadata(
    stats: RealAnnotationStats,
    *,
    pose_provider: Optional[PartPoseProvider],
    mode: str,
) -> Dict[str, Any]:
    metadata = {
        "source": ANNOTATION_SOURCE,
        "version": ANNOTATION_VERSION,
        "mode": mode,
        "complete": True,
        "target_point_frame": "robot_base",
        "front_projection_transform": "camera_to_april",
        "wrist_projection_transform": "robot_state.wrist_pose",
        "missing_pose_policy": "parts_founds + ee_rigid_propagation + held_last",
        "pose_tracking_policy": DEFAULT_POSE_TRACKING_POLICY,
        "release_pose_policy": "held_last",
        "sam2_override_enabled": pose_provider is not None,
        "place_target_policy": PLACE_TARGET_POLICY_TABLETOP,
        "place_target_formula": {
            "axis": "robot_world_positive_z",
            "socket_to_leg_center_leg_fraction": 0.5,
            "leg_center_to_ee_leg_fraction": LEG_TO_EE_LENGTH_FRACTION,
            "lateral_offset_m": [0.0, 0.0],
        },
        "push_target_formula": {
            "xy": "inherited_obstacle_corner_trailing_edge_center",
            "z": "tracked_tabletop_center",
        },
        "stats": stats.as_dict(),
    }
    if pose_provider is not None:
        metadata["pose_provider"] = pose_provider.metadata()
    return metadata


class RealSkillAnnotationSession:
    """Stateful per-frame API shared by live and offline annotation."""

    def __init__(
        self,
        furniture_name: str,
        camera_info: Mapping[str, Any],
        *,
        pose_provider: Optional[PartPoseProvider] = None,
        mode: str = "online",
    ):
        if not isinstance(camera_info, Mapping):
            raise ValueError("Real annotation requires trajectory camera_info")
        if mode not in {"online", "offline"}:
            raise ValueError(f"Unsupported real annotation mode {mode!r}")
        self.camera_info = camera_info
        self.pose_provider = pose_provider
        self.mode = mode
        self.annotator = RealSkillAnnotator(str(furniture_name))
        self.part_indices = {
            part.name: part.part_idx for part in self.annotator.furniture.parts
        }
        self.frame_idx = 0

    @property
    def stats(self) -> RealAnnotationStats:
        return self.annotator.stats

    def reset(self) -> None:
        self.annotator.reset()
        self.frame_idx = 0

    def annotate_observation(
        self, observation: MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        if not isinstance(observation, MutableMapping):
            raise ValueError("Real annotation observation must be mutable")
        estimate = (
            None
            if self.pose_provider is None
            else self.pose_provider.estimate(self.frame_idx, observation)
        )
        annotation_observation = (
            observation
            if estimate is None
            else _observation_with_pose_estimate(
                observation, estimate, self.part_indices
            )
        )
        bundle = self.annotator.annotate_observation(
            annotation_observation,
            self.camera_info,
            frame_idx=self.frame_idx,
        )
        if estimate is not None:
            bundle["debug"]["pose_override"] = {
                "part_name": estimate.part_name,
                "source": estimate.source,
                "confidence": estimate.confidence,
                **dict(estimate.details),
            }
        _apply_annotation_bundle(observation, bundle)
        self.frame_idx += 1
        return bundle

    def update_trajectory_metadata(self, data: MutableMapping[str, Any]) -> None:
        metadata = data.setdefault("metadata", {})
        if not isinstance(metadata, MutableMapping):
            raise ValueError("Trajectory metadata must be a mapping when present")
        metadata["real_skill_annotation"] = _real_annotation_metadata(
            self.stats,
            pose_provider=self.pose_provider,
            mode=self.mode,
        )
        data["annotation_source"] = ANNOTATION_SOURCE


def annotate_trajectory(
    data: Dict[str, Any],
    *,
    pose_provider: Optional[PartPoseProvider] = None,
) -> RealAnnotationStats:
    observations = data.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Trajectory must contain a non-empty observations list")
    furniture_name = data.get("furniture", data.get("task"))
    if furniture_name not in SUPPORTED_FURNITURE:
        raise ValueError(
            f"Real annotation currently supports {sorted(SUPPORTED_FURNITURE)}, "
            f"got {furniture_name!r}"
        )
    camera_info = data.get("camera_info")
    if not isinstance(camera_info, Mapping):
        raise ValueError("Trajectory is missing camera_info")

    session = RealSkillAnnotationSession(
        str(furniture_name),
        camera_info,
        pose_provider=pose_provider,
        mode="offline",
    )
    for observation in observations:
        session.annotate_observation(observation)
    session.update_trajectory_metadata(data)
    return session.stats


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy-2 pickles in the repository's older NumPy environment."""

    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _open_pickle(path: Path, mode: str) -> BinaryIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    if path.suffix == ".xz":
        return lzma.open(path, mode)
    return path.open(mode)


def load_trajectory_pickle(path: Path) -> Dict[str, Any]:
    with _open_pickle(Path(path), "rb") as file:
        data = _NumpyCompatUnpickler(file).load()
    if not isinstance(data, dict):
        raise ValueError(f"Expected trajectory mapping in {path}, got {type(data).__name__}")
    return data


def _annotated_output_path(input_path: Path) -> Path:
    for suffix in (".pkl.xz", ".pkl.gz", ".pkl"):
        if input_path.name.endswith(suffix):
            base_name = input_path.name[: -len(suffix)]
            return input_path.with_name(f"{base_name}.annotated{suffix}")
    return input_path.with_name(f"{input_path.name}.annotated.pkl")


def _atomic_pickle_dump(data: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_suffix = (
        f".tmp{output_path.suffix}"
        if output_path.suffix in {".gz", ".xz"}
        else ".tmp"
    )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{output_path.name}.",
        suffix=temporary_suffix,
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with _open_pickle(temporary_path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def annotate_pickle(
    input_path: Path,
    *,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    pose_provider: Optional[PartPoseProvider] = None,
) -> Tuple[Path, RealAnnotationStats]:
    """Annotate one trajectory and return ``(written_path, stats)``.

    With neither ``output_path`` nor ``overwrite``, the result is written next
    to the input as ``<stem>.annotated.pkl``.  ``overwrite=True`` and no output
    path atomically replaces the input pickle.
    """

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path is not None and overwrite:
        raise ValueError("Choose either output_path or overwrite=True, not both")
    if overwrite:
        destination = input_path
    elif output_path is not None:
        destination = Path(output_path).expanduser().resolve()
    else:
        destination = _annotated_output_path(input_path)

    if destination == input_path and not overwrite:
        raise ValueError(
            "Refusing to replace the input through output_path; use overwrite=True explicitly"
        )
    if destination.exists() and destination != input_path:
        raise FileExistsError(
            f"Refusing to replace existing output {destination}; choose another path"
        )
    data = load_trajectory_pickle(input_path)
    stats = annotate_trajectory(
        data,
        pose_provider=pose_provider,
    )
    _atomic_pickle_dump(data, destination)
    return destination, stats


def _discover_inputs(paths: Sequence[Path]) -> Iterable[Path]:
    seen = set()
    for path in paths:
        path = path.expanduser().resolve()
        candidates = [path] if path.is_file() else sorted(path.glob("*.pkl*"))
        for candidate in candidates:
            if ".annotated." in candidate.name or candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add real-robot skill and target-point annotations to pickle trajectories."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input pickle file(s), or directories containing pickle files.",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output",
        type=Path,
        help="Output path (valid only for a single input file).",
    )
    parser.add_argument(
        "--sam2-tabletop-recovery",
        "--tabletop-recovery",
        dest="sam2_tabletop_recovery",
        type=Path,
        help=(
            "Optional backup SAM2 recovery.json from "
            "recover_tabletop_pose_sam2.py. When omitted, annotation uses "
            "the default AprilTag -> EE rigid -> held-last pose policy. "
            "Valid only for a single input."
        ),
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help=(
            "Render the annotated output with the standard simulator "
            "RGB + guidance-point + skill video settings; valid only for a "
            "single input"
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
        help="FPS for --video-output (simulator rollout default: 20)",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace each input pickle instead of writing *.annotated.pkl.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    inputs = list(_discover_inputs(args.inputs))
    if not inputs:
        raise SystemExit("No input pickle files found")
    if args.output is not None and len(inputs) != 1:
        raise SystemExit("--output requires exactly one input pickle")
    if args.sam2_tabletop_recovery is not None and len(inputs) != 1:
        raise SystemExit(
            "--sam2-tabletop-recovery requires exactly one input pickle"
        )
    if args.video_output is not None and len(inputs) != 1:
        raise SystemExit("--video-output requires exactly one input pickle")

    pose_provider = None
    if args.sam2_tabletop_recovery is not None:
        from src.eval.real_pose_provider import RecoveredTabletopPoseProvider

        pose_provider = RecoveredTabletopPoseProvider.from_recovery_json(
            args.sam2_tabletop_recovery
        )

    for input_path in inputs:
        destination, stats = annotate_pickle(
            input_path,
            output_path=args.output,
            overwrite=args.overwrite,
            pose_provider=pose_provider,
        )
        print(
            f"Annotated {input_path} -> {destination} "
            f"frames={stats.frame_count} skills={dict(stats.skill_counts)} "
            f"front_points={stats.projected_front_count} "
            f"pose_mode={'sam2_override' if pose_provider else 'ee_rigid'}"
        )
        if args.video_output is not None:
            from src.eval.render_skill_annotation_video import (
                render_skill_annotation_video,
            )

            render_skill_annotation_video(
                destination,
                args.video_output.expanduser().resolve(),
                fps=args.video_fps,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
