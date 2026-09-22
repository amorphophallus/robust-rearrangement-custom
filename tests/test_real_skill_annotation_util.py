import pickle
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.eval.real_skill_annotation_util import (
    ANNOTATION_SOURCE,
    ANNOTATION_STATUS_ANNOTATED,
    ANNOTATION_STATUS_KEY,
    DEFAULT_POSE_TRACKING_POLICY,
    LEG_TO_EE_LENGTH_FRACTION,
    ONE_LEG_REAL_PLACE_ORIENTATION_THRESHOLD_RAD,
    REAL_INSERT_TO_SCREW_TIMEOUT_S,
    RealSkillAnnotator,
    RealSkillAnnotationSession,
    _parse_args,
    annotate_pickle,
    load_trajectory_pickle,
    PLACE_TARGET_POLICY_TABLETOP,
    _pose_vector_to_matrix,
    _matrix_to_pose_vector,
)
from src.eval.skill_annotation_util import SkillAnnotator
from src.eval.real_pose_provider import RecoveredTabletopPoseProvider


CAMERA_TO_APRIL = np.array(
    [
        [0.99918, -0.00549, -0.04002, 0.03103],
        [-0.03190, 0.50078, -0.86499, 0.67005],
        [0.02479, 0.86556, 0.50020, -0.30727],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

WRIST_POSE = np.array(
    [
        [0.99964, 0.01913, 0.01874, 0.45575],
        [0.01922, -0.99981, -0.00423, 0.03263],
        [0.01866, 0.00459, -0.99982, 0.16062],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

PART_POSES = np.array(
    [
        0.014855, 0.204731, -0.022244, 0.010260, -0.710459, 0.703589, 0.010294,
        -0.200000, 0.070000, -0.015000, 0.0, -0.707107, 0.0, 0.707107,
        -0.120000, 0.070000, -0.015000, 0.0, -0.707107, 0.0, 0.707107,
        0.120000, 0.070000, -0.015000, 0.0, -0.707107, 0.0, 0.707107,
        0.189208, 0.044931, -0.027879, 0.045658, -0.696710, 0.050818, 0.714092,
        0.006900, 0.362900, -0.015000, -1.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)


def _camera_info():
    return {
        "front": {
            "record_intrinsics": {
                "fx": 302.4068,
                "fy": 302.2362,
                "ppx": 161.5239,
                "ppy": 125.3197,
                "width": 320,
                "height": 240,
            }
        },
        "wrist": {
            "record_intrinsics": {
                "fx": 305.8882,
                "fy": 305.5067,
                "ppx": 160.5316,
                "ppy": 119.4659,
                "width": 320,
                "height": 240,
            }
        },
    }


def _observation(*, gripper_width=0.08, table_found=True, ee_x=0.4577):
    ee_pose = np.eye(4, dtype=np.float32)
    ee_pose[:3, 3] = [ee_x, 0.0322, 0.0572]
    return {
        "color_image1": np.zeros((8, 8, 3), dtype=np.uint8),
        "color_image2": np.zeros((8, 8, 3), dtype=np.uint8),
        "parts_poses": PART_POSES.copy(),
        "parts_founds": np.array(
            [table_found, False, False, False, True, False], dtype=bool
        ),
        "parts_pose_valid": np.ones(6, dtype=bool),
        "camera_to_april": CAMERA_TO_APRIL.copy(),
        "robot_state": {
            "ee_pose": ee_pose,
            "ee_pos": ee_pose[:3, 3].copy(),
            "ee_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "wrist_pose": WRIST_POSE.copy(),
            "gripper_width": gripper_width,
        },
        "skill": None,
        "guidance": None,
    }


def _trajectory():
    return {
        "observations": [
            _observation(),
            _observation(gripper_width=0.01, table_found=False),
            _observation(gripper_width=0.01, table_found=False, ee_x=0.4777),
        ],
        "actions": [[0.0] * 8, [0.0] * 8],
        "rewards": [0.0, 0.0],
        "camera_info": _camera_info(),
        "task": "one_leg",
        "furniture": "one_leg",
        "success": True,
        "metadata": {"schema": "deoxys_furniturebench_raw_v2"},
    }


def _multi_task_observation(annotator, *, width=0.08, ee_pose=None):
    poses = []
    for part in annotator.furniture.parts:
        part_pose = np.eye(4, dtype=np.float32)
        part_pose[:3, :3] = np.asarray(part.reset_ori[0], dtype=np.float32)[:3, :3]
        part_pose[:3, 3] = np.asarray(part.reset_pos[0], dtype=np.float32)[:3]
        poses.append(_matrix_to_pose_vector(part_pose))
    if annotator.furniture_name == "one_leg":
        poses.append(PART_POSES[-7:].copy())
    if ee_pose is None:
        ee_pose = annotator.april_to_robot @ _pose_vector_to_matrix(poses[0])
    return {
        "parts_poses": np.concatenate(poses),
        "parts_founds": np.ones(len(annotator.furniture.parts), dtype=bool),
        "parts_pose_valid": np.ones(len(annotator.furniture.parts), dtype=bool),
        "camera_to_april": CAMERA_TO_APRIL.copy(),
        "robot_state": {
            "ee_pose": np.asarray(ee_pose, dtype=np.float32).copy(),
            "wrist_pose": WRIST_POSE.copy(),
            "gripper_width": width,
        },
        "color_image1": np.zeros((8, 8, 3), dtype=np.uint8),
        "color_image2": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def _begin_operated_part(annotator, pair_idx):
    annotator.assemble_idx = pair_idx
    part1_idx, part2_idx = annotator.furniture.should_be_assembled[pair_idx]
    part1 = annotator.furniture.parts[part1_idx]
    part2 = annotator.furniture.parts[part2_idx]
    if hasattr(part1, "pre_assemble_done"):
        part1.pre_assemble_done = True
    initial = _multi_task_observation(annotator)
    part_pose = _pose_vector_to_matrix(
        initial["parts_poses"][part2_idx * 7 : (part2_idx + 1) * 7]
    )
    initial["robot_state"]["ee_pose"] = annotator.april_to_robot @ part_pose
    annotator.annotate_observation(initial, _camera_info(), frame_idx=0)
    grasp = copy.deepcopy(initial)
    grasp["robot_state"]["gripper_width"] = 0.01
    annotator.annotate_observation(grasp, _camera_info(), frame_idx=1)
    placed = annotator.annotate_observation(grasp, _camera_info(), frame_idx=2)
    return grasp, part1, part2, placed


def _move_attached_part_to_target(annotator, observation, part):
    moved = copy.deepcopy(observation)
    tracker = annotator._tracked_parts[part.name]
    target_ee = (
        part.skill_target_part_pose_robot.detach().cpu().numpy()
        @ np.linalg.inv(tracker.ee_to_part_robot)
    )
    moved["robot_state"]["ee_pose"] = target_ee.astype(np.float32)
    moved["parts_founds"][part.part_idx] = False
    return moved


def _detect_assembled_part(annotator, observation, part1, part2):
    detected = copy.deepcopy(observation)
    part1_pose = _pose_vector_to_matrix(
        detected["parts_poses"][part1.part_idx * 7 : (part1.part_idx + 1) * 7]
    )
    assembled = np.asarray(
        annotator.furniture.assembled_rel_poses[(part1.part_idx, part2.part_idx)][0],
        dtype=np.float32,
    )
    detected["parts_poses"][part2.part_idx * 7 : (part2.part_idx + 1) * 7] = (
        _matrix_to_pose_vector(part1_pose @ assembled)
    )
    detected["parts_founds"][part2.part_idx] = True
    return detected


class RealSkillAnnotationUtilTest(unittest.TestCase):
    def test_multi_task_push_advances_on_geometry_without_release(self):
        for task, expected_next in (
            ("round_table", "leg-top-pick"),
            ("lamp", "bulb-base-pick"),
        ):
            with self.subTest(task=task):
                annotator = RealSkillAnnotator(task)
                initial = _multi_task_observation(annotator)
                first = annotator.annotate_observation(
                    initial, _camera_info(), frame_idx=0
                )
                self.assertEqual(first["skill"], "push")
                grasp = copy.deepcopy(initial)
                grasp["robot_state"]["gripper_width"] = 0.01
                attached = annotator.annotate_observation(
                    grasp, _camera_info(), frame_idx=1
                )
                self.assertTrue(attached["debug"]["attached_on_this_frame"])
                part = annotator.furniture.parts[0]
                part_center_robot = (
                    annotator.april_to_robot
                    @ _pose_vector_to_matrix(initial["parts_poses"][:7])
                )[:3, 3]
                moved = copy.deepcopy(grasp)
                moved["robot_state"]["ee_pose"][:3, 3] += (
                    attached["guidance_point"] - part_center_robot
                )
                moved["parts_founds"][part.part_idx] = False
                advanced = annotator.annotate_observation(
                    moved, _camera_info(), frame_idx=2
                )
                self.assertEqual(advanced["skill_state"], expected_next)
                self.assertTrue(advanced["debug"]["gripper_closed"])
                self.assertEqual(
                    advanced["debug"]["part_pose_sources"][part.name],
                    "ee_propagated",
                )

    def test_multi_task_held_last_cannot_finish_push(self):
        for task in ("round_table", "lamp"):
            with self.subTest(task=task):
                annotator = RealSkillAnnotator(task)
                observation = _multi_task_observation(annotator)
                first = annotator.annotate_observation(
                    observation, _camera_info(), frame_idx=0
                )
                part = annotator.furniture.parts[0]
                target = (
                    annotator.april_to_robot
                    @ _pose_vector_to_matrix(observation["parts_poses"][:7])
                )
                target[:3, 3] = first["guidance_point"]
                annotator._tracked_parts[part.name].pose_april = _matrix_to_pose_vector(
                    annotator.robot_to_april @ target
                )
                stale = copy.deepcopy(observation)
                stale["parts_founds"][part.part_idx] = False
                bundle = annotator.annotate_observation(
                    stale, _camera_info(), frame_idx=1
                )
                self.assertEqual(bundle["skill"], "push")
                self.assertTrue(bundle["debug"]["blocked_stale_push_transition"])

    def test_round_table_and_lamp_insert_screw_and_pair_completion(self):
        for task, pair_idx, expected_next_idx in (
            ("one_leg", 0, 1),
            ("round_table", 0, 1),
            ("round_table", 1, 2),
            ("lamp", 0, 1),
        ):
            with self.subTest(task=task, pair_idx=pair_idx):
                annotator = RealSkillAnnotator(task)
                grasp, part1, part2, placed = _begin_operated_part(
                    annotator, pair_idx
                )
                self.assertEqual(placed["skill"], "place")
                self.assertTrue(annotator._tracked_parts[part2.name].attached)
                moved = _move_attached_part_to_target(annotator, grasp, part2)
                moved["step_timestamp_ns"] = 1_000_000_000
                inserted = annotator.annotate_observation(
                    moved, _camera_info(), frame_idx=3
                )
                self.assertEqual(inserted["skill"], "insert")
                self.assertEqual(
                    inserted["debug"]["part_pose_sources"][part2.name],
                    "ee_propagated",
                )
                opened = copy.deepcopy(moved)
                opened["robot_state"]["gripper_width"] = 0.08
                opened["step_timestamp_ns"] = 2_000_000_000
                still_insert = annotator.annotate_observation(
                    opened, _camera_info(), frame_idx=4
                )
                self.assertEqual(still_insert["skill"], "insert")
                self.assertFalse(
                    still_insert["debug"]["insert_to_screw_timeout_transition"]
                )
                timed_out = copy.deepcopy(opened)
                timed_out["robot_state"]["gripper_width"] = 0.01
                timed_out["step_timestamp_ns"] = int(
                    (1.0 + REAL_INSERT_TO_SCREW_TIMEOUT_S) * 1e9
                )
                screw = annotator.annotate_observation(
                    timed_out, _camera_info(), frame_idx=5
                )
                self.assertEqual(screw["skill"], "screw")
                self.assertTrue(
                    screw["debug"]["insert_to_screw_timeout_transition"]
                )
                self.assertEqual(annotator.assemble_idx, pair_idx)
                detected = _detect_assembled_part(
                    annotator, timed_out, part1, part2
                )
                completed = annotator.annotate_observation(
                    detected, _camera_info(), frame_idx=6
                )
                if task == "round_table" and pair_idx == 0:
                    self.assertEqual(annotator.assemble_idx, pair_idx)
                    self.assertEqual(completed["skill"], "screw")
                    self.assertTrue(
                        completed["debug"]["screw_assembled_latched"]
                    )
                    self.assertFalse(
                        completed["debug"]["screw_release_ready"]
                    )
                    released = copy.deepcopy(detected)
                    released["robot_state"]["gripper_width"] = 0.08
                    completed = annotator.annotate_observation(
                        released, _camera_info(), frame_idx=7
                    )
                    self.assertTrue(
                        completed["debug"]["screw_release_ready"]
                    )
                self.assertEqual(annotator.assemble_idx, expected_next_idx)

    def test_round_table_aligned_part_can_go_directly_from_pick_to_insert(self):
        for pair_idx in (0, 1):
            with self.subTest(pair_idx=pair_idx):
                annotator = RealSkillAnnotator("round_table")
                annotator.assemble_idx = pair_idx
                part1_idx, part2_idx = annotator.furniture.should_be_assembled[pair_idx]
                part1 = annotator.furniture.parts[part1_idx]
                part2 = annotator.furniture.parts[part2_idx]
                if hasattr(part1, "pre_assemble_done"):
                    part1.pre_assemble_done = True
                observation = _multi_task_observation(annotator)
                observation = _detect_assembled_part(
                    annotator, observation, part1, part2
                )
                part_pose = _pose_vector_to_matrix(
                    observation["parts_poses"][part2_idx * 7 : (part2_idx + 1) * 7]
                )
                observation["robot_state"]["ee_pose"] = (
                    annotator.april_to_robot @ part_pose
                )
                bundle = annotator.annotate_observation(
                    observation, _camera_info(), frame_idx=0
                )
                self.assertEqual(bundle["skill"], "insert")
                self.assertEqual(part2.skill_state, "insert")

    def test_one_leg_uses_selected_guidance_socket_for_fsm_target(self):
        annotator = RealSkillAnnotator("one_leg")
        table_idx, leg_idx = annotator.furniture.should_be_assembled[0]
        table = annotator.furniture.parts[table_idx]
        leg = annotator.furniture.parts[leg_idx]
        table.pre_assemble_done = True
        initial = _observation()
        annotator.annotate_observation(initial, _camera_info(), frame_idx=0)
        grasp = _observation(gripper_width=0.01)
        annotator.annotate_observation(grasp, _camera_info(), frame_idx=1)
        placed = annotator.annotate_observation(
            grasp, _camera_info(), frame_idx=2
        )

        selected = np.asarray(placed["debug"]["place_target_socket_local"])
        self.assertEqual(
            placed["debug"]["fsm_target_socket_label"],
            placed["debug"]["place_target_socket_label"],
        )
        self.assertAlmostEqual(float(leg.default_assembled_pose[0, 3]), selected[0])
        self.assertAlmostEqual(float(leg.default_assembled_pose[2, 3]), selected[2])

    def test_one_leg_real_orientation_threshold_does_not_change_sim(self):
        real = RealSkillAnnotator("one_leg")
        _, real_leg_idx = real.furniture.should_be_assembled[0]
        real_leg = real.furniture.parts[real_leg_idx]
        sim = SkillAnnotator("one_leg")
        _, sim_leg_idx = sim.furniture.should_be_assembled[0]
        sim_leg = sim.furniture.parts[sim_leg_idx]

        self.assertAlmostEqual(
            real_leg.skill_place_part_ori_threshold,
            ONE_LEG_REAL_PLACE_ORIENTATION_THRESHOLD_RAD,
        )
        self.assertAlmostEqual(sim_leg.skill_place_part_ori_threshold, 0.15)
        real.reset()
        _, reset_leg_idx = real.furniture.should_be_assembled[0]
        self.assertAlmostEqual(
            real.furniture.parts[reset_leg_idx].skill_place_part_ori_threshold,
            ONE_LEG_REAL_PLACE_ORIENTATION_THRESHOLD_RAD,
        )

    def test_one_leg_aligned_part_can_go_directly_from_pick_to_insert(self):
        annotator = RealSkillAnnotator("one_leg")
        table_idx, leg_idx = annotator.furniture.should_be_assembled[0]
        table = annotator.furniture.parts[table_idx]
        leg = annotator.furniture.parts[leg_idx]
        table.pre_assemble_done = True
        observation = _observation()
        annotator.annotate_observation(observation, _camera_info(), frame_idx=0)

        table_pose_april = _pose_vector_to_matrix(
            observation["parts_poses"][table_idx * 7 : (table_idx + 1) * 7]
        )
        target_leg_pose_april = table_pose_april @ np.asarray(
            leg.default_assembled_pose, dtype=np.float32
        )
        aligned = copy.deepcopy(observation)
        aligned["parts_poses"][leg_idx * 7 : (leg_idx + 1) * 7] = (
            _matrix_to_pose_vector(target_leg_pose_april)
        )
        aligned["parts_founds"][leg_idx] = True
        aligned["robot_state"]["ee_pose"] = (
            annotator.april_to_robot @ target_leg_pose_april
        )
        aligned["robot_state"]["gripper_width"] = 0.01
        annotator.annotate_observation(aligned, _camera_info(), frame_idx=1)
        bundle = annotator.annotate_observation(
            aligned, _camera_info(), frame_idx=2
        )

        self.assertEqual(bundle["skill"], "insert")
        self.assertEqual(leg.skill_state, "insert")
        self.assertTrue(bundle["debug"]["pick_to_insert_direct"])

    def test_round_table_leg_real_place_uses_top_local_xy_and_z(self):
        annotator = RealSkillAnnotator("round_table")
        grasp, top, leg, placed = _begin_operated_part(annotator, 0)
        self.assertEqual(placed["skill"], "place")
        top_pose_robot = annotator.april_to_robot @ _pose_vector_to_matrix(
            annotator._tracked_parts[top.name].pose_april
        )
        expected_leg_pose_robot = top_pose_robot @ np.asarray(
            leg.default_assembled_pose, dtype=np.float32
        )
        leg_length_m = max(
            float(value)
            for value in (
                getattr(leg, "reset_x_len", None),
                getattr(leg, "reset_y_len", None),
                getattr(leg, "reset_z_len", None),
            )
            if value is not None
        )
        expected_guidance = expected_leg_pose_robot[:3, 3].copy()
        expected_guidance[2] += 0.25 * leg_length_m
        np.testing.assert_allclose(
            placed["guidance_point"], expected_guidance, atol=1e-6
        )
        np.testing.assert_allclose(
            placed["guidance_pose"][:3, 3], expected_guidance, atol=1e-6
        )
        self.assertEqual(
            placed["debug"]["place_target_policy"],
            "target_leg_pose_plus_world_z_quarter_leg_length",
        )
        self.assertAlmostEqual(
            placed["debug"]["place_target_z_offset_m"],
            0.25 * leg_length_m,
            places=6,
        )

        moved = _move_attached_part_to_target(annotator, grasp, leg)
        local_offset = np.array([0.002, 0.002, 0.010], dtype=np.float32)
        moved["robot_state"]["ee_pose"][:3, 3] += (
            top_pose_robot[:3, :3] @ local_offset
        )

        inserted = annotator.annotate_observation(
            moved, _camera_info(), frame_idx=3
        )
        self.assertEqual(inserted["skill"], "insert")
        self.assertEqual(leg.skill_state, "insert")
        self.assertEqual(
            inserted["debug"]["real_place_rule"],
            "top_local_xy_z_ignore_z_rotation",
        )
        self.assertEqual(inserted["debug"]["real_place_planar_axes"], "xy")
        self.assertEqual(inserted["debug"]["real_place_axial_axis"], "z")
        self.assertEqual(
            inserted["debug"]["real_place_ignored_rotation_axis"], "z"
        )
        self.assertAlmostEqual(
            inserted["debug"]["real_place_xy_error_m"], 0.004, places=5
        )
        self.assertAlmostEqual(
            inserted["debug"]["real_place_axial_error_m"], 0.010, places=5
        )
        np.testing.assert_allclose(
            inserted["guidance_point"], placed["guidance_point"], atol=1e-6
        )
        np.testing.assert_allclose(
            inserted["guidance_pose"][:3, 3],
            placed["guidance_pose"][:3, 3],
            atol=1e-6,
        )
        self.assertEqual(
            inserted["debug"]["leg_guidance_applies_to_skill"], "insert"
        )

        target_rotation = torch.as_tensor(
            leg.default_assembled_pose[:3, :3], dtype=torch.float32
        )
        tabletop_z_rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        ignored_yaw_error = annotator._rotation_error_ignoring_parent_axis(
            tabletop_z_rotation @ target_rotation,
            target_rotation,
            ignored_axis=2,
        )
        self.assertAlmostEqual(float(ignored_yaw_error), 0.0, places=6)

    def test_round_table_base_pick_requires_ee_near_base(self):
        annotator = RealSkillAnnotator("round_table")
        annotator.assemble_idx = 1
        leg_idx, base_idx = annotator.furniture.should_be_assembled[1]
        leg = annotator.furniture.parts[leg_idx]
        base = annotator.furniture.parts[base_idx]
        leg.pre_assemble_done = True

        observation = _multi_task_observation(annotator)
        base_pose_april = _pose_vector_to_matrix(
            observation["parts_poses"][base_idx * 7 : (base_idx + 1) * 7]
        )
        base_pose_robot = annotator.april_to_robot @ base_pose_april
        far_ee_pose = base_pose_robot.copy()
        far_ee_pose[0, 3] += 0.15
        observation["robot_state"]["ee_pose"] = far_ee_pose
        annotator.annotate_observation(
            observation, _camera_info(), frame_idx=0
        )

        closed_far = copy.deepcopy(observation)
        closed_far["robot_state"]["gripper_width"] = 0.01
        annotator.annotate_observation(
            closed_far, _camera_info(), frame_idx=1
        )
        blocked = annotator.annotate_observation(
            closed_far, _camera_info(), frame_idx=2
        )
        self.assertEqual(blocked["skill"], "pick")
        self.assertEqual(base.skill_state, "pick")
        self.assertFalse(blocked["debug"]["base_pick_ee_distance_ok"])
        self.assertTrue(
            blocked["debug"]["base_pick_transition_blocked_by_ee_distance"]
        )

        opened_near = copy.deepcopy(observation)
        opened_near["robot_state"]["ee_pose"] = base_pose_robot
        annotator.annotate_observation(
            opened_near, _camera_info(), frame_idx=3
        )
        closed_near = copy.deepcopy(opened_near)
        closed_near["robot_state"]["gripper_width"] = 0.01
        annotator.annotate_observation(
            closed_near, _camera_info(), frame_idx=4
        )
        allowed = annotator.annotate_observation(
            closed_near, _camera_info(), frame_idx=5
        )
        self.assertTrue(allowed["debug"]["base_pick_ee_distance_ok"])
        self.assertEqual(allowed["skill"], "place")
        self.assertEqual(base.skill_state, "place")

        leg_pose_april = _pose_vector_to_matrix(
            closed_near["parts_poses"][leg_idx * 7 : (leg_idx + 1) * 7]
        )
        target_base_pose_robot = (
            annotator.april_to_robot
            @ leg_pose_april
            @ np.asarray(base.default_assembled_pose, dtype=np.float32)
        )
        expected_target_point = target_base_pose_robot[:3, 3]
        np.testing.assert_allclose(
            allowed["guidance_point"], expected_target_point, atol=1e-6
        )
        self.assertEqual(
            allowed["debug"]["base_target_policy"],
            "assembled_base_pose_center_from_leg",
        )

        base.skill_state = "insert"
        inserted = annotator.annotate_observation(
            closed_near, _camera_info(), frame_idx=6
        )
        self.assertEqual(inserted["skill"], "insert")
        np.testing.assert_allclose(
            inserted["guidance_point"], expected_target_point, atol=1e-6
        )

        base.skill_state = "screw"
        screw = annotator.annotate_observation(
            closed_near, _camera_info(), frame_idx=7
        )
        self.assertEqual(screw["skill"], "screw")
        np.testing.assert_allclose(
            screw["guidance_point"], expected_target_point, atol=1e-6
        )
        self.assertEqual(
            screw["debug"]["base_target_policy"],
            "assembled_base_pose_center_from_leg",
        )

    def test_unseated_release_returns_from_place_to_pick(self):
        for task, pair_idx, confirmation_frames in (
            ("round_table", 0, 1),
            ("round_table", 1, 1),
            ("lamp", 0, 4),
            ("lamp", 1, 4),
        ):
            with self.subTest(task=task, pair_idx=pair_idx):
                annotator = RealSkillAnnotator(task)
                grasp, _, part, placed = _begin_operated_part(
                    annotator, pair_idx
                )
                self.assertEqual(placed["skill"], "place")
                released = copy.deepcopy(grasp)
                released["robot_state"]["gripper_width"] = 0.08
                result = None
                for frame_idx in range(3, 3 + confirmation_frames):
                    result = annotator.annotate_observation(
                        released, _camera_info(), frame_idx=frame_idx
                    )
                self.assertEqual(result["skill"], "pick")
                self.assertEqual(part.skill_state, "pick")

    def test_stale_part_pose_cannot_enter_insert(self):
        annotator = RealSkillAnnotator("round_table")
        grasp, _, leg, _ = _begin_operated_part(annotator, 0)
        tracker = annotator._tracked_parts[leg.name]
        tracker.attached = False
        tracker.pose_april = _matrix_to_pose_vector(
            annotator.robot_to_april
            @ leg.skill_target_part_pose_robot.detach().cpu().numpy()
        )
        annotator._attached_part_name = None
        stale = copy.deepcopy(grasp)
        stale["parts_founds"][leg.part_idx] = False
        stale["robot_state"]["gripper_width"] = 0.08
        result = annotator.annotate_observation(
            stale, _camera_info(), frame_idx=3
        )
        self.assertEqual(result["skill_state"], "leg-top-place")
        self.assertTrue(result["debug"]["blocked_unreliable_transition"])

    def test_lamp_hood_requires_release_and_assembled_geometry(self):
        annotator = RealSkillAnnotator("lamp")
        grasp, base, hood, placed = _begin_operated_part(annotator, 1)
        self.assertEqual(placed["skill_state"], "hood-base-place")
        target = _move_attached_part_to_target(annotator, grasp, hood)
        annotator.annotate_observation(target, _camera_info(), frame_idx=3)
        self.assertEqual(annotator.assemble_idx, 1)
        at_assembly = _detect_assembled_part(annotator, target, base, hood)
        still_closed = annotator.annotate_observation(
            at_assembly, _camera_info(), frame_idx=4
        )
        self.assertEqual(still_closed["skill_state"], "hood-base-place")
        self.assertEqual(annotator.assemble_idx, 1)
        opened = copy.deepcopy(at_assembly)
        opened["robot_state"]["gripper_width"] = 0.08
        annotator.annotate_observation(opened, _camera_info(), frame_idx=5)
        annotator.annotate_observation(opened, _camera_info(), frame_idx=6)
        self.assertEqual(annotator.assemble_idx, 2)

    def test_multi_task_metadata_distinguishes_annotation_and_assembly(self):
        for task in ("round_table", "lamp"):
            for mode in ("online", "offline"):
                with self.subTest(task=task, mode=mode):
                    session = RealSkillAnnotationSession(
                        task, _camera_info(), mode=mode
                    )
                    observation = _multi_task_observation(session.annotator)
                    session.annotate_observation(observation)
                    trajectory = {"metadata": {}, "observations": [observation]}
                    session.update_trajectory_metadata(trajectory)
                    metadata = trajectory["metadata"]["real_skill_annotation"]
                    self.assertEqual(metadata["mode"], mode)
                    self.assertTrue(metadata["complete"])
                    self.assertFalse(metadata["task_fsm_complete"])
                    self.assertEqual(
                        metadata["obstacle_pose_source"], "configured_default"
                    )
                    self.assertEqual(trajectory[ANNOTATION_STATUS_KEY], "annotated")

    def test_stateful_session_supports_online_annotation_and_metadata(self):
        trajectory = _trajectory()
        session = RealSkillAnnotationSession(
            "one_leg",
            trajectory["camera_info"],
            mode="online",
        )

        for observation in trajectory["observations"]:
            session.annotate_observation(observation)
        session.update_trajectory_metadata(trajectory)

        self.assertEqual(session.frame_idx, 3)
        self.assertEqual(session.stats.frame_count, 3)
        self.assertEqual(trajectory["observations"][0]["skill"], "pick")
        self.assertEqual(trajectory["annotation_source"], ANNOTATION_SOURCE)
        self.assertEqual(
            trajectory[ANNOTATION_STATUS_KEY], ANNOTATION_STATUS_ANNOTATED
        )
        metadata = trajectory["metadata"]["real_skill_annotation"]
        self.assertEqual(metadata["mode"], "online")
        self.assertEqual(metadata["stats"]["frame_count"], 3)

    def test_cli_defaults_to_rigid_and_exposes_sam2_backup(self):
        rigid_args = _parse_args(["demo.pkl"])
        self.assertIsNone(rigid_args.sam2_tabletop_recovery)

        sam2_args = _parse_args(
            [
                "demo.pkl",
                "--sam2-tabletop-recovery",
                "recovery.json",
            ]
        )
        self.assertEqual(
            sam2_args.sam2_tabletop_recovery, Path("recovery.json")
        )

    def test_tabletop_place_target_uses_max_xy_socket_and_vertical_offsets(self):
        annotator = RealSkillAnnotator("one_leg")
        table_idx, leg_idx = annotator.furniture.should_be_assembled[0]
        table = annotator.furniture.parts[table_idx]
        leg = annotator.furniture.parts[leg_idx]
        rb_states = torch.tensor(
            [
                [
                    0.2,
                    0.3,
                    0.4,
                    np.sqrt(0.5),
                    0.0,
                    0.0,
                    np.sqrt(0.5),
                ],
                [0.8, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        inputs = {
            "rb_states": rb_states,
            "part_idxs": {table.name: [0], leg.name: [1]},
            "sim_to_april_mat": torch.eye(4),
            "april_to_robot_mat": torch.eye(4),
        }

        details = annotator._tabletop_place_target_details(
            leg, table.name, inputs
        )
        leg_center = details["leg_center_robot"]
        target = details["guidance_robot"]
        inputs["rb_states"][1, :3] = torch.tensor([-9.0, 7.0, 5.0])
        target_after_leg_motion = annotator._tabletop_place_target_details(
            leg, table.name, inputs
        )["guidance_robot"]

        socket_offset_x = abs(float(leg.default_assembled_pose[0, 3]))
        socket_offset_z = abs(float(leg.default_assembled_pose[2, 3]))
        half_longest_length = max(leg.reset_x_len, leg.reset_y_len) * 0.5
        expected_leg_center = torch.tensor(
            [
                0.2 + socket_offset_x,
                0.3 + socket_offset_z,
                0.4 + half_longest_length,
            ],
            dtype=torch.float32,
        )
        expected = expected_leg_center.clone()
        leg_to_ee_z_offset = (
            half_longest_length * 2.0 * LEG_TO_EE_LENGTH_FRACTION
        )
        expected[2] += leg_to_ee_z_offset

        self.assertEqual(details["socket_label"], 3)
        self.assertAlmostEqual(details["longest_leg_length_m"], 0.0875)
        self.assertAlmostEqual(
            details["leg_to_ee_z_offset_m"], leg_to_ee_z_offset
        )
        self.assertAlmostEqual(
            details["total_z_offset_m"],
            half_longest_length + leg_to_ee_z_offset,
        )
        torch.testing.assert_close(leg_center, expected_leg_center)
        torch.testing.assert_close(target, expected)
        torch.testing.assert_close(target_after_leg_motion, expected)

    def test_place_fallback_propagates_latest_leg_pose_with_ee_delta(self):
        annotator = RealSkillAnnotator("one_leg")
        annotator.april_to_robot = np.eye(4, dtype=np.float32)
        annotator.robot_to_april = np.eye(4, dtype=np.float32)
        table_idx, leg_idx = annotator.furniture.should_be_assembled[0]
        table = annotator.furniture.parts[table_idx]
        leg = annotator.furniture.parts[leg_idx]
        rb_states = torch.tensor(
            [
                [
                    0.2,
                    0.3,
                    0.4,
                    np.sqrt(0.5),
                    0.0,
                    0.0,
                    np.sqrt(0.5),
                ],
                [0.8, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        inputs = {
            "rb_states": rb_states,
            "part_idxs": {table.name: [0], leg.name: [1]},
            "sim_to_april_mat": torch.eye(4),
            "april_to_robot_mat": torch.eye(4),
            "ee_pos": torch.tensor([0.02, 0.0, 0.0]),
            "ee_quat": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        }
        tracker = annotator._tracked_parts[leg.name]
        tracker.attached = True
        tracker.rigid_reference_ee_pose_robot = np.eye(4, dtype=np.float32)
        tracker.rigid_reference_part_pose_robot = np.eye(4, dtype=np.float32)
        tracker.rigid_reference_part_pose_robot[:3, 3] = [0.1, 0.0, 0.0]
        tracker.rigid_reference_frame = 7
        annotator._start_place_rigid_reference(leg, inputs)

        fallback_inputs, debug = annotator._place_rigid_fallback_inputs(
            leg, inputs, table.name
        )
        self.assertIsNotNone(fallback_inputs)
        self.assertTrue(debug["place_rigid_fallback_used"])
        self.assertEqual(debug["place_rigid_reference_frame"], 7)
        torch.testing.assert_close(
            fallback_inputs["rb_states"][1, :3],
            torch.tensor([0.12, 0.0, 0.0]),
        )

    def test_inherits_sim_annotator_and_propagates_occluded_pose(self):
        annotator = RealSkillAnnotator("one_leg")
        self.assertIsInstance(annotator, SkillAnnotator)

        first = annotator.annotate_observation(
            _observation(), _camera_info(), frame_idx=0
        )
        annotator.annotate_observation(
            _observation(gripper_width=0.01, table_found=False),
            _camera_info(),
            frame_idx=1,
        )
        third = annotator.annotate_observation(
            _observation(gripper_width=0.01, table_found=False, ee_x=0.4777),
            _camera_info(),
            frame_idx=2,
        )

        self.assertEqual(first["skill"], "pick")
        self.assertIsNotNone(first["guidance_point_2d"]["color_image2"])
        self.assertEqual(
            third["debug"]["part_pose_sources"]["square_table_top"],
            "ee_propagated",
        )
        effective = np.asarray(
            third["debug"]["effective_part_poses_april"]["square_table_top"]
        )
        displacement = np.linalg.norm(effective[:3] - PART_POSES[:3])
        self.assertAlmostEqual(float(displacement), 0.02, places=5)

    def test_release_motion_detaches_before_pose_propagation(self):
        annotator = RealSkillAnnotator("one_leg")
        annotator.annotate_observation(
            _observation(), _camera_info(), frame_idx=0
        )
        annotator.annotate_observation(
            _observation(gripper_width=0.01, table_found=False),
            _camera_info(),
            frame_idx=1,
        )
        attached = annotator.annotate_observation(
            _observation(
                gripper_width=0.01, table_found=False, ee_x=0.4777
            ),
            _camera_info(),
            frame_idx=2,
        )
        released = annotator.annotate_observation(
            _observation(
                gripper_width=0.03, table_found=False, ee_x=0.4977
            ),
            _camera_info(),
            frame_idx=3,
        )

        attached_pose = np.asarray(
            attached["debug"]["effective_part_poses_april"]["square_table_top"]
        )
        released_pose = np.asarray(
            released["debug"]["effective_part_poses_april"]["square_table_top"]
        )
        np.testing.assert_allclose(released_pose, attached_pose, atol=1e-6)
        self.assertEqual(released["debug"]["release_started_part"], "square_table_top")
        self.assertEqual(
            released["debug"]["part_pose_sources"]["square_table_top"],
            "held_last",
        )
        self.assertIsNone(released["debug"]["attached_part"])

    def test_real_push_guidance_uses_tracked_tabletop_height(self):
        annotator = RealSkillAnnotator("one_leg")
        annotator.annotate_observation(
            _observation(), _camera_info(), frame_idx=0
        )
        annotator.annotate_observation(
            _observation(gripper_width=0.01), _camera_info(), frame_idx=1
        )
        push = annotator.annotate_observation(
            _observation(gripper_width=0.01), _camera_info(), frame_idx=2
        )

        tabletop_center_robot = annotator.april_to_robot @ np.append(
            PART_POSES[:3], 1.0
        )
        self.assertEqual(push["skill"], "push")
        self.assertAlmostEqual(
            float(push["guidance_point"][2]),
            float(tabletop_center_robot[2]),
            places=6,
        )
        self.assertAlmostEqual(
            float(push["guidance_pose"][2, 3]),
            float(tabletop_center_robot[2]),
            places=6,
        )
        self.assertEqual(
            push["debug"]["push_target_z_policy"],
            "tracked_tabletop_center",
        )

    def test_tabletop_release_finishes_push_without_displacement_threshold(self):
        annotator = RealSkillAnnotator("one_leg")
        annotator.annotate_observation(
            _observation(), _camera_info(), frame_idx=0
        )
        annotator.annotate_observation(
            _observation(gripper_width=0.01), _camera_info(), frame_idx=1
        )
        push = annotator.annotate_observation(
            _observation(gripper_width=0.01), _camera_info(), frame_idx=2
        )
        after_release = annotator.annotate_observation(
            _observation(gripper_width=0.08), _camera_info(), frame_idx=3
        )

        self.assertEqual(push["skill"], "push")
        self.assertTrue(annotator.furniture.parts[0].pre_assemble_done)
        self.assertEqual(after_release["debug"]["phase"], "assemble")

    def test_defaults_to_new_file_and_supports_atomic_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            input_path = tmp_path / "demo.pkl"
            with input_path.open("wb") as file:
                pickle.dump(_trajectory(), file)

            output_path, stats = annotate_pickle(input_path)
            self.assertEqual(output_path, tmp_path / "demo.annotated.pkl")
            self.assertEqual(stats.frame_count, 3)
            original = load_trajectory_pickle(input_path)
            annotated = load_trajectory_pickle(output_path)
            self.assertIsNone(original["observations"][0]["skill"])
            self.assertEqual(annotated["observations"][0]["skill"], "pick")
            self.assertEqual(annotated["annotation_source"], ANNOTATION_SOURCE)
            self.assertEqual(
                annotated[ANNOTATION_STATUS_KEY], ANNOTATION_STATUS_ANNOTATED
            )
            self.assertIn("target_point", annotated["observations"][0]["guidance"])
            annotation_metadata = annotated["metadata"]["real_skill_annotation"]
            self.assertEqual(
                annotation_metadata["pose_tracking_policy"],
                DEFAULT_POSE_TRACKING_POLICY,
            )
            self.assertEqual(
                annotation_metadata["release_pose_policy"], "held_last"
            )
            self.assertFalse(annotation_metadata["sam2_override_enabled"])

            overwrite_path = tmp_path / "overwrite.pkl"
            with overwrite_path.open("wb") as file:
                pickle.dump(_trajectory(), file)
            written_path, _ = annotate_pickle(overwrite_path, overwrite=True)
            self.assertEqual(written_path, overwrite_path)
            self.assertEqual(
                load_trajectory_pickle(overwrite_path)["annotation_source"],
                ANNOTATION_SOURCE,
            )

    def test_refuses_existing_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            input_path = tmp_path / "demo.pkl"
            output_path = tmp_path / "demo.annotated.pkl"
            with input_path.open("wb") as file:
                pickle.dump(_trajectory(), file)
            output_path.write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                annotate_pickle(input_path)

            with self.assertRaises(ValueError):
                annotate_pickle(input_path, output_path=input_path)

    def test_recovered_pose_provider_does_not_replace_raw_pose_fields(self):
        trajectory = _trajectory()
        raw_pose = trajectory["observations"][2]["parts_poses"].copy()
        raw_founds = trajectory["observations"][2]["parts_founds"].copy()
        recovered_pose = PART_POSES[:7].copy()
        recovered_pose[0] += 0.08
        provider = RecoveredTabletopPoseProvider(
            pose_april=recovered_pose,
            start_frame=2,
            keyframe=2,
            confidence=0.75,
        )

        from src.eval.real_skill_annotation_util import annotate_trajectory

        annotate_trajectory(trajectory, pose_provider=provider)

        frame = trajectory["observations"][2]
        np.testing.assert_array_equal(frame["parts_poses"], raw_pose)
        np.testing.assert_array_equal(frame["parts_founds"], raw_founds)
        self.assertEqual(
            frame["real_annotation_debug"]["part_pose_sources"][
                "square_table_top"
            ],
            "sam2_rgbd_full_tabletop_cad_chamfer",
        )
        self.assertEqual(
            frame["real_annotation_debug"]["pose_override"]["confidence"],
            0.75,
        )
        self.assertEqual(
            trajectory["metadata"]["real_skill_annotation"]["pose_provider"][
                "start_frame"
            ],
            2,
        )
        self.assertTrue(
            trajectory["metadata"]["real_skill_annotation"][
                "sam2_override_enabled"
            ]
        )


if __name__ == "__main__":
    unittest.main()
