import pickle
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
    RealSkillAnnotator,
    RealSkillAnnotationSession,
    _parse_args,
    annotate_pickle,
    load_trajectory_pickle,
    PLACE_TARGET_POLICY_TABLETOP,
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


class RealSkillAnnotationUtilTest(unittest.TestCase):
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
