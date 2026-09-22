import copy
import importlib.util
import os
import sys
import unittest

import numpy as np


PATCHED_MODULE = os.environ.get("PATCHED_REAL_ANNOTATOR")
if PATCHED_MODULE:
    spec = importlib.util.spec_from_file_location(
        "src.eval.real_skill_annotation_util", PATCHED_MODULE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
else:
    import src.eval.real_skill_annotation_util as module


RealSkillAnnotator = module.RealSkillAnnotator
_matrix_to_pose_vector = module._matrix_to_pose_vector


CAMERA_TO_APRIL = np.array(
    [
        [0.99918, -0.00549, -0.04002, 0.03103],
        [-0.03190, 0.50078, -0.86499, 0.67005],
        [0.02479, 0.86556, 0.50020, -0.30727],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _camera_info():
    intrinsics = {
        "fx": 302.0,
        "fy": 302.0,
        "ppx": 160.0,
        "ppy": 120.0,
        "width": 320,
        "height": 240,
    }
    return {
        "front": {"record_intrinsics": intrinsics},
        "wrist": {"record_intrinsics": intrinsics},
    }


def _observation(annotator, *, width=0.08, ee_pose=None):
    poses = []
    for part in annotator.furniture.parts:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.asarray(part.reset_ori[0], dtype=np.float32)[:3, :3]
        pose[:3, 3] = np.asarray(part.reset_pos[0], dtype=np.float32)[:3]
        poses.append(_matrix_to_pose_vector(pose))
    if ee_pose is None:
        ee_pose = np.eye(4, dtype=np.float32)
        ee_pose[:3, 3] = [0.5, 0.0, 0.2]
    return {
        "parts_poses": np.concatenate(poses),
        "parts_founds": np.ones(len(poses), dtype=bool),
        "parts_pose_valid": np.ones(len(poses), dtype=bool),
        "camera_to_april": CAMERA_TO_APRIL.copy(),
        "robot_state": {
            "ee_pose": np.asarray(ee_pose, dtype=np.float32).copy(),
            "wrist_pose": np.eye(4, dtype=np.float32),
            "gripper_width": width,
        },
        "color_image1": np.zeros((8, 8, 3), dtype=np.uint8),
        "color_image2": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def _start_pair(annotator, pair_idx):
    annotator.assemble_idx = pair_idx
    base_idx, operated_idx = annotator.furniture.should_be_assembled[pair_idx]
    base = annotator.furniture.parts[base_idx]
    operated = annotator.furniture.parts[operated_idx]
    base.pre_assemble_done = True
    base.skill_state = "done"
    operated.skill_state = "pick"
    return operated


class LampRealFsmV20Test(unittest.TestCase):
    def test_lamp_does_not_call_shared_part_state_machines(self):
        annotator = RealSkillAnnotator("lamp")
        _start_pair(annotator, 0)
        for part in annotator.furniture.parts:
            part.update_skill_state = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("shared FurnitureBench FSM was called")
            )
        observation = _observation(annotator)
        annotator.annotate_observation(observation, _camera_info(), frame_idx=0)

    def test_bulb_pick_advances_on_close_event_only(self):
        annotator = RealSkillAnnotator("lamp")
        bulb = _start_pair(annotator, 0)
        far = _observation(annotator)
        far["robot_state"]["ee_pose"][:3, 3] = [2.0, 2.0, 2.0]
        annotator.annotate_observation(far, _camera_info(), frame_idx=0)
        closed = copy.deepcopy(far)
        closed["robot_state"]["gripper_width"] = 0.0705
        result = annotator.annotate_observation(
            closed, _camera_info(), frame_idx=1
        )
        self.assertEqual(bulb.skill_state, "place")
        self.assertEqual(result["skill_state"], "bulb-base-place")
        self.assertEqual(
            result["debug"]["pick_transition_policy"],
            "lamp_bulb_partial_close_event_only",
        )

        still_closing = copy.deepcopy(closed)
        still_closing["robot_state"]["gripper_width"] = 0.0657
        result = annotator.annotate_observation(
            still_closing, _camera_info(), frame_idx=2
        )
        self.assertIsNone(result["debug"]["gripper_event"])
        self.assertTrue(result["debug"]["gripper_closed"])

        opened = copy.deepcopy(still_closing)
        opened["robot_state"]["gripper_width"] = 0.071
        result = annotator.annotate_observation(
            opened, _camera_info(), frame_idx=3
        )
        self.assertEqual(result["debug"]["gripper_event"], "opened")
        self.assertFalse(result["debug"]["gripper_closed"])

    def test_bulb_place_ignores_base_local_y_rotation(self):
        annotator = RealSkillAnnotator("lamp")
        _start_pair(annotator, 0)
        observation = _observation(annotator)
        ee_pose = module._ee_pose_robot(observation)
        effective = {
            part.name: annotator._tracked_pose(observation, part, ee_pose)
            for part in annotator.furniture.parts
        }
        inputs = annotator._annotation_inputs(observation, ee_pose, effective)
        _, _, target = annotator._lamp_real_bulb_place_target(inputs)
        base = annotator._part_pose_robot_from_inputs("lamp_base", inputs)
        current_relative = np.linalg.inv(base.cpu().numpy()) @ target.cpu().numpy()
        angle = np.deg2rad(75.0)
        rotate_y = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle), 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        bulb_april = annotator.robot_to_april @ (
            base.cpu().numpy() @ current_relative @ rotate_y
        )
        bulb_idx = annotator.furniture.parts[1].part_idx
        observation["parts_poses"][bulb_idx * 7 : (bulb_idx + 1) * 7] = (
            _matrix_to_pose_vector(bulb_april)
        )
        ee_pose = module._ee_pose_robot(observation)
        effective = {
            part.name: annotator._tracked_pose(observation, part, ee_pose)
            for part in annotator.furniture.parts
        }
        inputs = annotator._annotation_inputs(observation, ee_pose, effective)
        geometry = annotator._lamp_real_bulb_place_geometry(inputs)
        self.assertEqual(geometry["place_ignored_rotation_axis"], "y")
        self.assertLess(geometry["place_orientation_error_rad"], 1e-3)

    def test_bulb_place_guidance_is_target_center_plus_quarter_length(self):
        annotator = RealSkillAnnotator("lamp")
        _start_pair(annotator, 0)
        observation = _observation(annotator)
        ee_pose = module._ee_pose_robot(observation)
        effective = {
            part.name: annotator._tracked_pose(observation, part, ee_pose)
            for part in annotator.furniture.parts
        }
        inputs = annotator._annotation_inputs(observation, ee_pose, effective)
        guidance, guidance_pose, target_bulb_pose = (
            annotator._lamp_real_bulb_place_target(inputs)
        )
        bulb = next(
            part for part in annotator.furniture.parts if part.name == "lamp_bulb"
        )
        expected = target_bulb_pose[:3, 3].clone()
        expected[2] += 0.25 * annotator._longest_part_length(bulb)
        np.testing.assert_allclose(guidance.cpu().numpy(), expected.cpu().numpy())
        np.testing.assert_allclose(
            guidance_pose[:3, 3].cpu().numpy(), expected.cpu().numpy()
        )

    def test_hood_pick_requires_close_event_and_relaxed_distance(self):
        annotator = RealSkillAnnotator("lamp")
        hood = _start_pair(annotator, 1)
        far = _observation(annotator)
        far["robot_state"]["ee_pose"][:3, 3] = [2.0, 2.0, 2.0]
        annotator.annotate_observation(far, _camera_info(), frame_idx=0)
        far_closed = copy.deepcopy(far)
        far_closed["robot_state"]["gripper_width"] = 0.01
        result = annotator.annotate_observation(
            far_closed, _camera_info(), frame_idx=1
        )
        self.assertEqual(hood.skill_state, "pick")
        self.assertFalse(result["debug"]["pick_ee_distance_ok"])

        reopened = copy.deepcopy(far_closed)
        reopened["robot_state"]["gripper_width"] = 0.08
        annotator.annotate_observation(reopened, _camera_info(), frame_idx=2)
        near = copy.deepcopy(reopened)
        hood_pose_april = module._pose_vector_to_matrix(
            near["parts_poses"][hood.part_idx * 7 : (hood.part_idx + 1) * 7]
        )
        hood_pose_robot = annotator.april_to_robot @ hood_pose_april
        near["robot_state"]["ee_pose"][:3, 3] = (
            hood_pose_robot[:3, 3] + np.array([0.12, 0.0, 0.0], dtype=np.float32)
        )
        annotator.annotate_observation(near, _camera_info(), frame_idx=3)
        near_closed = copy.deepcopy(near)
        near_closed["robot_state"]["gripper_width"] = 0.01
        result = annotator.annotate_observation(
            near_closed, _camera_info(), frame_idx=4
        )
        self.assertEqual(hood.skill_state, "place")
        self.assertTrue(result["debug"]["pick_ee_distance_ok"])
        self.assertAlmostEqual(
            result["debug"]["pick_ee_distance_threshold_m"], 0.15
        )

    def test_hood_pick_guidance_uses_the_hood_center_without_offset(self):
        annotator = RealSkillAnnotator("lamp")
        _start_pair(annotator, 1)
        observation = _observation(annotator)
        ee_pose = module._ee_pose_robot(observation)
        effective = {
            part.name: annotator._tracked_pose(observation, part, ee_pose)
            for part in annotator.furniture.parts
        }
        inputs = annotator._annotation_inputs(observation, ee_pose, effective)
        guidance, _ = annotator._lamp_real_hood_pick_target(inputs)
        hood_pose = annotator._part_pose_robot_from_inputs("lamp_hood", inputs)
        np.testing.assert_allclose(
            guidance.cpu().numpy(), hood_pose[:3, 3].cpu().numpy()
        )

    def test_hood_place_guidance_is_assembled_pose_plus_world_z_height(self):
        annotator = RealSkillAnnotator("lamp")
        _start_pair(annotator, 1)
        observation = _observation(annotator)
        ee_pose = module._ee_pose_robot(observation)
        effective = {
            part.name: annotator._tracked_pose(observation, part, ee_pose)
            for part in annotator.furniture.parts
        }
        inputs = annotator._annotation_inputs(observation, ee_pose, effective)
        guidance, guidance_pose, target_hood_pose = (
            annotator._lamp_real_hood_place_target(inputs)
        )
        base_pose = annotator._part_pose_robot_from_inputs("lamp_base", inputs)
        target_relative = np.eye(4, dtype=np.float32)
        target_relative[:3, 3] = module.LAMP_REAL_HOOD_PLACE_REL_POS_M
        expected = base_pose.cpu().numpy() @ target_relative
        expected[2, 3] += module.LAMP_REAL_HOOD_PLACE_HEIGHT_M
        np.testing.assert_allclose(guidance.cpu().numpy(), expected[:3, 3])
        np.testing.assert_allclose(guidance_pose.cpu().numpy(), expected)
        np.testing.assert_allclose(target_hood_pose.cpu().numpy(), expected)


if __name__ == "__main__":
    unittest.main()
