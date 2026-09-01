import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.common.pickle_compat import load_pickle_file, load_pickle_path
from src.data_collection.pickle_migration import (
    canonicalize_furniturebench_trajectory,
)
from src.data_collection.io import save_raw_rollout
from src.data_collection.pickle_validator import (
    PickleContractError,
    observation_indices,
    validate_pickle_trajectory,
)


def _state():
    finger = np.array([0.04], dtype=np.float32)
    return {
        "ee_pos": np.array([0.4, 0.0, 0.2], dtype=np.float32),
        "ee_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "ee_pos_sim": np.array([0.4, 0.0, 0.2], dtype=np.float32),
        "ee_quat_sim": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "ee_pos_vel": np.zeros(3, dtype=np.float32),
        "ee_ori_vel": np.zeros(3, dtype=np.float32),
        "gripper_width": finger * 2,
        "joint_positions": np.zeros(7, dtype=np.float32),
        "joint_velocities": np.zeros(7, dtype=np.float32),
        "joint_torques": np.zeros(9, dtype=np.float32),
        "gripper_finger_1_pos": finger.copy(),
        "gripper_finger_2_pos": finger.copy(),
    }


def _observation():
    pose = np.array([0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return {
        "robot_state": _state(),
        "color_image1": np.zeros((224, 224, 3), dtype=np.uint8),
        "color_image2": np.zeros((224, 224, 3), dtype=np.uint8),
        "depth_image1": np.full((224, 224), 0.1, dtype=np.float32),
        "depth_image2": np.full((224, 224), 1.0, dtype=np.float32),
        "parts_poses": pose,
        "point_cloud": None,
        "skill": "pick",
        "guidance_point": np.array([0.2, 0.0, 0.1], dtype=np.float32),
        "guidance_point_clean": np.array([0.2, 0.0, 0.1], dtype=np.float32),
        "guidance_pose": None,
        "guidance_pose_clean": None,
        "guidance_gripper_width": None,
        "guidance_point_2d": {
            "color_image1": None,
            "color_image2": np.array([112.0, 112.0], dtype=np.float32),
        },
        "grasp_annotation_2d": {"color_image1": None, "color_image2": None},
    }


def _trajectory():
    camera_to_base = np.eye(4, dtype=np.float32)
    return {
        "env": "FurnitureBench",
        "task": "one_leg",
        "success": True,
        "action_type": "delta",
        "observations": [_observation(), _observation(), _observation()],
        "actions": [
            [0.01, 0, 0, 0, 0, 0, 1, -1],
            [0, 0.01, 0, 0, 0, 0, 1, 1],
        ],
        "rewards": [0.0, 1.0],
        "camera_info": {
            "front_camera": {
                "image_size": np.array([224, 224], dtype=np.int32),
                "intrinsics": np.array(
                    [[307.6, 0, 112], [0, 308.0, 112], [0, 0, 1]],
                    dtype=np.float32,
                ),
                "camera_to_sim_local": camera_to_base,
                "sim_local_to_camera": camera_to_base.copy(),
            }
        },
    }


class RawPickleValidatorTest(unittest.TestCase):
    def test_numpy2_core_namespace_loads_under_numpy1(self):
        payload = b"cnumpy._core.multiarray\n_reconstruct\n."
        reconstructed = load_pickle_file(BytesIO(payload))
        from numpy.core.multiarray import _reconstruct

        self.assertIs(reconstructed, _reconstruct)

    def test_valid_canonical_trajectory(self):
        summary = validate_pickle_trajectory(
            _trajectory(), front_focal_reference=307.6, front_focal_rtol=0.01
        )
        self.assertEqual(summary["transitions"], 2)
        self.assertEqual(summary["observations_checked"], 3)
        self.assertEqual(summary["depth_ranges"]["depth_image2"], (1.0, 1.0))

    def test_rejects_scalar_maniskill_gripper_state(self):
        trajectory = _trajectory()
        trajectory["env"] = "ManiSkill"
        trajectory["observations"][0]["robot_state"]["gripper_width"] = 0.08
        with self.assertRaisesRegex(PickleContractError, r"expected numpy.ndarray"):
            validate_pickle_trajectory(trajectory)

    def test_rejects_negative_depth_and_nonunit_action(self):
        trajectory = _trajectory()
        trajectory["observations"][1]["depth_image2"][0, 0] = -1.0
        with self.assertRaisesRegex(PickleContractError, "positive metres"):
            validate_pickle_trajectory(trajectory)

        trajectory = _trajectory()
        trajectory["actions"][0][6] = 0.5
        with self.assertRaisesRegex(PickleContractError, "unit xyzw"):
            validate_pickle_trajectory(trajectory)

    def test_rejects_automate_focal_mismatch_when_requested(self):
        trajectory = _trajectory()
        trajectory["env"] = "AutoMate"
        trajectory["camera_info"]["front_camera"]["intrinsics"][0, 0] = 616.0
        with self.assertRaisesRegex(PickleContractError, "differs from reference"):
            validate_pickle_trajectory(
                trajectory, front_focal_reference=307.6, front_focal_rtol=0.15
            )

    def test_rejects_world_frame_camera_translation_when_base_is_requested(self):
        trajectory = _trajectory()
        trajectory["camera_info"]["front_camera"]["camera_to_sim_local"][
            :3, 3
        ] = [0.9, 0.0, 0.65]
        trajectory["camera_info"]["front_camera"]["sim_local_to_camera"] = (
            np.linalg.inv(
                trajectory["camera_info"]["front_camera"]["camera_to_sim_local"]
            ).astype(np.float32)
        )
        with self.assertRaisesRegex(PickleContractError, "robot-base reference"):
            validate_pickle_trajectory(
                trajectory,
                front_translation_reference=[1.2, 0.0, 0.235],
                front_translation_atol=0.1,
            )

    def test_rejects_guidance_point_in_wrong_camera_frame(self):
        trajectory = _trajectory()
        for observation in trajectory["observations"]:
            observation["guidance_point"] = np.array(
                [0.0, 0.0, 1.0], dtype=np.float32
            )
            observation["guidance_point_clean"] = observation[
                "guidance_point"
            ].copy()
            observation["oracle_guidance_point_2d"] = {
                "color_image2": np.array([112.0, 112.0], dtype=np.float32)
            }
        validate_pickle_trajectory(trajectory)

        trajectory["observations"][1]["guidance_point_clean"][0] = 0.1
        with self.assertRaisesRegex(
            PickleContractError, "calibrated base-frame 3D guidance point"
        ):
            validate_pickle_trajectory(trajectory)

    def test_even_observation_sampling_includes_endpoints(self):
        self.assertEqual(observation_indices(11, 3), [0, 5, 10])
        self.assertEqual(observation_indices(2, None), [0, 1])

    def test_legacy_fb_migration_repairs_depth_images_state_and_camera(self):
        trajectory = _trajectory()
        trajectory["action_type"] = "pos"
        trajectory["rewards"] = [0.0, 1.0, 99.0]
        for observation in trajectory["observations"]:
            observation["color_image1"] = np.pad(
                observation["color_image1"], ((8, 8), (48, 48), (0, 0))
            )
            observation["color_image2"] = np.pad(
                observation["color_image2"], ((8, 8), (48, 48), (0, 0))
            )
            observation["depth_image1"] = -np.pad(
                observation["depth_image1"], ((8, 8), (48, 48))
            )
            observation["depth_image2"] = -np.pad(
                observation["depth_image2"], ((8, 8), (48, 48))
            )
            state = observation["robot_state"]
            state["ee_pos_sim"] = state["ee_pos"] + np.array(
                [-0.3, 0.0, 0.415], dtype=np.float32
            )
        front = trajectory["camera_info"]["front_camera"]
        front["image_size"] = np.array([320, 240], dtype=np.int32)
        front["intrinsics"][0, 2] = 160
        front["intrinsics"][1, 2] = 120
        front["camera_to_sim_local"][:3, 3] = [0.9, 0.0, 0.65]
        front["sim_local_to_camera"] = np.linalg.inv(
            front["camera_to_sim_local"]
        ).astype(np.float32)

        canonical = canonicalize_furniturebench_trajectory(
            trajectory, legacy_pos_actions_are_delta=True
        )
        summary = validate_pickle_trajectory(canonical)

        self.assertEqual(summary["transitions"], 2)
        self.assertEqual(canonical["observations"][0]["color_image1"].shape, (224, 224, 3))
        self.assertGreaterEqual(canonical["observations"][0]["depth_image1"].min(), 0)
        np.testing.assert_allclose(
            canonical["camera_info"]["front_camera"]["camera_to_sim_local"][:3, 3],
            [1.2, 0.0, 0.235],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            canonical["observations"][0]["parts_poses"][:3],
            [0.5, 0.0, -0.315],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            canonical["observations"][0]["guidance_point"],
            [0.5, 0.0, -0.315],
            atol=1.0e-6,
        )

    def test_furniturebench_writer_emits_a_canonical_raw_pickle(self):
        observations = 3
        robot_states = []
        for _ in range(observations):
            state = _state()
            state["ee_pos_sim"] = state["ee_pos"] + np.array(
                [-0.3, 0.0, 0.415], dtype=np.float32
            )
            robot_states.append(state)
        colors = np.zeros((observations, 240, 320, 3), dtype=np.uint8)
        depths = -np.ones((observations, 240, 320), dtype=np.float32)
        part_pose = np.array(
            [0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
        camera_to_world = np.eye(4, dtype=np.float32)
        camera_to_world[:3, 3] = [0.9, 0.0, 0.65]
        calibration = {
            "image_size": np.array([320, 240], dtype=np.int32),
            "intrinsics": np.array(
                [[307.6, 0, 160], [0, 308.0, 120], [0, 0, 1]],
                dtype=np.float32,
            ),
            "camera_to_sim_local": camera_to_world,
            "sim_local_to_camera": np.linalg.inv(camera_to_world).astype(np.float32),
        }
        actions = np.array(
            [
                [0.01, 0, 0, 0, 0, 0, 1, -1],
                [0, 0.01, 0, 0, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )

        with TemporaryDirectory() as tmpdir:
            save_raw_rollout(
                robot_states=np.asarray(robot_states, dtype=object),
                imgs1=colors,
                imgs2=colors,
                depth_image1=depths,
                depth_image2=depths,
                skills=None,
                guidance_points=[
                    np.array([0.2, 0.0, 0.5], dtype=np.float32)
                ]
                * observations,
                guidance_poses=None,
                guidance_gripper_widths=None,
                guidance_points_2d=None,
                grasp_annotations_2d=None,
                camera_infos=[{"color_image2": calibration}] * observations,
                actions=actions,
                rewards=np.array([0.0, 1.0, 99.0], dtype=np.float32),
                parts_poses=np.repeat(part_pose[None], observations, axis=0),
                success=True,
                task="one_leg",
                action_type="delta",
                rollout_save_dir=Path(tmpdir),
                output_only_pickle=True,
            )
            output_path = next((Path(tmpdir) / "success").glob("*.pkl"))
            written = load_pickle_path(output_path)

        summary = validate_pickle_trajectory(
            written,
            front_focal_reference=307.6,
            front_translation_reference=[1.2, 0.0, 0.235],
            front_translation_atol=1.0e-5,
        )
        self.assertEqual(summary["transitions"], 2)
        self.assertEqual(written["action_type"], "delta")
        self.assertEqual(written["image_annotation_mode"], "none")
        self.assertEqual(len(written["rewards"]), 2)
        np.testing.assert_allclose(
            written["observations"][0]["parts_poses"][:3],
            [0.5, 0.0, -0.315],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            written["observations"][0]["guidance_point"],
            [0.5, 0.0, 0.085],
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
