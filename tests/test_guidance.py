import numpy as np

from src.common.guidance import (
    camera_info_with_robot_base,
    robot_base_to_sim_local_from_state,
    transform_guidance_point,
    transform_guidance_pose,
)
from src.eval.skill_annotation_util import project_3d_to_2d


def test_legacy_sim_guidance_converts_to_robot_base():
    robot_to_sim = robot_base_to_sim_local_from_state(
        {
            "ee_pos": np.array([0.57, 0.01, 0.15], dtype=np.float32),
            "ee_pos_sim": np.array([0.27, 0.01, 0.565], dtype=np.float32),
        }
    )
    sim_to_robot = np.linalg.inv(robot_to_sim)
    point_sim = np.array([0.15, 0.02, 0.58], dtype=np.float32)
    pose_sim = np.eye(4, dtype=np.float32)
    pose_sim[:3, 3] = point_sim

    np.testing.assert_allclose(
        transform_guidance_point(point_sim, sim_to_robot),
        [0.45, 0.02, 0.165],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transform_guidance_pose(pose_sim, sim_to_robot)[:3, 3],
        [0.45, 0.02, 0.165],
        atol=1e-6,
    )


def test_robot_base_camera_transform_preserves_legacy_projection():
    robot_to_sim = np.eye(4, dtype=np.float32)
    robot_to_sim[:3, 3] = [-0.3, 0.0, 0.415]
    camera_to_sim = np.eye(4, dtype=np.float32)
    camera_to_sim[:3, 3] = [0.0, 0.0, -1.0]
    legacy_camera = {
        "image_size": np.array([100, 100]),
        "intrinsics": np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "camera_to_sim_local": camera_to_sim,
        "sim_local_to_camera": np.linalg.inv(camera_to_sim),
    }
    camera = camera_info_with_robot_base(legacy_camera, robot_to_sim)
    point_robot = np.array([0.3, 0.0, 0.585], dtype=np.float32)
    point_sim = transform_guidance_point(point_robot, robot_to_sim)

    point_sim_h = np.concatenate([point_sim, [1.0]]).astype(np.float32)
    point_camera = legacy_camera["sim_local_to_camera"] @ point_sim_h
    expected_u = round(100.0 * point_camera[0] / point_camera[2] + 50.0)
    expected_v = round(-100.0 * point_camera[1] / point_camera[2] + 50.0)

    np.testing.assert_array_equal(
        project_3d_to_2d(point_robot, camera),
        [expected_u, expected_v],
    )
