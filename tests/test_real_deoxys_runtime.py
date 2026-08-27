import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.real.deoxys_runtime import (
    GripperStateSample,
    RobotStateSample,
    interpolate_gripper_width,
    interpolate_robot_state,
)


class RealDeoxysRuntimeTest(unittest.TestCase):
    def test_robot_and_gripper_align_to_camera_source_time(self):
        samples = []
        for time_ns, x, yaw in ((110_000_000, 0.0, 0.0), (210_000_000, 1.0, 1.0)):
            pose = np.eye(4)
            pose[:3, 3] = [x, 0, 0]
            pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            samples.append(
                RobotStateSample(
                    receive_wall_time_ns=time_ns,
                    source_time=time_ns / 1e9,
                    frame=time_ns,
                    wrist_pose=pose,
                    joint_positions=np.full(7, x),
                    joint_velocities=np.full(7, 2 * x),
                    joint_torques=np.full(7, 3 * x),
                )
            )
        aligned = interpolate_robot_state(
            samples, 150_000_000, observation_latency_ms=10
        )
        np.testing.assert_allclose(aligned.wrist_pose[:3, 3], [0.5, 0, 0])
        np.testing.assert_allclose(
            Rotation.from_matrix(aligned.wrist_pose[:3, :3]).as_rotvec(),
            [0, 0, 0.5],
        )
        gripper = [
            GripperStateSample(120_000_000, 0.0, 0.08),
            GripperStateSample(220_000_000, 0.1, 0.00),
        ]
        self.assertAlmostEqual(
            interpolate_gripper_width(
                gripper, 150_000_000, observation_latency_ms=20
            ),
            0.04,
        )


if __name__ == "__main__":
    unittest.main()
