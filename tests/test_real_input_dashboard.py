import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.real.input_dashboard import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    EvalInputDashboard,
    EvalInputVideoRecorder,
    VIDEO_FRAME_HEIGHT,
    VIDEO_FRAME_WIDTH,
    render_input_dashboard,
    render_rgbd_video_frame,
)


class RealInputDashboardTest(unittest.TestCase):
    def _policy_inputs(self):
        return {
            "wrist": np.concatenate(
                [
                    np.full((1, 3, 240, 320), 0.25, dtype=np.float32),
                    np.full((1, 1, 240, 320), 0.7, dtype=np.float32),
                ],
                axis=1,
            ),
            "front": np.concatenate(
                [
                    np.full((1, 3, 240, 320), 0.5, dtype=np.float32),
                    np.full((1, 1, 240, 320), 0.9, dtype=np.float32),
                ],
                axis=1,
            ),
        }

    def _observation(self):
        return {
            "color_image1": np.full((240, 320, 3), 32, dtype=np.uint8),
            "color_image2": np.full((240, 320, 3), 96, dtype=np.uint8),
            "depth_image1": np.full((240, 320), 0.7, dtype=np.float32),
            "depth_image2": np.linspace(
                0.4, 1.2, 240 * 320, dtype=np.float32
            ).reshape(240, 320),
            "robot_state": {
                "ee_pos": np.asarray([0.5, 0.0, 0.2]),
                "ee_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "ee_pos_vel": np.zeros(3),
                "ee_ori_vel": np.zeros(3),
                "joint_positions": np.zeros(7),
                "joint_velocities": np.zeros(7),
                "joint_torques": np.zeros(7),
                "gripper_width": 0.08,
            },
            "skill": "pick",
            "skill_state": "pick",
            "assembly_step": 0,
            "guidance_point": np.asarray([0.5, 0.1, 0.02]),
            "guidance_point_2d": {
                "color_image1": np.asarray([140, 100]),
                "color_image2": np.asarray([160, 120]),
            },
            "parts_founds": np.ones(6, dtype=bool),
            "parts_pose_valid": np.ones(6, dtype=bool),
            "real_annotation_debug": {"part_pose_sources": {"leg": "apriltag"}},
        }

    def test_renders_four_inputs_without_mutating_policy_rgb(self):
        observation = self._observation()
        front_before = observation["color_image2"].copy()
        frame = render_input_dashboard(
            observation,
            policy_inputs=self._policy_inputs(),
            query_id=3,
            annotation_mode="guidance-point-colored",
            timing={
                "front_frame_number": 10,
                "wrist_frame_number": 11,
                "front_age_ms_at_build": 24.0,
                "wrist_residual_ms": 2.0,
                "prompt_depth_latency_ms": 41.0,
            },
            inference_latency_ms=38.0,
            action_chunk=np.tile(
                np.asarray([0.5, 0.0, 0.2, 1, 0, 0, 0, 1, 0, -1.0]),
                (8, 1),
            ),
            actions_accepted=6,
            common_stale_prefix=2,
            query_interval_steps=3,
            scheduled=True,
            arm_queue_pending=3,
            gripper_queue_pending=2,
            immutable_coverage_ms=600.0,
            occupied_preserved=2,
            warmstart_mapped_count=3,
        )

        self.assertEqual(frame.shape, (CANVAS_HEIGHT, CANVAS_WIDTH, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(frame.max()), 0)
        np.testing.assert_array_equal(observation["color_image2"], front_before)

    def test_invalid_depth_pixels_do_not_break_rendering(self):
        observation = self._observation()
        observation["depth_image1"][:] = np.nan
        frame = render_input_dashboard(
            observation,
            policy_inputs=self._policy_inputs(),
            query_id=0,
            annotation_mode="none",
            timing={},
            inference_latency_ms=0,
            action_chunk=np.zeros((0, 10), dtype=np.float32),
            actions_accepted=0,
            common_stale_prefix=0,
            query_interval_steps=3,
            scheduled=False,
        )
        self.assertEqual(frame.shape, (CANVAS_HEIGHT, CANVAS_WIDTH, 3))

    def test_captures_exact_camera_transform_outputs(self):
        actor = SimpleNamespace(
            camera1_transform=torch.nn.Identity(),
            camera2_transform=torch.nn.Identity(),
        )
        dashboard = EvalInputDashboard(enabled=True, actor=actor)
        wrist = torch.rand(1, 4, 240, 320)
        front = torch.rand(1, 4, 240, 320)

        actor.camera1_transform(wrist)
        actor.camera2_transform(front)

        torch.testing.assert_close(dashboard._policy_inputs["wrist"], wrist)
        torch.testing.assert_close(dashboard._policy_inputs["front"], front)
        dashboard.enabled = False
        dashboard.close()
        self.assertFalse(actor.camera1_transform._forward_hooks)
        self.assertFalse(actor.camera2_transform._forward_hooks)

    def test_renders_front_wrist_rgbd_video_grid(self):
        frame = render_rgbd_video_frame(self._observation())
        self.assertEqual(
            frame.shape,
            (VIDEO_FRAME_HEIGHT, VIDEO_FRAME_WIDTH, 3),
        )
        self.assertEqual(frame.dtype, np.uint8)

    def test_video_recorder_writes_all_submitted_query_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query-rgbd.mp4"
            recorder = EvalInputVideoRecorder(enabled=True, fps=2.5)
            recorder.start(path)
            self.assertTrue(recorder.submit(self._observation()))
            self.assertTrue(recorder.submit(self._observation()))
            result = recorder.close()

            self.assertIsNone(result["error"])
            self.assertEqual(result["frame_count"], 2)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
