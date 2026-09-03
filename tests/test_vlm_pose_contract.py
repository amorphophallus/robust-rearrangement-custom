from pathlib import Path

import numpy as np

from services.vlm_guidance.native_sft import parse_native_pose_prediction
from src.eval.vlm_guidance import (
    POSE_OUTPUT_SCHEMA,
    POSE_POLICY_VERSION,
    VLMGuidanceClient,
    materialize_pose_predictions,
    policy_bundles_from_vlm,
)


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _PoseSession:
    def get(self, url, *, headers, timeout):
        return _Response(
            {
                "status": "ready",
                "model_revision": "ckpts_ver2@revision",
                "policy_version": POSE_POLICY_VERSION,
                "output_schema": POSE_OUTPUT_SCHEMA,
                "model_mode": "original_sft",
            }
        )

    def post(self, url, *, files, headers, timeout):
        import json

        metadata = json.loads(files["metadata"][1])
        rows = [
            {
                "request_id": item["request_id"],
                "skill": "pick",
                "skill_confidence": None,
                "skill_probabilities": None,
                "point_1000": [501.5674, 502.0921],
                "point_px": [160.0, 120.0],
                "rotation_6d": [1, 0, 0, 0, 1, 0],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "generated_text": "pose-json",
            }
            for item in metadata["items"]
        ]
        return _Response(
            {
                "model_revision": "ckpts_ver2@revision",
                "policy_version": POSE_POLICY_VERSION,
                "output_schema": POSE_OUTPUT_SCHEMA,
                "model_mode": "original_sft",
                "predictions": rows,
                "timing_ms": {"total": 1.0},
            }
        )


class _InvalidPoseSession(_PoseSession):
    def post(self, url, *, files, headers, timeout):
        import json

        metadata = json.loads(files["metadata"][1])
        rows = [
            {
                "request_id": item["request_id"],
                "valid": False,
                "skill": None,
                "point_1000": None,
                "point_px": None,
                "rotation_6d": None,
                "rotation_matrix": None,
                "generated_text": "degenerate-rotation-json",
                "parse_error": "target_rotation_6d second row is degenerate",
            }
            for item in metadata["items"]
        ]
        return _Response(
            {
                "model_revision": "ckpts_ver2@revision",
                "policy_version": POSE_POLICY_VERSION,
                "output_schema": POSE_OUTPUT_SCHEMA,
                "model_mode": "original_sft",
                "predictions": rows,
                "timing_ms": {"total": 1.0},
            }
        )


def _camera_bundle():
    intrinsics = np.array(
        [[100.0, 0.0, 160.0], [0.0, 100.0, 120.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return {
        "skill": "place",
        "guidance_pose": np.diag([1.0, -1.0, -1.0, 1.0]),
        "guidance_point": np.array([9.0, 9.0, 9.0]),
        "guidance_gripper_width": 0.123,
        "guidance_point_2d": {"color_image2": np.array([5.0, 6.0])},
        "grasp_annotation_2d": {
            "color_image2": {
                "style": "grasp_rect",
                "center": np.array([5.0, 6.0]),
                "corners": np.ones((4, 2)),
            }
        },
        "camera_info": {
            "color_image2": {
                "intrinsics": intrinsics,
                "camera_to_robot_base": np.eye(4, dtype=np.float32),
                "robot_base_to_camera": np.eye(4, dtype=np.float32),
                "image_size": np.array([320, 240]),
            }
        },
    }


def test_hy_furniture_parser_is_the_selected_ver2_source(monkeypatch):
    source = Path("logs/vlm_grasp_ver2/hy_furniture").resolve()
    monkeypatch.setenv("VLM_HY_FURNITURE_ROOT", str(source))
    prediction = parse_native_pose_prediction(
        '{"skill":"place","target_point_2d":[160,120],'
        '"target_rotation_6d":[2,0,0,1,3,0]}'
    )
    assert prediction.rotation_matrix[1] == (0.0, 1.0, 0.0)


def test_pose_client_depth_geometry_and_bundle_do_not_use_oracle_pose():
    client = VLMGuidanceClient(
        "http://vlm",
        session=_PoseSession(),
        expected_policy_version=POSE_POLICY_VERSION,
        expected_output_schema=POSE_OUTPUT_SCHEMA,
    )
    client.check_ready()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    predictions, _ = client.predict(
        task="one_leg",
        front_images=[image],
        wrist_images=[image],
        state_infos=[{"base": {}}],
        step_idx=8,
    )
    depth = np.full((240, 320), 0.5, dtype=np.float32)
    materialized = materialize_pose_predictions(
        [_camera_bundle()], predictions, [depth]
    )
    bundle = policy_bundles_from_vlm(
        [_camera_bundle()], materialized, step_idx=11
    )[0]

    np.testing.assert_allclose(bundle["guidance_pose"][:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(bundle["guidance_pose"][:3, 3], [0.0, 0.0, 0.5])
    assert bundle["grasp_annotation_2d"]["color_image2"] is not None
    assert bundle["vlm_annotation"]["sampled_depth_m"] == 0.5
    assert bundle["vlm_annotation"]["depth_valid_count"] == 25
    assert bundle["vlm_annotation"]["cache_age_steps"] == 3
    assert bundle["guidance_gripper_width"] == 0.05
    assert not np.array_equal(bundle["guidance_point"], [9.0, 9.0, 9.0])


def test_invalid_depth_blanks_grasp_without_oracle_fallback():
    client = VLMGuidanceClient(
        "http://vlm",
        session=_PoseSession(),
        expected_policy_version=POSE_POLICY_VERSION,
        expected_output_schema=POSE_OUTPUT_SCHEMA,
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    predictions, _ = client.predict(
        task="one_leg",
        front_images=[image],
        wrist_images=[image],
        state_infos=[{"base": {}}],
        step_idx=0,
    )
    materialized = materialize_pose_predictions(
        [_camera_bundle()], predictions, [np.zeros((240, 320), dtype=np.float32)]
    )
    bundle = policy_bundles_from_vlm([_camera_bundle()], materialized, step_idx=0)[0]

    assert bundle["guidance_pose"] is None
    assert bundle["guidance_gripper_width"] is None
    assert bundle["grasp_annotation_2d"]["color_image2"] is None
    assert bundle["vlm_annotation"]["grasp_projection_valid"] is False


def test_isaac_negative_depth_uses_the_sensor_distance():
    client = VLMGuidanceClient(
        "http://vlm",
        session=_PoseSession(),
        expected_policy_version=POSE_POLICY_VERSION,
        expected_output_schema=POSE_OUTPUT_SCHEMA,
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    predictions, _ = client.predict(
        task="one_leg",
        front_images=[image],
        wrist_images=[image],
        state_infos=[{"base": {}}],
        step_idx=0,
    )
    materialized = materialize_pose_predictions(
        [_camera_bundle()], predictions, [np.full((240, 320), -0.75, dtype=np.float32)]
    )

    assert materialized[0].sampled_depth_m == 0.75
    np.testing.assert_allclose(materialized[0].guidance_pose[:3, 3], [0, 0, 0.75])


def test_invalid_pose_generation_becomes_blank_without_oracle_fallback():
    client = VLMGuidanceClient(
        "http://vlm",
        session=_InvalidPoseSession(),
        expected_policy_version=POSE_POLICY_VERSION,
        expected_output_schema=POSE_OUTPUT_SCHEMA,
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    predictions, _ = client.predict(
        task="round_table",
        front_images=[image],
        wrist_images=[image],
        state_infos=[{"base": {}}],
        step_idx=16,
    )
    materialized = materialize_pose_predictions(
        [_camera_bundle()], predictions, [np.full((240, 320), -0.75)]
    )
    bundle = policy_bundles_from_vlm([_camera_bundle()], materialized, step_idx=17)[0]

    assert bundle["skill"] is None
    assert bundle["guidance_point_2d"]["color_image2"] is None
    assert bundle["guidance_pose"] is None
    assert bundle["guidance_gripper_width"] is None
    assert bundle["grasp_annotation_2d"]["color_image2"] is None
    assert bundle["vlm_annotation"]["model_output_valid"] is False
    assert "degenerate" in bundle["vlm_annotation"]["parse_error"]
