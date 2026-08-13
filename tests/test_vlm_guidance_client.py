import json

import numpy as np

from src.eval.vlm_guidance import (
    VLMGuidanceClient,
    policy_bundles_from_vlm,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.last_files = None

    def post(self, url, *, files, headers, timeout):
        self.last_files = files
        metadata = json.loads(files["metadata"][1])
        rows = []
        for item in reversed(metadata["items"]):
            rows.append(
                {
                    "request_id": item["request_id"],
                    "skill": "place",
                    "skill_confidence": 0.8,
                    "skill_probabilities": {
                        "push": 0.0,
                        "pick": 0.0,
                        "place": 0.8,
                        "insert": 0.1,
                        "screw": 0.1,
                    },
                    "point_1000": [500.0, 500.0],
                    "point_px": [159.5, 119.5],
                }
            )
        return FakeResponse(
            {
                "model_revision": "revision",
                "policy_version": 2,
                "predictions": rows,
                "timing_ms": {"total": 12.0},
            }
        )

    def get(self, url, *, headers, timeout):
        return FakeResponse(
            {
                "status": "ready",
                "model_revision": "revision",
                "policy_version": 2,
                "device": "cuda:0",
            }
        )


def test_client_batches_images_and_restores_request_order():
    session = FakeSession()
    client = VLMGuidanceClient("http://vlm", session=session)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    predictions, timing = client.predict(
        task="one_leg",
        front_images=[image, image],
        wrist_images=[image, image],
        state_infos=[{"base": {}}, {"base": {}}],
        step_idx=8,
    )

    assert [prediction.request_id for prediction in predictions] == [
        "env0-step8",
        "env1-step8",
    ]
    assert all(prediction.query_step == 8 for prediction in predictions)
    assert timing["total"] == 12.0
    assert session.last_files["front_0"][2] == "image/png"


def test_client_validates_ready_policy_version():
    client = VLMGuidanceClient("http://vlm", session=FakeSession())
    readiness = client.check_ready()

    assert readiness["policy_version"] == 2
    assert client.ready_model_revision == "revision"


def test_policy_bundle_preserves_oracle_diagnostics_but_uses_vlm_outputs():
    session = FakeSession()
    client = VLMGuidanceClient("http://vlm", session=session)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    predictions, _ = client.predict(
        task="one_leg",
        front_images=[image],
        wrist_images=[image],
        state_infos=[{"base": {}}],
        step_idx=0,
    )
    oracle = {
        "skill": "pick",
        "guidance_point_2d": {"color_image2": np.array([10.0, 20.0])},
        "guidance_pose": np.eye(4),
    }
    bundle = policy_bundles_from_vlm([oracle], predictions, step_idx=3)[0]

    assert bundle["skill"] == "place"
    np.testing.assert_array_equal(bundle["guidance_point_2d"]["color_image2"], [159.5, 119.5])
    assert bundle["oracle_skill"] == "pick"
    np.testing.assert_array_equal(
        bundle["oracle_guidance_point_2d"]["color_image2"], [10.0, 20.0]
    )
    assert bundle["vlm_annotation"]["cache_age_steps"] == 3
