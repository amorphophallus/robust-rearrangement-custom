import json

import numpy as np

from src.eval.vlm_guidance import (
    VLMGuidanceClient,
    policy_bundles_from_vlm,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class ErrorResponse(FakeResponse):
    def __init__(self, payload):
        super().__init__(payload)
        self.status_code = 422

    def raise_for_status(self):
        raise RuntimeError("422 Unprocessable Entity")


class FakeSession:
    def __init__(self):
        self.last_files = None
        self.posted_metadata = []

    def post(self, url, *, files, headers, timeout):
        self.last_files = files
        metadata = json.loads(files["metadata"][1])
        self.posted_metadata.append(metadata)
        rows = []
        for item in reversed(metadata["items"]):
            rows.append(
                {
                    "request_id": item["request_id"],
                    "skill": "place",
                    "skill_confidence": None,
                    "skill_probabilities": None,
                    "point_1000": [500.0, 500.0],
                    "point_px": [159.5, 119.5],
                    "generated_text": '{"skill":"place","target_point_2d":[159.5,119.5]}',
                }
            )
        return FakeResponse(
            {
                "model_revision": "revision",
                "policy_version": 3,
                "model_mode": "original_sft",
                "predictions": rows,
                "timing_ms": {"total": 12.0},
            }
        )

    def get(self, url, *, headers, timeout):
        return FakeResponse(
            {
                "status": "ready",
                "model_revision": "revision",
                "policy_version": 3,
                "model_mode": "original_sft",
                "device": "cuda:0",
            }
        )


class ErrorSession(FakeSession):
    def post(self, url, *, files, headers, timeout):
        return ErrorResponse(
            {
                "detail": "env0-step495: target_point_2d is outside the front image"
            }
        )


class StructuredSession(FakeSession):
    def get(self, url, *, headers, timeout):
        response = super().get(url, headers=headers, timeout=timeout)
        response.payload["model_mode"] = "structured"
        return response


class BatchErrorThenSingletonSession(FakeSession):
    def post(self, url, *, files, headers, timeout):
        metadata = json.loads(files["metadata"][1])
        if len(metadata["items"]) > 1:
            self.last_files = files
            self.posted_metadata.append(metadata)
            return ErrorResponse({"detail": "invalid generated JSON"})
        return super().post(url, files=files, headers=headers, timeout=timeout)


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

    assert readiness["policy_version"] == 3
    assert client.ready_model_revision == "revision"
    assert client.ready_model_mode == "original_sft"


def test_original_sft_keeps_batched_request_after_readiness_check():
    session = FakeSession()
    client = VLMGuidanceClient("http://vlm", session=session)
    client.check_ready()
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    predictions, timing = client.predict(
        task="one_leg",
        front_images=[image, image, image],
        wrist_images=[image, image, image],
        state_infos=[{"base": {}}, {"base": {}}, {"base": {}}],
        step_idx=48,
    )

    assert [len(metadata["items"]) for metadata in session.posted_metadata] == [3]
    assert [prediction.request_id for prediction in predictions] == [
        "env0-step48",
        "env1-step48",
        "env2-step48",
    ]
    assert timing["total"] == 12.0


def test_original_sft_retries_batch_422_as_singletons():
    session = BatchErrorThenSingletonSession()
    client = VLMGuidanceClient("http://vlm", session=session)
    client.check_ready()
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    predictions, timing = client.predict(
        task="one_leg",
        front_images=[image, image, image],
        wrist_images=[image, image, image],
        state_infos=[{"base": {}}, {"base": {}}, {"base": {}}],
        step_idx=48,
    )

    assert [len(metadata["items"]) for metadata in session.posted_metadata] == [3, 1, 1, 1]
    assert [prediction.request_id for prediction in predictions] == [
        "env0-step48",
        "env1-step48",
        "env2-step48",
    ]
    assert timing["total"] == 36.0
    assert client.transport_stats() == {
        "model_mode": "original_sft",
        "prediction_http_request_count": 4,
        "batch_prediction_request_count": 1,
        "singleton_prediction_request_count": 3,
        "batch_422_retry_count": 1,
    }


def test_structured_model_keeps_batched_request_after_readiness_check():
    session = StructuredSession()
    client = VLMGuidanceClient("http://vlm", session=session)
    client.check_ready()
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    client.predict(
        task="lamp",
        front_images=[image, image, image],
        wrist_images=[image, image, image],
        state_infos=[{"base": {}}, {"base": {}}, {"base": {}}],
        step_idx=8,
    )

    assert [len(metadata["items"]) for metadata in session.posted_metadata] == [3]


def test_client_preserves_server_error_detail():
    client = VLMGuidanceClient("http://vlm", session=ErrorSession())
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    try:
        client.predict(
            task="round_table",
            front_images=[image],
            wrist_images=[image],
            state_infos=[{"base": {}}],
            step_idx=495,
        )
    except Exception as error:
        message = str(error)
    else:
        raise AssertionError("expected prediction failure")

    assert "outside the front image" in message
    assert "env0-step495" in message


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
    assert bundle["vlm_annotation"]["skill_confidence"] is None
    assert "target_point_2d" in bundle["vlm_annotation"]["generated_text"]
