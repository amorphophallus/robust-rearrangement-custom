"""HTTP client and rollout helpers for remote VLM annotations."""

from __future__ import annotations

import ast
import io
import json
import logging
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import requests
from PIL import Image

try:
    import torch
except ImportError:  # Lightweight client-only environments need not install torch.
    torch = None


STATE_INFO_BASE_KEYS = (
    "ee_pos_sim",
    "ee_quat_sim",
    "ee_pos_vel",
    "ee_ori_vel",
    "gripper_width",
)
VALID_SKILLS = {"push", "pick", "place", "insert", "screw"}
EXPECTED_POLICY_VERSION = 3
POSE_POLICY_VERSION = 4
POSE_OUTPUT_SCHEMA = "skill_point_rotation6d"
LOGGER = logging.getLogger(__name__)


class VLMGuidanceError(RuntimeError):
    """Raised when remote annotations cannot safely be used."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_payload: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_payload = response_payload


@dataclass(frozen=True)
class VLMPrediction:
    request_id: str
    skill: str | None
    skill_confidence: float | None
    skill_probabilities: dict[str, float] | None
    point_1000: np.ndarray | None
    point_px: np.ndarray | None
    model_revision: str
    query_step: int
    model_output_valid: bool = True
    parse_error: str | None = None
    pose_contract: bool = False
    generated_text: str | None = None
    recovered_from_invalid_json: bool = False
    rotation_6d: np.ndarray | None = None
    rotation_matrix: np.ndarray | None = None
    guidance_pose: np.ndarray | None = None
    grasp_annotation_2d: dict[str, Any] | None = None
    sampled_depth_m: float | None = None
    depth_valid_count: int = 0
    grasp_projection_valid: bool | None = None


def _parse_recoverable_generated_prefix(text: str) -> tuple[str, list[float]] | None:
    """Recover only a complete skill/2-D-point prefix from malformed JSON.

    The original-SFT model also emits a target_point_3d field, but the 2-D
    guidance policy does not consume it. A rare malformed suffix must not make
    us accept a partial or invalid control annotation: both control fields are
    reconstructed as a standalone JSON object and then validated strictly.
    """

    object_start = text.find("{")
    key_start = text.find('"target_point_2d"', object_start + 1)
    list_start = text.find("[", key_start + 1)
    list_end = text.find("]", list_start + 1)
    if min(object_start, key_start, list_start, list_end) < 0:
        return None
    try:
        payload = json.loads(text[object_start : list_end + 1] + "}")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    skill = payload.get("skill")
    point = payload.get("target_point_2d")
    if skill not in VALID_SKILLS or not isinstance(point, list) or len(point) != 2:
        return None
    try:
        point_px = [float(point[0]), float(point[1])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in point_px):
        return None
    if not (0.0 <= point_px[0] <= 319.0 and 0.0 <= point_px[1] <= 239.0):
        return None
    return str(skill), point_px


def _json_value(value: Any, env_idx: int):
    if value is None:
        return None
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] > env_idx:
        value = value[env_idx]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def state_info_for_env(robot_state: Mapping[str, Any], env_idx: int) -> dict[str, Any]:
    return {
        "base": {
            key: _json_value(robot_state.get(key), env_idx)
            for key in STATE_INFO_BASE_KEYS
        }
    }


def _image_uint8(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim != 3:
        raise VLMGuidanceError(f"expected HxWxC image, got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.size and float(image.max()) <= 1.0:
            image *= 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] != 3:
        raise VLMGuidanceError(f"expected RGB image, got {image.shape}")
    return image


def _png_bytes(value: Any) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(_image_uint8(value), mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


class VLMGuidanceClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        api_token: str | None = None,
        session: requests.Session | None = None,
        expected_policy_version: int = EXPECTED_POLICY_VERSION,
        expected_output_schema: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.api_token = api_token if api_token is not None else os.getenv("VLM_API_TOKEN")
        self.session = session or requests.Session()
        self.expected_policy_version = int(expected_policy_version)
        self.expected_output_schema = expected_output_schema
        self.ready_model_revision: str | None = None
        self.ready_model_mode: str | None = None
        self.prediction_http_request_count = 0
        self.batch_prediction_request_count = 0
        self.singleton_prediction_request_count = 0
        self.prediction_transport_retry_count = 0
        self.batch_422_retry_count = 0
        self.invalid_json_prefix_recovery_count = 0

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_token:
            return {}
        return {"Authorization": f"Bearer {self.api_token}"}

    def check_ready(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/health/ready",
                headers=self.headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise VLMGuidanceError(f"VLM readiness check failed: {exc}") from exc
        if payload.get("status") != "ready":
            raise VLMGuidanceError(f"VLM returned invalid readiness payload: {payload}")
        if payload.get("policy_version") != self.expected_policy_version:
            raise VLMGuidanceError(
                "VLM policy version mismatch: "
                f"expected {self.expected_policy_version}, got {payload.get('policy_version')}"
            )
        if (
            self.expected_output_schema is not None
            and payload.get("output_schema") != self.expected_output_schema
        ):
            raise VLMGuidanceError(
                "VLM output schema mismatch: "
                f"expected {self.expected_output_schema!r}, "
                f"got {payload.get('output_schema')!r}"
            )
        self.ready_model_revision = str(payload.get("model_revision", "unknown"))
        self.ready_model_mode = str(payload.get("model_mode", "unknown"))
        return payload

    def _post_prediction_batch(
        self,
        *,
        task: str,
        items: Sequence[dict[str, Any]],
        front_images: Sequence[Any],
        wrist_images: Sequence[Any],
    ) -> dict[str, Any]:
        self.prediction_http_request_count += 1
        if len(items) > 1:
            self.batch_prediction_request_count += 1
        else:
            self.singleton_prediction_request_count += 1
        files: dict[str, tuple[str | None, bytes | str, str]] = {}
        for request_idx, (front_image, wrist_image) in enumerate(
            zip(front_images, wrist_images)
        ):
            files[f"front_{request_idx}"] = (
                f"front_{request_idx}.png",
                _png_bytes(front_image),
                "image/png",
            )
            files[f"wrist_{request_idx}"] = (
                f"wrist_{request_idx}.png",
                _png_bytes(wrist_image),
                "image/png",
            )
        files["metadata"] = (
            None,
            json.dumps({"task": task, "items": list(items)}, separators=(",", ":")),
            "application/json",
        )
        for attempt in range(2):
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/guidance/predict",
                    files=files,
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
                break
            except requests.exceptions.ReadTimeout as exc:
                # The server may already be generating after it has accepted
                # the request. Replaying an ambiguous read timeout against the
                # single-worker service duplicates inference and worsens the
                # queue, so the caller must use a sufficient read timeout.
                raise VLMGuidanceError(
                    f"VLM prediction request exceeded the read timeout: {exc}"
                ) from exc
            except (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                if attempt == 1:
                    raise VLMGuidanceError(
                        f"VLM prediction request failed after one transport retry: {exc}"
                    ) from exc
                self.prediction_transport_retry_count += 1
                self.prediction_http_request_count += 1
                if len(items) > 1:
                    self.batch_prediction_request_count += 1
                else:
                    self.singleton_prediction_request_count += 1
                LOGGER.warning(
                    "Retrying VLM prediction once after transient transport error: %s",
                    exc,
                )
            except Exception as exc:
                raise VLMGuidanceError(f"VLM prediction request failed: {exc}") from exc
        try:
            response.raise_for_status()
        except Exception as exc:
            try:
                response_payload = response.json()
                response_detail = json.dumps(response_payload, ensure_ascii=False)
            except Exception:
                response_payload = None
                response_detail = str(getattr(response, "text", ""))
            response_detail = response_detail[:2048]
            suffix = f"; response={response_detail}" if response_detail else ""
            status_code = getattr(response, "status_code", None)
            raise VLMGuidanceError(
                f"VLM prediction request failed: {exc}{suffix}",
                status_code=(int(status_code) if status_code is not None else None),
                response_payload=response_payload,
            ) from exc
        try:
            return response.json()
        except Exception as exc:
            raise VLMGuidanceError(
                f"VLM prediction response is not valid JSON: {exc}"
            ) from exc

    def _recover_original_sft_singleton_422(
        self,
        error: VLMGuidanceError,
        *,
        item: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Recover a valid control prefix from one original-SFT 422 response."""

        if (
            error.status_code != 422
            or self.ready_model_mode != "original_sft"
            or self.expected_policy_version != EXPECTED_POLICY_VERSION
        ):
            return None
        response_payload = error.response_payload
        if not isinstance(response_payload, dict):
            return None
        detail = response_payload.get("detail")
        if not isinstance(detail, str) or "invalid generated JSON" not in detail:
            return None
        marker = "generated_text="
        marker_index = detail.rfind(marker)
        if marker_index < 0:
            return None
        try:
            generated_text = ast.literal_eval(
                detail[marker_index + len(marker) :].strip()
            )
        except (SyntaxError, ValueError):
            return None
        if not isinstance(generated_text, str):
            return None
        recovered = _parse_recoverable_generated_prefix(generated_text)
        if recovered is None:
            return None
        skill, point_px = recovered
        point_1000 = [point_px[0] / 319.0 * 1000.0, point_px[1] / 239.0 * 1000.0]
        self.invalid_json_prefix_recovery_count += 1
        LOGGER.warning(
            "Recovered original-SFT %s from malformed JSON suffix using validated "
            "skill=%s point_px=%s",
            item["request_id"],
            skill,
            point_px,
        )
        return {
            "model_revision": self.ready_model_revision or "unknown",
            "policy_version": self.expected_policy_version,
            "model_mode": "original_sft",
            "predictions": [
                {
                    "request_id": item["request_id"],
                    "skill": skill,
                    "skill_confidence": None,
                    "skill_probabilities": None,
                    "point_1000": point_1000,
                    "point_px": point_px,
                    "generated_text": generated_text,
                    "recovered_from_invalid_json": True,
                }
            ],
            "timing_ms": {},
        }

    def predict(
        self,
        *,
        task: str,
        front_images: Sequence[Any],
        wrist_images: Sequence[Any],
        state_infos: Sequence[dict[str, Any]],
        step_idx: int,
    ) -> tuple[list[VLMPrediction], dict[str, float]]:
        batch_size = len(front_images)
        if not (batch_size == len(wrist_images) == len(state_infos)):
            raise VLMGuidanceError("VLM batch fields have different lengths")
        items = []
        for env_idx in range(batch_size):
            request_id = f"env{env_idx}-step{step_idx}"
            items.append(
                {
                    "request_id": request_id,
                    "state_info": state_infos[env_idx],
                }
            )

        try:
            payload = self._post_prediction_batch(
                task=task,
                items=items,
                front_images=front_images,
                wrist_images=wrist_images,
            )
        except VLMGuidanceError as error:
            if batch_size == 1:
                recovered_payload = self._recover_original_sft_singleton_422(
                    error, item=items[0]
                )
                if recovered_payload is not None:
                    payload = recovered_payload
                else:
                    raise
            else:
                should_retry_singletons = (
                    self.ready_model_mode == "original_sft"
                    and self.expected_policy_version == EXPECTED_POLICY_VERSION
                    and error.status_code == 422
                )
                if not should_retry_singletons:
                    raise
                self.batch_422_retry_count += 1
                payloads = []
                for env_idx in range(batch_size):
                    try:
                        singleton_payload = self._post_prediction_batch(
                            task=task,
                            items=[items[env_idx]],
                            front_images=[front_images[env_idx]],
                            wrist_images=[wrist_images[env_idx]],
                        )
                    except VLMGuidanceError as singleton_error:
                        singleton_payload = self._recover_original_sft_singleton_422(
                            singleton_error, item=items[env_idx]
                        )
                        if singleton_payload is None:
                            raise
                    payloads.append(singleton_payload)
                first_payload = payloads[0]
                payload = {
                    "model_revision": first_payload.get("model_revision"),
                    "policy_version": first_payload.get("policy_version"),
                    "predictions": [],
                    "timing_ms": {},
                }
                for singleton_payload in payloads:
                    if (
                        singleton_payload.get("model_revision")
                        != payload["model_revision"]
                        or singleton_payload.get("policy_version")
                        != payload["policy_version"]
                    ):
                        raise VLMGuidanceError(
                            "VLM model contract changed across singleton requests"
                        )
                    payload["predictions"].extend(
                        singleton_payload.get("predictions", [])
                    )
                    for key, value in dict(
                        singleton_payload.get("timing_ms", {})
                    ).items():
                        payload["timing_ms"][str(key)] = payload["timing_ms"].get(
                            str(key), 0.0
                        ) + float(value)

        rows = payload.get("predictions")
        if payload.get("policy_version") != self.expected_policy_version:
            raise VLMGuidanceError(
                "VLM prediction policy version mismatch: "
                f"expected {self.expected_policy_version}, got {payload.get('policy_version')}"
            )
        if (
            self.expected_output_schema is not None
            and payload.get("output_schema") != self.expected_output_schema
        ):
            raise VLMGuidanceError(
                "VLM prediction output schema mismatch: "
                f"expected {self.expected_output_schema!r}, "
                f"got {payload.get('output_schema')!r}"
            )
        if not isinstance(rows, list) or len(rows) != batch_size:
            raise VLMGuidanceError("VLM response batch size mismatch")
        by_id = {row.get("request_id"): row for row in rows}
        if set(by_id) != {item["request_id"] for item in items}:
            raise VLMGuidanceError("VLM response request IDs do not match")
        revision = str(payload.get("model_revision", "unknown"))
        if (
            self.ready_model_revision is not None
            and revision != self.ready_model_revision
        ):
            raise VLMGuidanceError(
                "VLM model revision changed after readiness check: "
                f"{self.ready_model_revision} -> {revision}"
            )
        predictions = []
        for item in items:
            row = by_id[item["request_id"]]
            if row.get("valid") is False:
                if self.expected_policy_version != POSE_POLICY_VERSION:
                    raise VLMGuidanceError(
                        "VLM returned an invalid-row sentinel outside the pose contract"
                    )
                parse_error = row.get("parse_error")
                if not isinstance(parse_error, str) or not parse_error:
                    raise VLMGuidanceError("invalid VLM row is missing parse_error")
                predictions.append(
                    VLMPrediction(
                        request_id=item["request_id"],
                        skill=None,
                        skill_confidence=None,
                        skill_probabilities=None,
                        point_1000=None,
                        point_px=None,
                        model_revision=revision,
                        query_step=step_idx,
                        model_output_valid=False,
                        parse_error=parse_error,
                        pose_contract=True,
                        generated_text=(
                            str(row["generated_text"])
                            if row.get("generated_text") is not None
                            else None
                        ),
                    )
                )
                continue
            skill = str(row.get("skill"))
            if skill not in VALID_SKILLS:
                raise VLMGuidanceError(f"VLM returned invalid skill: {skill!r}")
            point_px = np.asarray(row.get("point_px"), dtype=np.float32)
            point_1000 = np.asarray(row.get("point_1000"), dtype=np.float32)
            if point_px.shape != (2,) or point_1000.shape != (2,):
                raise VLMGuidanceError("VLM returned invalid point shape")
            if not np.isfinite(point_px).all() or not np.isfinite(point_1000).all():
                raise VLMGuidanceError("VLM returned non-finite point")
            if not (0 <= point_px[0] <= 319 and 0 <= point_px[1] <= 239):
                raise VLMGuidanceError(f"VLM point is outside front image: {point_px}")
            rotation_6d = None
            rotation_matrix = None
            if self.expected_policy_version == POSE_POLICY_VERSION:
                rotation_6d = np.asarray(row.get("rotation_6d"), dtype=np.float32)
                rotation_matrix = np.asarray(row.get("rotation_matrix"), dtype=np.float32)
                if rotation_6d.shape != (6,) or not np.isfinite(rotation_6d).all():
                    raise VLMGuidanceError("VLM returned invalid Rotation6D")
                if rotation_matrix.shape != (3, 3) or not np.isfinite(rotation_matrix).all():
                    raise VLMGuidanceError("VLM returned invalid rotation matrix")
                orthogonality = rotation_matrix @ rotation_matrix.T
                if not np.allclose(orthogonality, np.eye(3), atol=1e-4):
                    raise VLMGuidanceError("VLM returned a non-orthogonal rotation matrix")
                if not math.isclose(
                    float(np.linalg.det(rotation_matrix)), 1.0, abs_tol=1e-4
                ):
                    raise VLMGuidanceError("VLM returned a non-right-handed rotation matrix")
            raw_probabilities = row.get("skill_probabilities")
            raw_confidence = row.get("skill_confidence")
            if raw_probabilities is None and raw_confidence is None:
                probabilities = None
                confidence = None
            else:
                probabilities = {
                    str(key): float(value)
                    for key, value in dict(raw_probabilities or {}).items()
                }
                if set(probabilities) != VALID_SKILLS or not all(
                    math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in probabilities.values()
                ):
                    raise VLMGuidanceError("VLM returned invalid skill probabilities")
                if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
                    raise VLMGuidanceError("VLM skill probabilities do not sum to one")
                confidence = float(raw_confidence)
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise VLMGuidanceError("VLM returned invalid skill confidence")
            predictions.append(
                VLMPrediction(
                    request_id=item["request_id"],
                    skill=skill,
                    skill_confidence=confidence,
                    skill_probabilities=probabilities,
                    point_1000=point_1000,
                    point_px=point_px,
                    model_revision=revision,
                    query_step=step_idx,
                    model_output_valid=True,
                    parse_error=(
                        str(row["parse_error"])
                        if row.get("parse_error") is not None
                        else None
                    ),
                    pose_contract=self.expected_policy_version == POSE_POLICY_VERSION,
                    generated_text=(
                        str(row["generated_text"])
                        if row.get("generated_text") is not None
                        else None
                    ),
                    recovered_from_invalid_json=bool(
                        row.get("recovered_from_invalid_json", False)
                    ),
                    rotation_6d=rotation_6d,
                    rotation_matrix=rotation_matrix,
                )
            )
        timing = {
            str(key): float(value)
            for key, value in dict(payload.get("timing_ms", {})).items()
        }
        return predictions, timing

    def transport_stats(self) -> dict[str, int | str | None]:
        return {
            "model_mode": self.ready_model_mode,
            "prediction_http_request_count": self.prediction_http_request_count,
            "batch_prediction_request_count": self.batch_prediction_request_count,
            "singleton_prediction_request_count": self.singleton_prediction_request_count,
            "prediction_transport_retry_count": self.prediction_transport_retry_count,
            "batch_422_retry_count": self.batch_422_retry_count,
            "invalid_json_prefix_recovery_count": self.invalid_json_prefix_recovery_count,
        }


def _front_depth_median(
    depth_image: Any,
    point_px: np.ndarray,
    *,
    window_radius: int = 2,
    min_depth_m: float = 0.10,
    max_depth_m: float = 1.50,
) -> tuple[float | None, int]:
    if torch is not None and torch.is_tensor(depth_image):
        depth_image = depth_image.detach().cpu().numpy()
    # Isaac Gym camera tensors encode forward depth as negative values; saved
    # rollout depth is normalized with the same absolute-value convention.
    depth = np.abs(np.asarray(depth_image, dtype=np.float32).squeeze())
    if depth.ndim != 2:
        raise VLMGuidanceError(f"expected HxW front depth, got {depth.shape}")
    x = int(round(float(point_px[0])))
    y = int(round(float(point_px[1])))
    x0 = max(0, x - window_radius)
    x1 = min(depth.shape[1], x + window_radius + 1)
    y0 = max(0, y - window_radius)
    y1 = min(depth.shape[0], y + window_radius + 1)
    values = depth[y0:y1, x0:x1]
    valid = values[
        np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)
    ]
    if not valid.size:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def _backproject_front_pixel(
    point_px: np.ndarray,
    depth_m: float,
    camera_info: Mapping[str, Any],
) -> np.ndarray:
    intrinsics = np.asarray(camera_info.get("intrinsics"), dtype=np.float64)
    camera_to_robot_base = np.asarray(
        camera_info.get("camera_to_robot_base"), dtype=np.float64
    )
    if intrinsics.shape != (3, 3) or camera_to_robot_base.shape != (4, 4):
        raise VLMGuidanceError("front camera calibration is incomplete")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if not all(math.isfinite(value) and abs(value) > 1e-8 for value in (fx, fy)):
        raise VLMGuidanceError("front camera focal length is invalid")
    x_camera = (float(point_px[0]) - cx) * depth_m / fx
    y_cv = (float(point_px[1]) - cy) * depth_m / fy
    point_camera = np.array([x_camera, -y_cv, depth_m, 1.0], dtype=np.float64)
    point_robot_base = camera_to_robot_base @ point_camera
    if not np.isfinite(point_robot_base).all() or abs(point_robot_base[3]) < 1e-8:
        raise VLMGuidanceError("front pixel backprojection is non-finite")
    return (point_robot_base[:3] / point_robot_base[3]).astype(np.float32)


def materialize_pose_predictions(
    oracle_bundles: Sequence[Mapping[str, Any]],
    predictions: Sequence[VLMPrediction],
    front_depth_images: Sequence[Any],
) -> list[VLMPrediction]:
    """Build pose/grasp geometry using only VLM output, depth, and calibration."""

    if not (len(oracle_bundles) == len(predictions) == len(front_depth_images)):
        raise VLMGuidanceError("pose materialization batch size mismatch")
    from src.eval.skill_annotation_util import (
        DEFAULT_GRASP_HEIGHT_M,
        DEFAULT_GRASP_WIDTH_M,
        project_pose_to_grasp_annotation_2d,
    )

    output = []
    for calibration_bundle, prediction, depth_image in zip(
        oracle_bundles, predictions, front_depth_images
    ):
        if not prediction.model_output_valid:
            output.append(
                replace(
                    prediction,
                    guidance_pose=None,
                    grasp_annotation_2d=None,
                    sampled_depth_m=None,
                    depth_valid_count=0,
                    grasp_projection_valid=False,
                )
            )
            continue
        if prediction.rotation_matrix is None:
            raise VLMGuidanceError("pose policy prediction is missing Rotation6D")
        camera_info = calibration_bundle.get("camera_info", {}).get("color_image2")
        if not isinstance(camera_info, Mapping):
            raise VLMGuidanceError("front camera calibration is missing")
        depth_m, valid_count = _front_depth_median(depth_image, prediction.point_px)
        pose = None
        grasp = None
        projection_valid: bool | None = None
        if depth_m is not None:
            center = _backproject_front_pixel(prediction.point_px, depth_m, camera_info)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = prediction.rotation_matrix
            pose[:3, 3] = center
            if prediction.skill in {"pick", "place"}:
                grasp = project_pose_to_grasp_annotation_2d(
                    pose,
                    dict(camera_info),
                    gripper_width=DEFAULT_GRASP_WIDTH_M,
                    grasp_height=DEFAULT_GRASP_HEIGHT_M,
                )
                projection_valid = grasp is not None
        elif prediction.skill in {"pick", "place"}:
            projection_valid = False
        output.append(
            replace(
                prediction,
                guidance_pose=pose,
                grasp_annotation_2d=grasp,
                sampled_depth_m=depth_m,
                depth_valid_count=valid_count,
                grasp_projection_valid=projection_valid,
            )
        )
    return output


def policy_bundles_from_vlm(
    oracle_bundles: Sequence[dict[str, Any]],
    predictions: Sequence[VLMPrediction],
    *,
    step_idx: int,
) -> list[dict[str, Any]]:
    if len(oracle_bundles) != len(predictions):
        raise VLMGuidanceError("oracle/VLM batch size mismatch")
    output = []
    for oracle, prediction in zip(oracle_bundles, predictions):
        bundle = dict(oracle)
        bundle["oracle_skill"] = oracle.get("skill")
        bundle["oracle_guidance_point_2d"] = dict(
            oracle.get("guidance_point_2d", {})
        )
        bundle["skill"] = prediction.skill
        bundle["guidance_point_2d"] = {
            "color_image2": (
                prediction.point_px.copy() if prediction.point_px is not None else None
            )
        }
        if prediction.pose_contract:
            # These are the only geometry fields consumed by grasp-part
            # rendering. Overwrite them unconditionally for pose-policy rows so
            # invalid depth/projection becomes blank, never scripted fallback.
            bundle["guidance_point"] = (
                prediction.guidance_pose[:3, 3].copy()
                if prediction.guidance_pose is not None
                else None
            )
            bundle["guidance_pose"] = (
                prediction.guidance_pose.copy()
                if prediction.guidance_pose is not None
                else None
            )
            bundle["guidance_gripper_width"] = (
                0.05 if prediction.guidance_pose is not None else None
            )
            bundle["grasp_annotation_2d"] = {
                "color_image2": (
                    {
                        key: value.copy() if isinstance(value, np.ndarray) else value
                        for key, value in prediction.grasp_annotation_2d.items()
                    }
                    if prediction.grasp_annotation_2d is not None
                    else None
                )
            }
        bundle["vlm_annotation"] = {
            "request_id": prediction.request_id,
            "model_revision": prediction.model_revision,
            "skill_confidence": prediction.skill_confidence,
            "skill_probabilities": prediction.skill_probabilities,
            "model_output_valid": prediction.model_output_valid,
            "parse_error": prediction.parse_error,
            "point_1000": (
                prediction.point_1000.copy()
                if prediction.point_1000 is not None
                else None
            ),
            "point_px": (
                prediction.point_px.copy()
                if prediction.point_px is not None
                else None
            ),
            "rotation_6d": (
                prediction.rotation_6d.copy()
                if prediction.rotation_6d is not None
                else None
            ),
            "rotation_matrix": (
                prediction.rotation_matrix.copy()
                if prediction.rotation_matrix is not None
                else None
            ),
            "guidance_pose": (
                prediction.guidance_pose.copy()
                if prediction.guidance_pose is not None
                else None
            ),
            "sampled_depth_m": prediction.sampled_depth_m,
            "depth_valid_count": prediction.depth_valid_count,
            "grasp_projection_valid": prediction.grasp_projection_valid,
            "grasp_annotation_2d": (
                {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in prediction.grasp_annotation_2d.items()
                }
                if prediction.grasp_annotation_2d is not None
                else None
            ),
            "depth_window": "5x5_median_abs_valid_0.10_1.50m",
            "fixed_grasp_width_m": 0.05,
            "fixed_grasp_height_m": 0.02,
            "generated_text": prediction.generated_text,
            "recovered_from_invalid_json": prediction.recovered_from_invalid_json,
            "query_step": prediction.query_step,
            "cache_age_steps": step_idx - prediction.query_step,
        }
        output.append(bundle)
    return output
