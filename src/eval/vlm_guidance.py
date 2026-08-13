"""HTTP client and rollout helpers for remote VLM annotations."""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
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
EXPECTED_POLICY_VERSION = 2


class VLMGuidanceError(RuntimeError):
    """Raised when remote annotations cannot safely be used."""


@dataclass(frozen=True)
class VLMPrediction:
    request_id: str
    skill: str
    skill_confidence: float
    skill_probabilities: dict[str, float]
    point_1000: np.ndarray
    point_px: np.ndarray
    model_revision: str
    query_step: int


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.api_token = api_token if api_token is not None else os.getenv("VLM_API_TOKEN")
        self.session = session or requests.Session()
        self.ready_model_revision: str | None = None

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
        if payload.get("policy_version") != EXPECTED_POLICY_VERSION:
            raise VLMGuidanceError(
                "VLM point-policy version mismatch: "
                f"expected {EXPECTED_POLICY_VERSION}, got {payload.get('policy_version')}"
            )
        self.ready_model_revision = str(payload.get("model_revision", "unknown"))
        return payload

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
        files: dict[str, tuple[str, bytes, str]] = {}
        for env_idx in range(batch_size):
            request_id = f"env{env_idx}-step{step_idx}"
            items.append(
                {
                    "request_id": request_id,
                    "state_info": state_infos[env_idx],
                }
            )
            files[f"front_{env_idx}"] = (
                f"front_{env_idx}.png",
                _png_bytes(front_images[env_idx]),
                "image/png",
            )
            files[f"wrist_{env_idx}"] = (
                f"wrist_{env_idx}.png",
                _png_bytes(wrist_images[env_idx]),
                "image/png",
            )
        files["metadata"] = (
            None,
            json.dumps({"task": task, "items": items}, separators=(",", ":")),
            "application/json",
        )
        try:
            response = self.session.post(
                f"{self.base_url}/v1/guidance/predict",
                files=files,
                headers=self.headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise VLMGuidanceError(f"VLM prediction request failed: {exc}") from exc

        rows = payload.get("predictions")
        if payload.get("policy_version") != EXPECTED_POLICY_VERSION:
            raise VLMGuidanceError(
                "VLM prediction point-policy version mismatch: "
                f"expected {EXPECTED_POLICY_VERSION}, got {payload.get('policy_version')}"
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
            probabilities = {
                str(key): float(value)
                for key, value in dict(row.get("skill_probabilities", {})).items()
            }
            if set(probabilities) != VALID_SKILLS or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in probabilities.values()
            ):
                raise VLMGuidanceError("VLM returned invalid skill probabilities")
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
                raise VLMGuidanceError("VLM skill probabilities do not sum to one")
            confidence = float(row.get("skill_confidence", float("nan")))
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
                )
            )
        timing = {
            str(key): float(value)
            for key, value in dict(payload.get("timing_ms", {})).items()
        }
        return predictions, timing


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
        bundle["guidance_point_2d"] = {"color_image2": prediction.point_px.copy()}
        bundle["vlm_annotation"] = {
            "request_id": prediction.request_id,
            "model_revision": prediction.model_revision,
            "skill_confidence": prediction.skill_confidence,
            "skill_probabilities": prediction.skill_probabilities,
            "point_1000": prediction.point_1000.copy(),
            "query_step": prediction.query_step,
            "cache_age_steps": step_idx - prediction.query_step,
        }
        output.append(bundle)
    return output
