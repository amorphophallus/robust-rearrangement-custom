"""Lightweight parsing helpers for native/original-SFT generations."""

from __future__ import annotations

import json
import math
from typing import Sequence

from services.vlm_guidance import SKILL_NAMES


def configure_native_processor(processor) -> None:
    """Configure decoder-only batching to match per-sample reference generation."""

    processor.tokenizer.padding_side = "left"


def try_parse_native_prediction(
    text: str,
) -> tuple[str, list[float] | None, str | None]:
    """Parse a generation while retaining a valid skill on point failures."""

    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        start = cleaned.find("{")
        if start < 0:
            return "invalid", None, "generated text contains no JSON object"
        try:
            payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as error:
            return "invalid", None, f"invalid generated JSON: {error}"
    if not isinstance(payload, dict):
        return "invalid", None, "generated JSON is not an object"
    skill = payload.get("skill")
    if skill not in SKILL_NAMES:
        return "invalid", None, f"unsupported generated skill: {skill!r}"
    point = payload.get("target_point_2d")
    if not isinstance(point, list) or len(point) != 2:
        return str(skill), None, "target_point_2d is not a two-value list"
    try:
        point_px = [float(point[0]), float(point[1])]
    except (TypeError, ValueError):
        return str(skill), None, "target_point_2d contains non-numeric values"
    if not all(math.isfinite(value) for value in point_px):
        return str(skill), None, "target_point_2d contains non-finite values"
    if not (0.0 <= point_px[0] <= 319.0 and 0.0 <= point_px[1] <= 239.0):
        return str(skill), point_px, f"target_point_2d is outside the front image: {point_px}"
    return str(skill), point_px, None


def parse_native_prediction(text: str) -> tuple[str, list[float]]:
    """Strict production parser for one original-SFT JSON generation."""

    skill, point_px, error = try_parse_native_prediction(text)
    if error is not None or point_px is None:
        raise ValueError(error or "missing target_point_2d")
    return skill, point_px


def pixels_to_qwen(point_px: Sequence[float]) -> list[float]:
    return [
        float(point_px[0]) / 319.0 * 1000.0,
        float(point_px[1]) / 239.0 * 1000.0,
    ]
