"""FastAPI application for remote VLM guidance inference."""

from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

from services.vlm_guidance.engine import FurnitureInferenceEngine


def _engine_from_env() -> FurnitureInferenceEngine:
    return FurnitureInferenceEngine(
        base_model_dir=os.getenv("VLM_BASE_MODEL_DIR"),
        checkpoint_dir=os.environ["VLM_CHECKPOINT_DIR"],
        model_mode=os.getenv("VLM_MODEL_MODE", "auto"),
        device=os.getenv("VLM_DEVICE", "cuda:0"),
        attention_backend=os.getenv("VLM_ATTENTION_BACKEND", "sdpa"),
        max_length=int(os.getenv("VLM_MAX_LENGTH", "4096")),
        image_max_pixels=int(os.getenv("VLM_IMAGE_MAX_PIXELS", "262144")),
        max_micro_batch_size=int(os.getenv("VLM_MAX_MICRO_BATCH_SIZE", "8")),
        max_new_tokens=int(os.getenv("VLM_MAX_NEW_TOKENS", "256")),
        model_revision=os.getenv("VLM_MODEL_REVISION", "unknown"),
        manifest_path=os.getenv("VLM_MANIFEST_PATH"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.engine = _engine_from_env()
    app.state.engine.warmup()
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(title="HY Furniture VLM Guidance", lifespan=lifespan)


def _authorize(request: Request) -> None:
    expected = os.getenv("VLM_API_TOKEN")
    if not expected:
        return
    if request.headers.get("authorization") != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health/live")
def live():
    return {"status": "live"}


@app.get("/health/ready")
def ready(request: Request):
    _authorize(request)
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="model is not ready")
    engine = request.app.state.engine
    return {
        "status": "ready",
        "model_revision": engine.model_revision,
        "policy_version": engine.policy_version,
        "model_mode": engine.model_mode,
        "device": str(engine.device),
        "attention_backend": engine.attention_backend,
    }


@app.post("/v1/guidance/predict")
async def predict(request: Request):
    _authorize(request)
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="model is not ready")
    try:
        form = await request.form()
        metadata = json.loads(str(form["metadata"]))
        task = metadata["task"]
        items = metadata["items"]
        request_ids = [item["request_id"] for item in items]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate request_id")
        samples = []
        for index, item in enumerate(items):
            front_upload = form[f"front_{index}"]
            wrist_upload = form[f"wrist_{index}"]
            front = Image.open(io.BytesIO(await front_upload.read())).convert("RGB")
            wrist = Image.open(io.BytesIO(await wrist_upload.read())).convert("RGB")
            if front.size != (320, 240):
                raise ValueError(f"front_{index} must be 320x240, got {front.size}")
            if wrist.size != (320, 240):
                raise ValueError(f"wrist_{index} must be 320x240, got {wrist.size}")
            samples.append(
                {
                    "request_id": item["request_id"],
                    "task": task,
                    "state_info": item["state_info"],
                    "front": front,
                    "wrist": wrist,
                }
            )
        return request.app.state.engine.predict_batch(samples)
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[name-defined]
        request.app.state.ready = False
        raise HTTPException(status_code=503, detail="CUDA out of memory") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"inference failed: {exc}") from exc
