"""Model loading and batched structured/native-SFT inference."""

from __future__ import annotations

import json
import hashlib
import math
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from services.vlm_guidance import (
    ORIGINAL_SFT_POLICY_VERSION,
    POINT_POLICY_VERSION,
    SKILL_NAMES,
)
from services.vlm_guidance.modeling import (
    FurniturePolicyModel,
    apply_torch29_qwen35_conv3d_patch,
)
from services.vlm_guidance.native_sft import (
    configure_native_processor,
    parse_native_prediction,
    pixels_to_qwen,
    try_parse_native_prediction,
)
from src.vlm_data_generator import DEFAULT_USER_PROMPT, TASK_SYSTEM_PROMPTS


def _load_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")
    pytorch_path = checkpoint_dir / "pytorch_model.bin"
    if pytorch_path.is_file():
        return torch.load(pytorch_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no model weights found under {checkpoint_dir}")


def _interleave_images(user_text: str, images: Sequence[Image.Image]):
    chunks = user_text.split("<image>")
    if len(chunks) - 1 != len(images):
        raise ValueError("prompt/image count mismatch")
    content = []
    for index, chunk in enumerate(chunks):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if index < len(images):
            content.append({"type": "image", "image": images[index]})
    return content


class FurnitureInferenceEngine:
    def __init__(
        self,
        *,
        base_model_dir: str | None,
        checkpoint_dir: str,
        model_mode: str = "auto",
        device: str = "cuda:0",
        attention_backend: str = "sdpa",
        max_length: int = 4096,
        image_max_pixels: int = 262144,
        max_micro_batch_size: int = 8,
        max_new_tokens: int = 256,
        allow_invalid_predictions: bool = False,
        model_revision: str = "unknown",
        manifest_path: str | None = None,
    ) -> None:
        self.base_model_dir = Path(base_model_dir) if base_model_dir else None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.requested_model_mode = model_mode
        self.device = torch.device(device)
        self.attention_backend = attention_backend
        self.max_length = int(max_length)
        self.image_max_pixels = int(image_max_pixels)
        self.max_micro_batch_size = int(max_micro_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.allow_invalid_predictions = bool(allow_invalid_predictions)
        self.model_revision = model_revision
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self._lock = threading.Lock()
        self._load()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_manifest(self) -> None:
        if self.manifest_path is None:
            return
        with self.manifest_path.open() as stream:
            manifest = json.load(stream)
        roots = {"checkpoint": self.checkpoint_dir}
        if self.base_model_dir is not None:
            roots["base_model"] = self.base_model_dir
        for relative, expected in manifest.get("files", {}).items():
            root_name, file_name = relative.split("/", 1)
            if root_name not in roots:
                raise RuntimeError(f"manifest requires unavailable artifact root: {root_name}")
            path = roots[root_name] / file_name
            if path.stat().st_size != int(expected["size"]):
                raise RuntimeError(f"model artifact size mismatch: {path}")
            if self._sha256(path) != expected["sha256"]:
                raise RuntimeError(f"model artifact checksum mismatch: {path}")
        self.model_revision = str(
            manifest.get("checkpoint_revision", self.model_revision)
        )
        manifest_mode = manifest.get("model_mode")
        if manifest_mode is not None and self.requested_model_mode not in {"auto", manifest_mode}:
            raise RuntimeError(
                f"manifest model mode {manifest_mode!r} conflicts with "
                f"requested {self.requested_model_mode!r}"
            )

    def _load(self) -> None:
        self._verify_manifest()
        if not torch.cuda.is_available() and self.device.type == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
        apply_torch29_qwen35_conv3d_patch()
        config_path = self.checkpoint_dir / "config.json"
        with config_path.open() as stream:
            checkpoint_config = json.load(stream)
        policy = checkpoint_config.get("hy_furniture_policy", {})
        detected_mode = "structured" if policy else "original_sft"
        if self.requested_model_mode not in {"auto", "structured", "original_sft"}:
            raise ValueError(f"unsupported model mode: {self.requested_model_mode}")
        self.model_mode = (
            detected_mode if self.requested_model_mode == "auto" else self.requested_model_mode
        )
        if self.model_mode != detected_mode:
            raise RuntimeError(
                f"checkpoint looks like {detected_mode}, not requested {self.model_mode}"
            )
        if self.model_mode == "structured":
            if policy.get("version") != POINT_POLICY_VERSION:
                raise RuntimeError(f"unsupported point policy: {policy}")
            if tuple(policy.get("skill_names", ())) != SKILL_NAMES:
                raise RuntimeError("checkpoint skill order does not match service")
            if self.base_model_dir is None:
                raise RuntimeError("structured mode requires a base model directory")
            self.policy_version = POINT_POLICY_VERSION
        else:
            architectures = tuple(checkpoint_config.get("architectures", ()))
            if "Qwen3_5ForConditionalGeneration" not in architectures:
                raise RuntimeError(
                    "original_sft checkpoint is not Qwen3_5ForConditionalGeneration: "
                    f"{architectures}"
                )
            self.policy_version = ORIGINAL_SFT_POLICY_VERSION

        self.processor = AutoProcessor.from_pretrained(
            str(self.checkpoint_dir), trust_remote_code=True
        )
        if self.model_mode == "original_sft":
            # The reference visualizer runs one sample at a time and therefore
            # never pads. API batches must left-pad decoder-only generation;
            # right padding can make generate continue from pad tokens and emit
            # an empty or malformed assistant response.
            configure_native_processor(self.processor)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = self.image_max_pixels
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        attention_backend = self.attention_backend
        if self.device.type != "cuda" and attention_backend == "flash_attention_2":
            attention_backend = "eager"
        if self.model_mode == "original_sft":
            load_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if attention_backend not in {None, "auto", "disabled"}:
                load_kwargs["attn_implementation"] = attention_backend
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(self.checkpoint_dir), **load_kwargs
            )
        else:
            assert self.base_model_dir is not None
            self.model = FurniturePolicyModel.from_qwen_pretrained(
                str(self.base_model_dir),
                torch_dtype=dtype,
                attn_implementation=attention_backend,
            )
            state_dict = _load_state_dict(self.checkpoint_dir)
            self.model.load_state_dict(state_dict, strict=True)
            del state_dict
        self.model.to(self.device).eval()

    def _collate(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        batch_images = []
        for sample in samples:
            task = str(sample["task"])
            if task not in TASK_SYSTEM_PROMPTS:
                raise ValueError(f"unsupported task: {task}")
            base_state = sample["state_info"]["base"]
            compact_state = json.dumps(
                {"base": base_state}, ensure_ascii=False, separators=(",", ":")
            )
            user_text = DEFAULT_USER_PROMPT.replace("<state_info>", compact_state)
            images = [sample["front"].convert("RGB"), sample["wrist"].convert("RGB")]
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": TASK_SYSTEM_PROMPTS[task]}],
                },
                {"role": "user", "content": _interleave_images(user_text, images)},
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            batch_images.append(images)
        inputs = self.processor(
            text=texts,
            images=batch_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _predict_micro_batch(self, samples: Sequence[dict[str, Any]]):
        preprocess_started = time.perf_counter()
        inputs = self._collate(samples)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        forward_started = time.perf_counter()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            if self.model_mode == "original_sft":
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
                input_length = inputs["input_ids"].shape[1]
                generated_texts = self.processor.batch_decode(
                    generated_ids[:, input_length:], skip_special_tokens=True
                )
                outputs = None
            else:
                outputs = self.model(**inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        predictions = []
        if self.model_mode == "original_sft":
            for sample, generated_text in zip(samples, generated_texts):
                if self.allow_invalid_predictions:
                    skill, point_px, parse_error = try_parse_native_prediction(
                        generated_text
                    )
                else:
                    try:
                        skill, point_px = parse_native_prediction(generated_text)
                    except ValueError as error:
                        raise ValueError(
                            f"{sample['request_id']}: {error}; "
                            f"generated_text={generated_text!r}"
                        ) from error
                    parse_error = None
                predictions.append(
                    {
                        "request_id": sample["request_id"],
                        "skill": skill,
                        "skill_confidence": None,
                        "skill_probabilities": None,
                        "point_1000": pixels_to_qwen(point_px) if point_px is not None else None,
                        "point_px": point_px,
                        "generated_text": generated_text,
                        "parse_error": parse_error,
                    }
                )
        else:
            assert outputs is not None
            probabilities = outputs["skill_logits"].float().softmax(dim=-1).cpu()
            points_1000 = outputs["point_predictions"].float().cpu()
            points_px = outputs["point_predictions_px"].float().cpu()
            for index, sample in enumerate(samples):
                skill_id = int(probabilities[index].argmax())
                point_1000 = [float(value) for value in points_1000[index]]
                point_px = [float(value) for value in points_px[index]]
                if not all(math.isfinite(value) for value in (*point_1000, *point_px)):
                    raise RuntimeError("model returned non-finite point")
                if not (0.0 <= point_px[0] <= 319.0 and 0.0 <= point_px[1] <= 239.0):
                    raise RuntimeError(f"model returned out-of-range point: {point_px}")
                predictions.append(
                    {
                        "request_id": sample["request_id"],
                        "skill": SKILL_NAMES[skill_id],
                        "skill_confidence": float(probabilities[index, skill_id]),
                        "skill_probabilities": {
                            name: float(probabilities[index, idx])
                            for idx, name in enumerate(SKILL_NAMES)
                        },
                        "point_1000": point_1000,
                        "point_px": point_px,
                    }
                )
        return predictions, preprocess_ms, forward_ms

    def predict_batch(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("empty prediction batch")
        total_started = time.perf_counter()
        predictions = []
        preprocess_ms = 0.0
        forward_ms = 0.0
        with self._lock:
            queue_ms = (time.perf_counter() - total_started) * 1000.0
            for start in range(0, len(samples), self.max_micro_batch_size):
                chunk = samples[start : start + self.max_micro_batch_size]
                chunk_predictions, chunk_preprocess, chunk_forward = (
                    self._predict_micro_batch(chunk)
                )
                predictions.extend(chunk_predictions)
                preprocess_ms += chunk_preprocess
                forward_ms += chunk_forward
        return {
            "model_revision": self.model_revision,
            "policy_version": self.policy_version,
            "model_mode": self.model_mode,
            "predictions": predictions,
            "timing_ms": {
                "queue": queue_ms,
                "preprocess": preprocess_ms,
                "forward": forward_ms,
                "total": (time.perf_counter() - total_started) * 1000.0,
            },
        }

    def warmup(self) -> None:
        blank = Image.new("RGB", (320, 240), color=(0, 0, 0))
        base = {
            "ee_pos_sim": [0.0, 0.0, 0.0],
            "ee_quat_sim": [0.0, 0.0, 0.0, 1.0],
            "ee_pos_vel": [0.0, 0.0, 0.0],
            "ee_ori_vel": [0.0, 0.0, 0.0],
            "gripper_width": 0.0,
        }
        self.predict_batch(
            [
                {
                    "request_id": "warmup",
                    "task": "one_leg",
                    "state_info": {"base": base},
                    "front": blank,
                    "wrist": blank,
                }
            ]
        )
