"""Structured Qwen model used by the HY furniture guidance checkpoint."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModelForImageTextToText, PreTrainedModel

from services.vlm_guidance import POINT_POLICY_VERSION, SKILL_NAMES


QWEN_COORDINATE_MAX = 1000.0
FRONT_PIXEL_MAX_X = 319.0
FRONT_PIXEL_MAX_Y = 239.0
def qwen_to_front_pixels(points_1000: torch.Tensor) -> torch.Tensor:
    pixel_max = points_1000.new_tensor((FRONT_PIXEL_MAX_X, FRONT_PIXEL_MAX_Y))
    return points_1000 * (pixel_max / QWEN_COORDINATE_MAX)


def _hidden_size(config: Any) -> int:
    text_config = getattr(config, "text_config", config)
    for name in ("hidden_size", "d_model"):
        value = getattr(text_config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("cannot determine the Qwen text hidden size")


class FurniturePolicyModel(PreTrainedModel):
    """Qwen backbone with skill-classification and 2-D point heads."""

    base_model_prefix = "backbone"
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True

    def __init__(self, config: Any, backbone: nn.Module) -> None:
        super().__init__(config)
        self.backbone = backbone
        hidden_size = _hidden_size(config)
        self.skill_head = nn.Linear(hidden_size, len(SKILL_NAMES))
        self.point_head = nn.Linear(hidden_size, 2)

    @classmethod
    def from_qwen_pretrained(
        cls,
        model_path: str,
        *,
        torch_dtype: torch.dtype,
        attn_implementation: str | None,
    ) -> "FurniturePolicyModel":
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if attn_implementation and attn_implementation not in {"auto", "disabled"}:
            kwargs["attn_implementation"] = attn_implementation
        conditional_model = AutoModelForImageTextToText.from_pretrained(
            model_path, **kwargs
        )
        model = cls(conditional_model.config, conditional_model.model)
        model.skill_head.to(dtype=torch_dtype)
        model.point_head.to(dtype=torch_dtype)
        del conditional_model
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_inputs: Any,
    ) -> dict[str, torch.Tensor]:
        model_inputs.pop("labels", None)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **model_inputs,
        )
        hidden_states = outputs.last_hidden_state
        token_positions = torch.arange(
            attention_mask.shape[1], device=attention_mask.device
        ).unsqueeze(0)
        last_positions = (token_positions * attention_mask.long()).argmax(dim=1)
        pooled = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
            last_positions,
        ].to(dtype=self.skill_head.weight.dtype)
        skill_logits = self.skill_head(pooled)
        point_predictions = (
            torch.sigmoid(self.point_head(pooled).float()) * QWEN_COORDINATE_MAX
        )
        return {
            "skill_logits": skill_logits,
            "point_predictions": point_predictions,
            "point_predictions_px": qwen_to_front_pixels(point_predictions),
        }


def apply_torch29_qwen35_conv3d_patch() -> bool:
    """Apply the checkpoint author's Conv3d workaround on PyTorch 2.9."""

    import torch.nn.functional as F

    version = tuple(
        int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
    )
    if version != (2, 9):
        return False

    class LinearizedConv3d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.weight = nn.Parameter(
                torch.empty(out_channels, in_channels, *kernel_size)
            )
            self.bias = nn.Parameter(torch.empty(out_channels))

        def forward(self, values):
            kt, kh, kw = self.kernel_size
            values = values.unfold(2, kt, kt).unfold(3, kh, kh).unfold(4, kw, kw)
            batch, _, time, height, width = values.shape[:5]
            values = values.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
                batch, time, height, width, -1
            )
            values = F.linear(
                values,
                self.weight.reshape(self.out_channels, -1),
                self.bias,
            )
            return values.permute(0, 4, 1, 2, 3)

    class Conv3dPatchEmbed(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.patch_size = config.patch_size
            self.temporal_patch_size = config.temporal_patch_size
            self.in_channels = config.in_channels
            self.embed_dim = config.hidden_size
            kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
            self.proj = LinearizedConv3d(self.in_channels, self.embed_dim, kernel)

        def forward(self, hidden_states):
            kt, kh, kw = self.proj.kernel_size
            values = hidden_states.view(-1, self.in_channels, kt, kh, kw)
            return self.proj(values.to(dtype=self.proj.weight.dtype)).view(
                -1, self.embed_dim
            )

    import transformers

    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionPatchEmbed = (
        Conv3dPatchEmbed
    )
    transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeVisionPatchEmbed = (
        Conv3dPatchEmbed
    )
    return True
