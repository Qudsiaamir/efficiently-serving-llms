"""Simple uint8 affine quantization helpers for educational experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from efficient_llm_serving.dependencies import require_torch


@dataclass(frozen=True)
class QuantizationState:
    """Scale and zero point needed to reconstruct a quantized tensor."""

    scale: Any
    zero_point: Any


def quantize_tensor(tensor: Any) -> tuple[Any, QuantizationState]:
    """Quantize a tensor to uint8 and return dequantization state."""

    torch = require_torch()
    min_val = tensor.min()
    max_val = tensor.max()
    value_range = max_val - min_val
    scale = (
        value_range / 255
        if value_range.item() != 0
        else torch.tensor(1.0, device=tensor.device)
    )
    zero_point = min_val

    quantized = torch.clamp((tensor - zero_point) / scale, min=0, max=255).to(torch.uint8)
    return quantized, QuantizationState(scale=scale, zero_point=zero_point)


def dequantize_tensor(tensor: Any, state: QuantizationState) -> Any:
    """Restore a quantized uint8 tensor to float32."""

    require_torch()
    return tensor.to(dtype=state.scale.dtype) * state.scale + state.zero_point


def tensor_size_bytes(tensor: Any) -> int:
    """Return tensor memory footprint in bytes."""

    return int(tensor.numel() * tensor.element_size())


def quantize_model_parameters(model: Any) -> tuple[Any, dict[str, QuantizationState]]:
    """Quantize model parameters in place.

    This mirrors the notebooks' educational example. It is not a replacement
    for production quantization backends such as bitsandbytes, AWQ, or GPTQ.
    """

    states: dict[str, QuantizationState] = {}
    for name, param in model.named_parameters():
        param.requires_grad = False
        param.data, states[name] = quantize_tensor(param.data)
    return model, states


def dequantize_model_parameters(model: Any, states: dict[str, QuantizationState]) -> Any:
    """Dequantize model parameters in place using saved states."""

    for name, param in model.named_parameters():
        param.data = dequantize_tensor(param.data, states[name])
    return model
