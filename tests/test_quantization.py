import pytest

from efficient_llm_serving.dependencies import require_torch


torch = pytest.importorskip("torch")

from efficient_llm_serving.quantization import dequantize_tensor, quantize_tensor, tensor_size_bytes


def test_quantize_dequantize_tensor_round_trip_shape_and_dtype():
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0])

    quantized, state = quantize_tensor(tensor)
    restored = dequantize_tensor(quantized, state)

    assert quantized.dtype == torch.uint8
    assert restored.shape == tensor.shape
    assert tensor_size_bytes(quantized) < tensor_size_bytes(tensor)


def test_require_torch_returns_module_when_installed():
    assert require_torch().__name__ == "torch"
