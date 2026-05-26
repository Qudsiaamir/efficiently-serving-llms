"""LoRA and multi-LoRA model components extracted from the notebooks."""

from __future__ import annotations

from typing import Any

from efficient_llm_serving.dependencies import require_torch


DETOKENIZER = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "violet",
    "black",
    "white",
    "gray",
]


def build_test_model(hidden_size: int = 10) -> Any:
    """Create the toy model used by the LoRA notebooks."""

    torch = require_torch()

    class TestModel(torch.nn.Module):
        def __init__(self, size: int) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(10, size)
            self.linear = torch.nn.Linear(size, size)
            self.lm_head = torch.nn.Linear(size, 10)

        def forward(self, input_ids: Any) -> Any:
            x = self.embedding(input_ids)
            x = self.linear(x)
            return self.lm_head(x)

    return TestModel(hidden_size)


def build_lora_layer(base_layer: Any, rank: int) -> Any:
    """Wrap a linear layer with a trainable low-rank update."""

    if rank <= 0:
        raise ValueError("rank must be positive")

    torch = require_torch()

    class LoraLayer(torch.nn.Module):
        def __init__(self, wrapped_layer: Any, r: int) -> None:
            super().__init__()
            self.base_layer = wrapped_layer
            in_features = wrapped_layer.in_features
            out_features = wrapped_layer.out_features
            self.lora_a = torch.nn.Parameter(torch.randn(in_features, r))
            self.lora_b = torch.nn.Parameter(torch.randn(r, out_features))

        def forward(self, x: Any) -> Any:
            return self.base_layer(x) + x @ self.lora_a @ self.lora_b

    return LoraLayer(base_layer, rank)


def generate_toy_token(
    model: Any,
    detokenizer: list[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    """Generate one greedy token from a toy model and map ids to strings."""

    torch = require_torch()
    labels = detokenizer or DETOKENIZER
    with torch.no_grad():
        logits = model(**kwargs)
    next_token_ids = logits[:, -1, :].argmax(dim=1)
    return [labels[int(token_id)] for token_id in next_token_ids]


def build_multi_lora_model(strategy: str = "gathered") -> Any:
    """Build a toy multi-LoRA model.

    Args:
        strategy: ``"loop"`` for row-by-row application or ``"gathered"``
            for vectorized adapter selection.
    """

    torch = require_torch()

    class AbstractMultiLoraModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 10)
            self.linear = torch.nn.Linear(10, 10)
            self.lm_head = torch.nn.Linear(10, 10)

        def linear_lora(self, x: Any, loras_a: Any, loras_b: Any, lora_indices: Any) -> Any:
            raise NotImplementedError

        def forward(self, input_ids: Any, loras_a: Any, loras_b: Any, lora_indices: Any) -> Any:
            x = self.embedding(input_ids)
            x = self.linear_lora(x, loras_a, loras_b, lora_indices)
            return self.lm_head(x)

    class LoopMultiLoraModel(AbstractMultiLoraModel):
        def linear_lora(self, x: Any, loras_a: Any, loras_b: Any, lora_indices: Any) -> Any:
            y = self.linear(x)
            for batch_idx, lora_idx in enumerate(lora_indices.detach().cpu().tolist()):
                y[batch_idx] += x[batch_idx] @ loras_a[lora_idx] @ loras_b[lora_idx]
            return y

    class GatheredMultiLoraModel(AbstractMultiLoraModel):
        def linear_lora(self, x: Any, loras_a: Any, loras_b: Any, lora_indices: Any) -> Any:
            y = self.linear(x)
            lora_a = torch.index_select(loras_a, 0, lora_indices)
            lora_b = torch.index_select(loras_b, 0, lora_indices)
            return y + x @ lora_a @ lora_b

    if strategy == "loop":
        return LoopMultiLoraModel()
    if strategy == "gathered":
        return GatheredMultiLoraModel()
    raise ValueError("strategy must be 'loop' or 'gathered'")
