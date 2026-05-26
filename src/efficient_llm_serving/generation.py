"""Greedy token generation helpers with optional KV-cache support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from efficient_llm_serving.dependencies import require_torch


def configure_tokenizer_for_batching(tokenizer: Any, model: Any) -> None:
    """Configure common tokenizer/model settings for left-padded decoding."""

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.config.pad_token_id = model.config.eos_token_id


def greedy_next_token_id(model: Any, inputs: Mapping[str, Any], batched: bool = False) -> Any:
    """Return the highest-probability next token id for a model forward pass."""

    torch = require_torch()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[:, -1, :] if batched else logits[0, -1, :]
    dim = 1 if batched else 0
    return last_logits.argmax(dim=dim)


def greedy_next_token_with_past(
    model: Any, inputs: Mapping[str, Any], batched: bool = False
) -> tuple[Any, Any]:
    """Return the greedy next token id and model KV-cache."""

    torch = require_torch()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[:, -1, :] if batched else logits[0, -1, :]
    dim = 1 if batched else 0
    return last_logits.argmax(dim=dim), outputs.past_key_values


def generate_with_kv_cache(
    model: Any,
    tokenizer: Any,
    inputs: Mapping[str, Any],
    max_new_tokens: int,
) -> str:
    """Generate text greedily for a single prompt using a KV-cache."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    torch = require_torch()
    generated_tokens: list[str] = []
    next_inputs = dict(inputs)

    for _ in range(max_new_tokens):
        next_token_id, past_key_values = greedy_next_token_with_past(model, next_inputs)
        next_inputs = {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]])],
                dim=1,
            ),
            "past_key_values": past_key_values,
        }
        generated_tokens.append(tokenizer.decode(next_token_id))

    return "".join(generated_tokens)
