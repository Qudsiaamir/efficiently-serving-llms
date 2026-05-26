"""Batch and continuous-batching helpers for decoder-only LLM inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from efficient_llm_serving.dependencies import require_torch
from efficient_llm_serving.generation import greedy_next_token_with_past


@dataclass
class GenerationRequest:
    """A small request object for continuous batching simulations."""

    prompt: str
    max_new_tokens: int
    request_id: str | None = None
    generated_text: str = ""
    generated_token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether this request has generated its requested number of tokens."""

        return self.generated_token_count >= self.max_new_tokens


def build_position_ids(attention_mask: Any) -> Any:
    """Build GPT-style position ids for left-padded batch inputs."""

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def append_decode_inputs(
    inputs: Mapping[str, Any],
    next_token_ids: Any,
    past_key_values: Any,
) -> dict[str, Any]:
    """Create the next model inputs after one batched decode step."""

    torch = require_torch()
    return {
        "input_ids": next_token_ids.reshape((-1, 1)),
        "position_ids": inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
        "attention_mask": torch.cat(
            [
                inputs["attention_mask"],
                torch.ones((next_token_ids.shape[0], 1), device=next_token_ids.device),
            ],
            dim=1,
        ),
        "past_key_values": past_key_values,
    }


def generate_batch_with_kv_cache(
    model: Any,
    tokenizer: Any,
    inputs: Mapping[str, Any],
    max_new_tokens: int,
) -> list[str]:
    """Generate text greedily for a padded batch using a KV-cache."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    batch_size = inputs["input_ids"].shape[0]
    generated_tokens: list[list[str]] = [[] for _ in range(batch_size)]
    next_inputs = {"position_ids": build_position_ids(inputs["attention_mask"]), **dict(inputs)}

    for _ in range(max_new_tokens):
        next_token_ids, past_key_values = greedy_next_token_with_past(
            model, next_inputs, batched=True
        )
        next_inputs = append_decode_inputs(next_inputs, next_token_ids, past_key_values)
        for idx, token in enumerate(tokenizer.batch_decode(next_token_ids)):
            generated_tokens[idx].append(token)

    return ["".join(tokens) for tokens in generated_tokens]


def chunk_requests(
    requests: Sequence[GenerationRequest],
    batch_size: int,
) -> list[list[GenerationRequest]]:
    """Split generation requests into fixed-size batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(requests[i : i + batch_size]) for i in range(0, len(requests), batch_size)]


def filter_incomplete_requests(requests: Sequence[GenerationRequest]) -> list[GenerationRequest]:
    """Return requests that still need decode steps."""

    return [request for request in requests if not request.complete]
