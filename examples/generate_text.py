"""Minimal Hugging Face generation example using the package helpers."""

from __future__ import annotations

import argparse
import sys

from efficient_llm_serving.dependencies import require_transformers
from efficient_llm_serving.generation import (
    configure_tokenizer_for_batching,
    generate_with_kv_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with greedy KV-cache decoding.")
    parser.add_argument("--model", default="distilgpt2", help="Hugging Face causal LM id.")
    parser.add_argument(
        "--prompt",
        default="Efficient LLM serving matters because",
        help="Prompt text.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        transformers = require_transformers()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    configure_tokenizer_for_batching(tokenizer, model)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    output = generate_with_kv_cache(model, tokenizer, inputs, max_new_tokens=args.max_new_tokens)
    print(args.prompt + output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
