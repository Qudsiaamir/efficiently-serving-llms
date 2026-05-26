"""Run local benchmarks for efficient LLM serving primitives.

The script reports only measurements collected on the current machine. It does
not ship benchmark numbers in the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from efficient_llm_serving.benchmarking import time_callable
from efficient_llm_serving.config import BenchmarkConfig
from efficient_llm_serving.dependencies import require_torch
from efficient_llm_serving.lora import build_multi_lora_model, generate_toy_token


def run_multi_lora_benchmark(strategy: str, config: BenchmarkConfig) -> list[dict[str, object]]:
    """Benchmark one-token generation for the toy multi-LoRA model."""

    torch = require_torch()
    model = build_multi_lora_model(strategy=strategy)
    loras_a = torch.randn(config.num_loras, config.hidden_size, config.lora_rank)
    loras_b = torch.randn(config.num_loras, config.lora_rank, config.hidden_size)

    rows: list[dict[str, object]] = []
    for batch_size in config.batch_sizes:
        input_ids = torch.randint(config.vocab_size, (batch_size, config.seq_len), dtype=torch.long)
        lora_indices = torch.randint(config.num_loras, (batch_size,), dtype=torch.long)

        def generate_once(input_ids=input_ids, lora_indices=lora_indices) -> None:
            generate_toy_token(
                model,
                input_ids=input_ids,
                loras_a=loras_a,
                loras_b=loras_b,
                lora_indices=lora_indices,
            )

        summary = time_callable(generate_once, samples=config.num_samples)
        rows.append(
            {
                "scenario": f"multi_lora_{strategy}",
                "batch_size": batch_size,
                "samples": summary.count,
                "mean_s": summary.mean_s,
                "p50_s": summary.p50_s,
                "p95_s": summary.p95_s,
                "min_s": summary.min_s,
                "max_s": summary.max_s,
            }
        )
    return rows


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    """Write benchmark rows as pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local benchmark scenarios.")
    parser.add_argument(
        "--scenario",
        choices=["multi-lora-loop", "multi-lora-gathered"],
        default="multi-lora-gathered",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategy = "loop" if args.scenario == "multi-lora-loop" else "gathered"
    config = BenchmarkConfig(batch_sizes=tuple(args.batch_sizes), num_samples=args.samples)

    try:
        rows = run_multi_lora_benchmark(strategy=strategy, config=config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for row in rows:
        print(
            "{scenario} batch_size={batch_size} samples={samples} "
            "p50_s={p50_s:.6f} p95_s={p95_s:.6f} mean_s={mean_s:.6f}".format(**row)
        )

    if args.output:
        write_json(args.output, rows)
        print(f"Wrote benchmark results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
