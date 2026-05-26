"""Configuration models used by examples and benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime settings for local benchmark experiments.

    Defaults are intentionally small so the benchmark can run on a laptop CPU.
    Increase these values when running on a GPU-backed workstation.
    """

    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    num_samples: int = 10
    seq_len: int = 8
    vocab_size: int = 10
    hidden_size: int = 10
    num_loras: int = 8
    lora_rank: int = 4
