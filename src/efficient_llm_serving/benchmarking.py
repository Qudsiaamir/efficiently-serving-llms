"""Benchmark utilities shared by benchmark scripts and tests."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencySummary:
    """Summary statistics for a latency sample."""

    count: int
    mean_s: float
    p50_s: float
    p95_s: float
    min_s: float
    max_s: float


def summarize_latencies(latencies_s: list[float]) -> LatencySummary:
    """Compute stable latency statistics from a non-empty sample."""

    if not latencies_s:
        raise ValueError("latencies_s must not be empty")

    sorted_latencies = sorted(latencies_s)
    p95_idx = min(len(sorted_latencies) - 1, int(round(0.95 * (len(sorted_latencies) - 1))))
    return LatencySummary(
        count=len(sorted_latencies),
        mean_s=statistics.fmean(sorted_latencies),
        p50_s=statistics.median(sorted_latencies),
        p95_s=sorted_latencies[p95_idx],
        min_s=sorted_latencies[0],
        max_s=sorted_latencies[-1],
    )


def time_callable(fn: Callable[[], object], samples: int) -> LatencySummary:
    """Run a callable repeatedly and summarize elapsed wall-clock time."""

    if samples <= 0:
        raise ValueError("samples must be positive")

    latencies_s: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        fn()
        latencies_s.append(time.perf_counter() - start)
    return summarize_latencies(latencies_s)
