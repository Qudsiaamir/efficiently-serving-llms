# Benchmarking

## Goal

The benchmark framework makes it easy to measure local behavior without
inventing or hardcoding performance results. Use it to compare strategies,
hardware, dependency versions, and configuration choices.

## Run benchmarks

```bash
make benchmark
```

Run a specific scenario and save JSON:

```bash
python benchmarks/run_benchmark.py \
  --scenario multi-lora-gathered \
  --samples 25 \
  --batch-sizes 1 2 4 8 16 \
  --output benchmarks/local_results.json
```

Compare against the loop-based implementation:

```bash
python benchmarks/run_benchmark.py --scenario multi-lora-loop --samples 25
```

## Result fields

| Field | Meaning |
|---|---|
| `scenario` | Benchmark scenario name |
| `batch_size` | Number of requests in the synthetic batch |
| `samples` | Number of timed runs |
| `mean_s` | Mean wall-clock latency in seconds |
| `p50_s` | Median latency in seconds |
| `p95_s` | Approximate p95 latency in seconds |
| `min_s` | Minimum observed latency in seconds |
| `max_s` | Maximum observed latency in seconds |

## Reporting template

| Scenario | Dataset/Input Size | Runtime | Latency p50 | Latency p95 | Memory | Notes |
|---|---:|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | Generated locally |

## Interpreting results

Benchmark output is sensitive to CPU/GPU type, PyTorch version, thread settings,
thermal state, model size, sequence length, and batch size. Do not compare
numbers across machines without recording the environment.

Recommended metadata to capture:

- CPU/GPU model and memory
- Python, PyTorch, CUDA, and driver versions
- Batch sizes and sequence length
- Model and adapter dimensions
- Number of warmup and measurement samples
- Whether measurements are CPU-only or GPU-backed
