# Efficiently Serving LLMs

Production-oriented examples for understanding, implementing, and benchmarking
core LLM inference serving techniques.

## Why this project matters

LLM serving performance is shaped by practical engineering choices: greedy
decode loops, KV-cache usage, static and continuous batching, quantization,
adapter routing, and benchmark discipline. This repository turns notebook-heavy
experiments into a clean Python project that demonstrates how those mechanics
can be organized, tested, documented, and evaluated like production software.

## Key features

- Reusable Python package for LLM serving primitives under `src/`
- Extracted helpers for token generation, batching, quantization, LoRA, and multi-LoRA
- Preserved exploratory notebooks with cleaned repository placement
- Local benchmark harness with JSON output support
- Pytest test suite for package imports, request utilities, benchmark summaries, and quantization
- Dockerfile, Makefile, dependency metadata, and GitHub Actions CI
- Architecture, benchmarking, troubleshooting, and design-decision documentation

## Architecture overview

The project separates exploratory notebooks from reusable code:

- `efficient_llm_serving.generation`: greedy next-token generation and KV-cache decode helpers
- `efficient_llm_serving.batching`: batch decode utilities and request objects for continuous batching simulations
- `efficient_llm_serving.quantization`: educational uint8 affine quantization helpers
- `efficient_llm_serving.lora`: toy LoRA and multi-LoRA model components
- `efficient_llm_serving.benchmarking`: latency timing and summary utilities
- `benchmarks/run_benchmark.py`: local benchmark runner for adapter-serving scenarios

See [docs/architecture.md](docs/architecture.md) for component and data-flow details.

## Tech stack

- Python 3.10+
- PyTorch and Hugging Face Transformers for model execution
- pytest for tests
- Ruff for linting and formatting
- Docker for reproducible container execution
- GitHub Actions for CI

The package imports without PyTorch installed. Tensor/model functionality raises
clear installation guidance at runtime.

## Repository structure

```text
.
├── benchmarks/              # Local benchmark runner and results template
├── docs/                    # Architecture, design, benchmarking, troubleshooting
├── examples/                # Runnable examples
├── notebooks/exploratory/   # Preserved notebook experiments
├── src/efficient_llm_serving/
├── tests/
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml]"
make test
```

For a lighter install without PyTorch or Transformers:

```bash
pip install -e ".[dev]"
```

## Configuration

Runtime examples use command-line arguments. Environment variables are not
required for the local toy benchmarks. If you connect to hosted model APIs, copy
`.env.example` and provide your own endpoint and API key.

## Usage examples

Run a small Hugging Face decode example:

```bash
python examples/generate_text.py \
  --model distilgpt2 \
  --prompt "Efficient LLM serving matters because" \
  --max-new-tokens 16
```

Use the package directly:

```python
from efficient_llm_serving.benchmarking import summarize_latencies

summary = summarize_latencies([0.012, 0.015, 0.014])
print(summary.p50_s)
```

## Benchmarking

Run the local multi-LoRA benchmark:

```bash
make benchmark
```

Or write JSON results:

```bash
python benchmarks/run_benchmark.py \
  --scenario multi-lora-gathered \
  --samples 25 \
  --batch-sizes 1 2 4 8 16 \
  --output benchmarks/local_results.json
```

No benchmark results are claimed in this repository. Use
[benchmarks/results_template.md](benchmarks/results_template.md) and
[docs/benchmarking.md](docs/benchmarking.md) to record measurements generated
locally on your target hardware.

## Testing

```bash
make test
```

Some tests require PyTorch and are skipped automatically when it is unavailable.

## Docker usage

Build and run the test container:

```bash
docker build -t efficiently-serving-llms .
docker run --rm efficiently-serving-llms
```

## CI/CD

The GitHub Actions workflow installs the package with development extras, runs
Ruff, and executes pytest. ML-heavy benchmark jobs are intentionally left for
local or GPU-enabled runners.

## Professional highlights

- Clean Python package structure extracted from notebook experiments
- Reproducible development workflow with Makefile, Dockerfile, and CI
- Benchmark-ready framework that avoids invented performance claims
- Modular implementation of LLM serving concepts: KV-cache decoding, batching, quantization, LoRA, and multi-LoRA
- Documentation written for technical reviewers, recruiters, and engineering managers
- Clear separation between exploratory research artifacts and reusable production-style code

## Roadmap

- Add GPU benchmark profiles for A10/T4/L4/A100-class hardware
- Add FastAPI inference endpoint for batch and streaming generation demos
- Add OpenTelemetry metrics hooks for latency, throughput, and request queue depth
- Add Kubernetes deployment manifests for an inference service demo
- Add richer notebook smoke tests with Papermill once model dependencies are available

## Limitations

- Quantization and LoRA modules are educational implementations, not replacements
  for production inference engines.
- Benchmark outputs depend heavily on hardware, installed libraries, and model
  choice; generate results locally before making performance claims.
- The preserved notebooks originated as lesson experiments and may require
  additional local dependencies for full execution.

## License

MIT. See [LICENSE](LICENSE).
