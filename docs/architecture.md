# Architecture

## Project type

This is a mixed ML systems and GenAI infrastructure repository. The original
work was notebook-heavy experimentation around LLM inference mechanics. The
current structure preserves those notebooks while extracting reusable serving
logic into a testable Python package.

## Main components

| Component | Purpose |
|---|---|
| `generation.py` | Greedy next-token generation and KV-cache decode helpers |
| `batching.py` | Batched decode helpers and request utilities for continuous batching simulations |
| `quantization.py` | Educational uint8 affine quantization and dequantization utilities |
| `lora.py` | Toy LoRA and multi-LoRA model components for adapter-serving experiments |
| `benchmarking.py` | Timing and latency summary utilities |
| `benchmarks/run_benchmark.py` | CLI for local multi-LoRA benchmark runs |
| `notebooks/exploratory/` | Preserved learning notebooks and visual experiments |

## Data and control flow

1. A prompt or batch of prompts is tokenized by a Hugging Face tokenizer.
2. Generation helpers call a decoder-only model and select greedy next tokens.
3. KV-cache outputs are fed back into subsequent decode calls to avoid repeated prefill work.
4. Batch helpers maintain attention masks and position ids across decode steps.
5. LoRA helpers demonstrate how one base model can apply different adapter weights per request.
6. Benchmark scripts repeatedly execute a local scenario and report latency summaries from the current hardware.

## Important design decisions

- Heavy dependencies are optional so the package can be imported in lightweight CI.
- Notebook logic is extracted into small modules rather than hidden in Jupyter cells.
- Benchmark scripts generate local measurements only; no synthetic performance claims are stored.
- Educational implementations are documented as such to avoid overstating production readiness.

## Extension points

- Add a FastAPI layer around `generation.py` for HTTP or streaming inference.
- Add request queues and scheduler policies around `GenerationRequest`.
- Swap the toy LoRA model for PEFT, vLLM, TGI, or LoRAX-backed serving.
- Add benchmark scenarios for model prefill latency, decode throughput, and memory use.
- Persist benchmark results to JSON, Markdown, MLflow, or a metrics backend.

## Productionization considerations

- Add structured logging for request id, model id, adapter id, batch size, and generated token count.
- Track p50/p95/p99 latency, time to first token, tokens per second, queue depth, GPU memory, and error rate.
- Use mature inference runtimes for production quantization and paged attention.
- Add explicit model artifact management, cache paths, and compatibility checks.
- Use backpressure, request timeouts, and cancellation for serving endpoints.

## Cloud and Kubernetes readiness

This repository is ready to grow into a cloud-native inference demo. A practical
next step would add:

- Container image build and vulnerability scan in CI
- Kubernetes Deployment, Service, HPA/KEDA, and GPU node selector examples
- ConfigMap/Secret templates for model IDs and API credentials
- Prometheus metrics and OpenTelemetry traces
- Separate CPU CI tests from GPU benchmark workflows
