# Troubleshooting

## PyTorch is missing

Install the ML extras:

```bash
pip install -e ".[ml]"
```

For CUDA-enabled installs, follow the official PyTorch command for your CUDA
version and platform.

## Transformers cannot download a model

Check network access and Hugging Face authentication. Some environments require:

```bash
export HF_TOKEN=...
```

Never commit tokens. Use `.env.example` as the template for local secrets.

## Benchmarks exit with an optional dependency error

The benchmark scenarios require PyTorch. Install:

```bash
pip install -e ".[dev,ml]"
```

## Notebooks reference old helper modules

Some original notebooks were created in an external learning environment. Use
the package modules under `src/efficient_llm_serving` for reproducible logic and
treat notebooks as preserved exploratory artifacts.

## CI skips ML-heavy validation

The default CI is intentionally lightweight. Add a separate GPU-enabled workflow
or self-hosted runner before publishing hardware-specific benchmark results.
