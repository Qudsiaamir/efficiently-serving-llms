# Repository Audit

## Project classification

Mixed ML systems, GenAI serving, and notebook experiment repository focused on
efficient LLM inference.

## Original entry points

- `Lesson_1-Text_Generation.ipynb`
- `Lesson_2-Batching.ipynb`
- `Lesson_3-Continuous_Batching.ipynb`
- `Lesson_4-Quantization.ipynb`
- `Lesson_5-Low-Rank_Adaptation.ipynb`
- `Lesson_6-Multi-LoRA.ipynb`
- `predibase_lorax.ipynb`

These are now preserved in `notebooks/exploratory/`.

## Reusable notebook logic extracted

- Greedy next-token generation
- KV-cache decode loops
- Batched generation with position ids and attention masks
- Request utilities for continuous batching simulations
- Simple tensor quantization/dequantization
- Toy LoRA and multi-LoRA models
- Benchmark timing helpers

## Gaps fixed

- Added README, license, `.gitignore`, `.env.example`, package metadata, Makefile, Dockerfile, CI, tests, benchmarks, examples, and docs.
- Added importable package modules under `src/`.
- Added benchmark result template without fabricated metrics.

## Known limitations

- The local environment used during restructuring did not include PyTorch,
  Transformers, or pytest, so full ML execution must be validated after
  installing dependencies.
- Original notebooks are preserved but may still need dependency/path updates for
  fully automated execution.
