# Design Decisions

## Preserve notebooks, extract production logic

The notebooks are useful learning artifacts, but reusable code should live in
`src/efficient_llm_serving`. This keeps experiments available while making the
project testable, importable, and easier to review.

## Optional ML dependencies

PyTorch and Transformers are large dependencies and may not be available in
basic CI or recruiter review environments. The package imports without them and
raises clear runtime errors only for tensor/model operations.

## Pytest plus lightweight CI

The test suite covers standard-library-only utilities by default and skips
PyTorch-specific tests when PyTorch is not installed. This makes CI fast while
still supporting deeper validation in ML environments.

## Benchmark framework without claimed results

The repository includes benchmark code and a results template, but does not
publish numbers that were not generated and verified locally. This keeps the
portfolio credible and avoids misleading performance claims.

## Educational implementations labeled clearly

The quantization and LoRA examples demonstrate mechanics. They are not presented
as replacements for production serving engines, optimized kernels, or
quantization libraries.
