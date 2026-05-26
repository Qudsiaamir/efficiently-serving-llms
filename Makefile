.PHONY: install install-ml test lint format run benchmark clean

install:
	python -m pip install -e ".[dev]"

install-ml:
	python -m pip install -e ".[dev,ml,serving]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

run:
	python examples/generate_text.py

benchmark:
	python benchmarks/run_benchmark.py --scenario multi-lora-gathered --samples 10 --batch-sizes 1 2 4 8

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
