.PHONY: install dev lint test bench clean docker

install:
	pip install -e .

dev:
	pip install -e ".[dev,profiling]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest tests/ -v --tb=short -m "not gpu and not slow"

test-all:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=inference_lab --cov-report=html --cov-report=term-missing

bench:
	python -m inference_lab.bench.runner \
		--config benchmarks/configs/llama3_8b.yaml \
		--output benchmarks/baselines/latest.json

bench-regression:
	python -m inference_lab.bench.regression \
		--baseline benchmarks/baselines/latest.json \
		--current /tmp/bench_current.json \
		--threshold 0.05

kv-analysis:
	python -m inference_lab.profiler.kv_cache \
		--model meta-llama/Llama-3.1-8B-Instruct \
		--seq-lens 512 1024 2048 4096 8192 16384 \
		--batch-sizes 1 4 8 16 32 64

serve:
	python -m inference_lab.serving.api \
		--model meta-llama/Llama-3.1-8B-Instruct \
		--port 8000

docker:
	docker build -t inference-lab:latest .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
