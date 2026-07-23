# llm-inference-lab

A production-grade LLM inference benchmarking, profiling, and serving toolkit built on top of [vLLM](https://github.com/vllm-project/vllm).

**What this does:**
- Benchmarks LLM inference throughput and latency with statistical rigor
- Estimates and analyzes KV cache memory usage across model configurations
- Detects performance regressions automatically in CI
- Serves models via FastAPI with request lifecycle tracing
- Validates model quality before deployment (eval gating)

**Why this exists:**
To provide a structured, testable, extensible foundation for understanding and improving LLM inference — from memory estimation through serving to automated regression detection.

---

## Quickstart

### Prerequisites
- Python 3.10+
- CUDA 12.1+ with a compatible GPU (RTX 3060+ for development, A100/H100 for serious benchmarking)
- 16GB+ system RAM

### Install

```bash
git clone https://github.com/YOUR_USER/llm-inference-lab.git
cd llm-inference-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the KV cache analyzer (no GPU required)

```bash
python -m inference_lab.profiler.kv_cache \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --seq-lens 512 1024 2048 4096 8192 \
    --batch-sizes 1 4 8 16 32 \
    --dtype float16
```

### Run a benchmark (requires GPU)

```bash
python -m inference_lab.bench.runner \
    --config benchmarks/configs/llama3_8b.yaml \
    --output benchmarks/baselines/llama3_8b_$(date +%Y%m%d).json
```

### Run the serving API

```bash
python -m inference_lab.serving.api \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

### Run tests

```bash
pytest tests/ -v
```

---

## Architecture

```
src/inference_lab/
├── bench/                  # Benchmarking engine
│   ├── runner.py           # Orchestrates benchmark runs against vLLM
│   ├── metrics.py          # Throughput, latency percentiles, TTFT, ITL
│   ├── regression.py       # Statistical regression detection (CI-ready)
│   └── workloads.py        # Synthetic workload generators (prefill/decode mix)
│
├── profiler/               # Analysis and profiling tools
│   ├── kv_cache.py         # KV cache memory estimator + fragmentation analyzer
│   └── trace.py            # Request lifecycle tracer with span events
│
├── serving/                # Model serving layer
│   ├── engine.py           # Thin wrapper around vLLM AsyncLLMEngine
│   ├── scheduler.py        # Pluggable scheduling policies (FCFS, priority, fair)
│   └── api.py              # FastAPI server with /generate, /health, /metrics
│
└── eval/                   # Model quality validation
    └── validator.py        # Pre-deployment eval gate (accuracy, perplexity)
```

See [docs/architecture.md](docs/architecture.md) for detailed design decisions.

---

## Expansion Roadmap

This project is designed to grow. Each phase adds a layer that maps to real production inference work:

| Phase | Focus | Key additions |
|-------|-------|---------------|
| **1 (this repo)** | Foundations | KV cache analysis, benchmarking, basic serving, regression CI |
| **2** | Optimization | Custom CUDA attention kernel, FlashAttention integration, Nsight profiling scripts |
| **3** | Multi-GPU | Tensor parallelism benchmarks, NCCL profiling, NVLink vs PCIe analysis |
| **4** | Advanced serving | Speculative decoding, disaggregated prefill/decode, prefix caching |
| **5** | Platform | Kubernetes deployment, Helm charts, model CI/CD pipeline, Go control plane |

---

## Contributing

PRs welcome. Run `make lint && make test` before submitting. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT
