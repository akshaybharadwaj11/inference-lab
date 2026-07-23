# Architecture & Design Decisions

## Overview

inference-lab is structured as four independent but composable modules:

```
bench/       →  Measure inference performance
profiler/    →  Understand why performance is what it is
serving/     →  Run a model with observability
eval/        →  Gate deployments on quality
```

Each module can be used standalone or composed into a pipeline:
```
model checkpoint → eval gate → deploy to serving → benchmark → regression check
```

---

## Key Design Decisions

### 1. KV Cache Estimation is Pure Math (No GPU Required)

The KV cache analyzer (`profiler/kv_cache.py`) computes memory requirements
from model architecture parameters alone. This is intentional:

- **Capacity planning happens before you have GPU access.** Teams need to
  estimate how many sequences an A100 can serve before provisioning hardware.
- **The math is the interview question.** Being able to derive `num_layers × 2
  × num_kv_heads × head_dim × seq_len × dtype_bytes` from first principles,
  and then extend it to PagedAttention block fragmentation, demonstrates the
  exact reasoning NVIDIA/Fireworks interviewers test for.
- **GQA matters.** Llama 3 uses 8 KV heads vs 32 query heads (4:1 GQA). This
  4x reduction in KV cache is why GQA exists — and why the estimator tracks
  `num_kv_heads` separately from `num_attention_heads`.

### 2. Request Tracing Uses Phase-Based Spans

The tracer (`profiler/trace.py`) models requests as a sequence of timed phases:
```
received → queued → prefill → decode → detokenize → complete
```

This mirrors how production inference actually works:

- **Queue time** reveals scheduler pressure. If queue time dominates TTFT,
  the bottleneck is scheduling, not compute.
- **Prefill vs decode separation** matters because they have fundamentally
  different performance characteristics: prefill is compute-bound (matrix
  multiplications over the full prompt), decode is memory-bandwidth-bound
  (reading the entire KV cache to generate one token).
- **TTFT = queue + prefill.** This is what users experience as "time to
  start seeing a response." ITL (inter-token latency) determines perceived
  streaming speed.

### 3. Regression Detection is Threshold-Based, Not Statistical

The regression detector (`bench/regression.py`) uses simple percentage
thresholds rather than statistical tests (t-tests, Mann-Whitney). Reasons:

- **Inference benchmarks have low variance** when run on dedicated hardware
  with fixed workloads. The signal-to-noise ratio is high enough that a 5%
  threshold catches real regressions reliably.
- **False negatives are costlier than false positives** in inference. A 3%
  throughput regression that ships to production costs real money at scale.
  We'd rather flag it and investigate.
- **CI needs a binary gate.** The output is exit code 0 (pass) or 1 (fail).
  Statistical tests add complexity without changing the decision.

The `RegressionConfig` is configurable: use `.strict()` for production
gates, and adjust thresholds for development branches.

### 4. The Serving Layer is Intentionally Thin

`serving/api.py` is a FastAPI wrapper around vLLM's engine, not a replacement
for vLLM's built-in server. The value it adds:

- **Request lifecycle tracing** that vLLM's server doesn't expose at this
  granularity (per-phase timing, not just total latency).
- **Mock mode** for development without GPU — hit `/v1/completions` with
  simulated latency to test integration code.
- **A clean surface for extensions**: custom scheduling policies, request
  prioritization, multi-tenant isolation — the things Fireworks builds.

### 5. Eval Gating is the Pipeline, Not the Eval

The validator (`eval/validator.py`) provides the deployment gate interface,
not the eval implementation. In production, you'd integrate with:
- `lm-evaluation-harness` for standard benchmarks
- Custom eval suites for domain-specific quality
- A/B metrics from production traffic

The interface is what matters: `run_suite()` → `ValidationReport` → 
`all_passed` → deploy or block.

---

## Expansion Architecture

### Phase 2: Custom CUDA Kernels

```
src/inference_lab/kernels/
├── attention/
│   ├── naive_attention.cu     # Baseline: materialized attention matrix
│   ├── flash_attention.cu     # Tiled attention (FlashAttention paper)
│   └── paged_attention.cu     # Non-contiguous KV blocks (vLLM)
├── gemm/
│   └── tiled_gemm.cu          # Shared memory GEMM
└── CMakeLists.txt
```

Each kernel has a naive → optimized progression visible in git history.
Benchmark each with `ncu` (Nsight Compute) and document:
- Achieved vs theoretical memory bandwidth
- Occupancy analysis
- Bank conflict check

### Phase 3: Multi-GPU

```
src/inference_lab/distributed/
├── tp_benchmark.py            # Tensor parallelism profiling
├── nccl_profiler.py          # NCCL allreduce timing
└── placement.py              # Model layer → GPU placement strategy
```

Key analysis: measure allreduce overhead as a function of tensor size,
TP degree, and interconnect (NVLink vs PCIe). This is exactly what the
NVIDIA NCCL team works on.

### Phase 4: Advanced Serving

```
src/inference_lab/serving/
├── speculative.py            # Speculative decoding runtime
├── disaggregated.py          # Separate prefill/decode pools
└── prefix_cache.py           # Prefix-aware KV cache sharing
```

### Phase 5: Platform

```
src/inference_lab/platform/
├── control_plane/            # Go service for model lifecycle
│   ├── main.go
│   ├── handlers/
│   └── Dockerfile
├── ci_pipeline/              # Model CI/CD: eval → canary → promote
└── helm/                     # Helm chart for K8s deployment
```
