"""Benchmark runner for LLM inference engines.

Orchestrates end-to-end benchmarking:
    1. Load configuration from YAML
    2. Generate synthetic workload
    3. Run requests against vLLM (offline or online)
    4. Collect per-request timing
    5. Compute aggregate metrics
    6. Save results for regression detection

Supports two modes:
    - Offline: Use vLLM's LLM class directly (no network, pure engine perf)
    - Online: Hit a running vLLM API server (measures full serving stack)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from inference_lab.bench.metrics import BenchmarkReport, RequestResult, LatencyStats
from inference_lab.bench.workloads import (
    WorkloadConfig,
    DistributionType,
    generate_workload,
    workload_stats,
)


@dataclass
class BenchConfig:
    """Benchmark run configuration, loaded from YAML."""
    model: str
    backend: str = "vllm"
    dtype: str = "float16"
    tp_degree: int = 1
    gpu: str = "unknown"
    max_model_len: int = 8192

    # Workload
    num_requests: int = 100
    concurrency: int = 1
    prompt_distribution: str = "uniform"
    prompt_min: int = 128
    prompt_max: int = 2048
    output_distribution: str = "uniform"
    output_min: int = 64
    output_max: int = 512
    seed: int = 42

    # Engine settings
    enforce_eager: bool = False
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.90

    # Online mode
    api_url: Optional[str] = None  # If set, use online mode

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _run_offline_benchmark(config: BenchConfig) -> BenchmarkReport:
    """Run benchmark using vLLM's offline LLM class.

    This measures pure engine performance without network/API overhead.
    Requires vLLM to be installed with GPU support.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise RuntimeError(
            "vLLM is required for offline benchmarks. "
            "Install with: pip install vllm"
        )

    # Generate workload
    wl_config = WorkloadConfig(
        num_requests=config.num_requests,
        prompt_distribution=DistributionType(config.prompt_distribution),
        prompt_min=config.prompt_min,
        prompt_max=config.prompt_max,
        output_distribution=DistributionType(config.output_distribution),
        output_min=config.output_min,
        output_max=config.output_max,
        seed=config.seed,
    )
    workload = generate_workload(wl_config)
    wl_stats = workload_stats(workload)

    print(f"  Workload: {wl_stats['num_requests']} requests")
    print(f"  Prompt tokens: mean={wl_stats['prompt_tokens']['mean']}, "
          f"range=[{wl_stats['prompt_tokens']['min']}, {wl_stats['prompt_tokens']['max']}]")
    print(f"  Output tokens: mean={wl_stats['output_tokens']['mean']}, "
          f"range=[{wl_stats['output_tokens']['min']}, {wl_stats['output_tokens']['max']}]")

    # Initialize engine
    print(f"\n  Loading {config.model}...")
    llm = LLM(
        model=config.model,
        dtype=config.dtype,
        tensor_parallel_size=config.tp_degree,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=config.enable_prefix_caching,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )

    # Prepare sampling params per request
    prompts = [req.prompt_text for req in workload]
    sampling_params_list = [
        SamplingParams(
            max_tokens=req.max_output_tokens,
            temperature=0.0,  # greedy for reproducibility
        )
        for req in workload
    ]

    # Run benchmark
    print(f"  Running {config.num_requests} requests...")
    start_time = time.monotonic()
    outputs = llm.generate(prompts, sampling_params_list)
    total_duration = time.monotonic() - start_time

    # Collect results
    results: list[RequestResult] = []
    for i, output in enumerate(outputs):
        gen_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)

        # vLLM offline mode doesn't expose per-request timing breakdown,
        # so we estimate proportionally from total time
        est_total_ms = total_duration * 1000 / len(outputs)

        results.append(RequestResult(
            request_id=str(i),
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            ttft_ms=0.0,  # not available in offline mode
            total_latency_ms=est_total_ms,
            itl_ms=est_total_ms / gen_tokens if gen_tokens > 0 else 0,
        ))

    timestamp = datetime.now(timezone.utc).isoformat()
    report = BenchmarkReport.from_results(
        results=results,
        model=config.model,
        backend=config.backend,
        dtype=config.dtype,
        tp_degree=config.tp_degree,
        gpu=config.gpu,
        timestamp=timestamp,
        concurrency=config.concurrency,
        total_duration_s=total_duration,
    )

    return report


async def _run_online_benchmark(config: BenchConfig) -> BenchmarkReport:
    """Run benchmark against a running vLLM API server.

    Sends concurrent requests and measures real TTFT, ITL, and E2E latency.
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx is required for online benchmarks: pip install httpx")

    wl_config = WorkloadConfig(
        num_requests=config.num_requests,
        prompt_distribution=DistributionType(config.prompt_distribution),
        prompt_min=config.prompt_min,
        prompt_max=config.prompt_max,
        output_distribution=DistributionType(config.output_distribution),
        output_min=config.output_min,
        output_max=config.output_max,
        seed=config.seed,
    )
    workload = generate_workload(wl_config)
    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[RequestResult] = []

    async def send_request(client: httpx.AsyncClient, req_idx: int) -> RequestResult:
        req = workload[req_idx]
        payload = {
            "model": config.model,
            "prompt": req.prompt_text,
            "max_tokens": req.max_output_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        async with semaphore:
            start = time.monotonic()
            first_token_time: float | None = None
            token_times: list[float] = []
            generated_tokens = 0

            try:
                async with client.stream(
                    "POST",
                    f"{config.api_url}/v1/completions",
                    json=payload,
                    timeout=120.0,
                ) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        now = time.monotonic()
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)
                        generated_tokens += 1

                end = time.monotonic()
                ttft = (first_token_time - start) * 1000 if first_token_time else 0.0
                total_ms = (end - start) * 1000

                # Compute average inter-token latency
                itl = 0.0
                if len(token_times) > 1:
                    deltas = [
                        (token_times[i] - token_times[i - 1]) * 1000
                        for i in range(1, len(token_times))
                    ]
                    itl = sum(deltas) / len(deltas)

                return RequestResult(
                    request_id=str(req_idx),
                    prompt_tokens=req.prompt_tokens,
                    generated_tokens=generated_tokens,
                    ttft_ms=ttft,
                    total_latency_ms=total_ms,
                    itl_ms=itl,
                )

            except Exception as e:
                return RequestResult(
                    request_id=str(req_idx),
                    prompt_tokens=req.prompt_tokens,
                    generated_tokens=0,
                    ttft_ms=0,
                    total_latency_ms=0,
                    itl_ms=0,
                    success=False,
                    error=str(e),
                )

    # Fire all requests concurrently (bounded by semaphore)
    async with httpx.AsyncClient() as client:
        start_time = time.monotonic()
        tasks = [send_request(client, i) for i in range(len(workload))]
        results = await asyncio.gather(*tasks)
        total_duration = time.monotonic() - start_time

    timestamp = datetime.now(timezone.utc).isoformat()
    report = BenchmarkReport.from_results(
        results=list(results),
        model=config.model,
        backend=config.backend,
        dtype=config.dtype,
        tp_degree=config.tp_degree,
        gpu=config.gpu,
        timestamp=timestamp,
        concurrency=config.concurrency,
        total_duration_s=total_duration,
    )
    return report


def run_benchmark(config: BenchConfig) -> BenchmarkReport:
    """Run benchmark in the appropriate mode."""
    print(f"\n{'='*60}")
    print(f"  Inference Benchmark: {config.model}")
    print(f"  Mode: {'online' if config.api_url else 'offline'}")
    print(f"  Backend: {config.backend} | Dtype: {config.dtype} | TP: {config.tp_degree}")
    print(f"{'='*60}\n")

    if config.api_url:
        return asyncio.run(_run_online_benchmark(config))
    else:
        return _run_offline_benchmark(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference benchmarks")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--api-url", type=str, default=None,
        help="vLLM API server URL (enables online mode)",
    )
    args = parser.parse_args()

    config = BenchConfig.from_yaml(args.config)
    if args.api_url:
        config.api_url = args.api_url

    report = run_benchmark(config)
    report.print_summary()

    if args.output:
        report.save(args.output)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
