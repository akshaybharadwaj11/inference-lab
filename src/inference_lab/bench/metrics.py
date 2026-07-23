"""Inference benchmark metrics with statistical analysis.

Computes production-relevant metrics for LLM inference benchmarking:
    - Throughput: tokens/second (prefill and decode separately)
    - Latency: TTFT, ITL, end-to-end at p50/p95/p99
    - Goodput: successful requests per second under load
    - Efficiency: tokens per GPU-second, utilization vs theoretical peak
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RequestResult:
    """Timing results for a single benchmark request."""
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float               # time to first token
    total_latency_ms: float      # end-to-end
    itl_ms: float                # average inter-token latency
    queue_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    @property
    def decode_throughput_tps(self) -> float:
        """Tokens per second during decode phase."""
        if self.decode_time_ms <= 0:
            return 0.0
        return self.generated_tokens / (self.decode_time_ms / 1000)

    @property
    def prefill_throughput_tps(self) -> float:
        """Prompt tokens per second during prefill."""
        if self.prefill_time_ms <= 0:
            return 0.0
        return self.prompt_tokens / (self.prefill_time_ms / 1000)


@dataclass
class LatencyStats:
    """Percentile-based latency statistics."""
    metric_name: str
    count: int
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float

    @classmethod
    def from_values(cls, name: str, values: list[float]) -> LatencyStats:
        if not values:
            return cls(name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        s = sorted(values)
        n = len(s)
        mean = statistics.mean(s)
        std = statistics.stdev(s) if n > 1 else 0.0

        def pct(p: float) -> float:
            idx = int(n * p / 100)
            idx = min(idx, n - 1)
            return round(s[idx], 3)

        return cls(
            metric_name=name,
            count=n,
            mean=round(mean, 3),
            std=round(std, 3),
            min=round(s[0], 3),
            p25=pct(25),
            p50=pct(50),
            p75=pct(75),
            p90=pct(90),
            p95=pct(95),
            p99=pct(99),
            max=round(s[-1], 3),
        )


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results with all relevant metrics."""

    # Metadata
    model: str
    backend: str  # "vllm", "sglang", "trt-llm"
    dtype: str
    tp_degree: int
    gpu: str
    timestamp: str

    # Workload config
    num_requests: int
    concurrency: int
    prompt_len_mean: int
    output_len_mean: int

    # Aggregate throughput
    total_duration_s: float
    total_prompt_tokens: int
    total_generated_tokens: int
    requests_per_second: float
    prompt_throughput_tps: float   # total prompt tokens / total time
    decode_throughput_tps: float   # total generated tokens / total time

    # Success rate
    successful_requests: int
    failed_requests: int
    success_rate: float

    # Latency distributions
    ttft: LatencyStats = field(default_factory=lambda: LatencyStats("ttft", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    itl: LatencyStats = field(default_factory=lambda: LatencyStats("itl", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    e2e: LatencyStats = field(default_factory=lambda: LatencyStats("e2e", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    @classmethod
    def from_results(
        cls,
        results: list[RequestResult],
        model: str,
        backend: str,
        dtype: str,
        tp_degree: int,
        gpu: str,
        timestamp: str,
        concurrency: int,
        total_duration_s: float,
    ) -> BenchmarkReport:
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_prompt = sum(r.prompt_tokens for r in successful)
        total_generated = sum(r.generated_tokens for r in successful)

        return cls(
            model=model,
            backend=backend,
            dtype=dtype,
            tp_degree=tp_degree,
            gpu=gpu,
            timestamp=timestamp,
            num_requests=len(results),
            concurrency=concurrency,
            prompt_len_mean=round(statistics.mean([r.prompt_tokens for r in results])) if results else 0,
            output_len_mean=round(statistics.mean([r.generated_tokens for r in results])) if results else 0,
            total_duration_s=round(total_duration_s, 3),
            total_prompt_tokens=total_prompt,
            total_generated_tokens=total_generated,
            requests_per_second=round(len(successful) / total_duration_s, 2) if total_duration_s > 0 else 0,
            prompt_throughput_tps=round(total_prompt / total_duration_s, 2) if total_duration_s > 0 else 0,
            decode_throughput_tps=round(total_generated / total_duration_s, 2) if total_duration_s > 0 else 0,
            successful_requests=len(successful),
            failed_requests=len(failed),
            success_rate=round(len(successful) / len(results) * 100, 1) if results else 0,
            ttft=LatencyStats.from_values("ttft_ms", [r.ttft_ms for r in successful]),
            itl=LatencyStats.from_values("itl_ms", [r.itl_ms for r in successful]),
            e2e=LatencyStats.from_values("e2e_ms", [r.total_latency_ms for r in successful]),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkReport:
        with open(path) as f:
            data = json.load(f)
        # Reconstruct LatencyStats
        for key in ("ttft", "itl", "e2e"):
            if key in data and isinstance(data[key], dict):
                data[key] = LatencyStats(**data[key])
        return cls(**data)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Benchmark Report: {self.model}")
        print(f"  Backend: {self.backend} | GPU: {self.gpu} | TP: {self.tp_degree}")
        print(f"{'='*60}")
        print(f"\n  Workload")
        print(f"    Requests: {self.num_requests} ({self.successful_requests} ok, {self.failed_requests} failed)")
        print(f"    Concurrency: {self.concurrency}")
        print(f"    Avg prompt: {self.prompt_len_mean} tokens | Avg output: {self.output_len_mean} tokens")
        print(f"\n  Throughput")
        print(f"    Requests/s:      {self.requests_per_second}")
        print(f"    Prefill tok/s:   {self.prompt_throughput_tps}")
        print(f"    Decode tok/s:    {self.decode_throughput_tps}")
        print(f"\n  Latency (ms)")
        for name, stats in [("TTFT", self.ttft), ("ITL", self.itl), ("E2E", self.e2e)]:
            if stats.count > 0:
                print(f"    {name:6s}  p50={stats.p50:8.1f}  p95={stats.p95:8.1f}  "
                      f"p99={stats.p99:8.1f}  mean={stats.mean:8.1f}")
        print()
