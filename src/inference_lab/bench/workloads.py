"""Synthetic workload generators for inference benchmarking.

Generates realistic request distributions that model production traffic:
    - Fixed: Constant prompt/output length (for controlled experiments)
    - Uniform: Uniform random distribution within a range
    - Zipf: Long-tail distribution (most requests short, some very long)
    - ShareGPT: Replay real conversation distributions from ShareGPT dataset
    - Poisson: Arrival rate follows Poisson process for load testing
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class DistributionType(str, Enum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    ZIPF = "zipf"
    NORMAL = "normal"


@dataclass
class WorkloadRequest:
    """A single synthetic benchmark request."""
    request_id: int
    prompt_text: str
    prompt_tokens: int          # expected prompt token count
    max_output_tokens: int      # max tokens to generate
    scheduled_at: float         # relative time to send (for Poisson arrivals)


@dataclass
class WorkloadConfig:
    """Configuration for synthetic workload generation."""
    num_requests: int = 100
    prompt_distribution: DistributionType = DistributionType.UNIFORM
    prompt_min: int = 128
    prompt_max: int = 2048
    prompt_fixed: int = 512
    output_distribution: DistributionType = DistributionType.UNIFORM
    output_min: int = 64
    output_max: int = 512
    output_fixed: int = 128
    arrival_rate: float = 0.0   # requests/sec (0 = send as fast as possible)
    seed: int = 42


def _sample_length(
    dist: DistributionType,
    min_val: int,
    max_val: int,
    fixed: int,
    rng: random.Random,
) -> int:
    """Sample a token length from the configured distribution."""
    if dist == DistributionType.FIXED:
        return fixed
    elif dist == DistributionType.UNIFORM:
        return rng.randint(min_val, max_val)
    elif dist == DistributionType.NORMAL:
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 6  # 99.7% within range
        val = int(rng.gauss(mean, std))
        return max(min_val, min(val, max_val))
    elif dist == DistributionType.ZIPF:
        # Zipf-like: most values near min, heavy tail toward max
        # Use inverse transform: x = min * (max/min)^U where U ~ Uniform(0,1)
        u = rng.random()
        val = int(min_val * ((max_val / max(min_val, 1)) ** u))
        return max(min_val, min(val, max_val))
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def _generate_dummy_prompt(target_tokens: int) -> str:
    """Generate a dummy prompt string of approximately target_tokens length.

    Uses a 4-character average per token approximation. The actual tokenization
    will vary by model, but this is close enough for benchmarking purposes.
    """
    # Approximate 4 chars per token
    words = [
        "The", "system", "processes", "multiple", "concurrent", "requests",
        "through", "a", "distributed", "pipeline", "that", "handles",
        "inference", "workloads", "across", "GPU", "clusters", "with",
        "automatic", "scaling", "and", "load", "balancing", "to",
        "maximize", "throughput", "while", "minimizing", "latency",
        "for", "each", "individual", "request", "in", "the", "queue",
    ]
    # Repeat words to fill target length
    chars_needed = target_tokens * 4
    prompt_words: list[str] = []
    total_chars = 0
    while total_chars < chars_needed:
        word = words[len(prompt_words) % len(words)]
        prompt_words.append(word)
        total_chars += len(word) + 1  # +1 for space
    return " ".join(prompt_words)


def generate_workload(config: WorkloadConfig) -> list[WorkloadRequest]:
    """Generate a synthetic benchmark workload.

    Returns a list of WorkloadRequest objects with prompt text and timing.
    """
    rng = random.Random(config.seed)
    requests: list[WorkloadRequest] = []
    current_time = 0.0

    for i in range(config.num_requests):
        prompt_tokens = _sample_length(
            config.prompt_distribution,
            config.prompt_min, config.prompt_max, config.prompt_fixed,
            rng,
        )
        output_tokens = _sample_length(
            config.output_distribution,
            config.output_min, config.output_max, config.output_fixed,
            rng,
        )

        # Poisson arrival time
        if config.arrival_rate > 0:
            inter_arrival = rng.expovariate(config.arrival_rate)
            current_time += inter_arrival
        else:
            current_time = 0.0  # fire as fast as possible

        requests.append(WorkloadRequest(
            request_id=i,
            prompt_text=_generate_dummy_prompt(prompt_tokens),
            prompt_tokens=prompt_tokens,
            max_output_tokens=output_tokens,
            scheduled_at=current_time,
        ))

    return requests


def workload_stats(requests: list[WorkloadRequest]) -> dict:
    """Summarize a workload's distribution characteristics."""
    prompts = [r.prompt_tokens for r in requests]
    outputs = [r.max_output_tokens for r in requests]

    def stats(values: list[int]) -> dict:
        s = sorted(values)
        n = len(s)
        return {
            "count": n,
            "mean": round(sum(s) / n, 1),
            "min": s[0],
            "max": s[-1],
            "p50": s[n // 2],
            "p95": s[int(n * 0.95)],
        }

    return {
        "num_requests": len(requests),
        "prompt_tokens": stats(prompts),
        "output_tokens": stats(outputs),
        "total_tokens": sum(prompts) + sum(outputs),
    }
