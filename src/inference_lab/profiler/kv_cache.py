"""KV cache memory estimation and fragmentation analysis.

Computes exact KV cache memory requirements for transformer models across
different configurations (sequence lengths, batch sizes, dtypes, parallelism).
This is the single most important calculation in LLM inference capacity planning.

The KV cache stores key and value tensors for all previous tokens so they don't
need to be recomputed during autoregressive decoding. For each token position,
we store:
    - K tensor: [num_kv_heads, head_dim] per layer
    - V tensor: [num_kv_heads, head_dim] per layer

Total KV cache memory per sequence:
    num_layers × 2 (K+V) × num_kv_heads × head_dim × seq_len × dtype_bytes

With PagedAttention (vLLM), the KV cache is allocated in fixed-size blocks
rather than contiguous per-sequence buffers. This eliminates fragmentation
from variable-length sequences but introduces block-level waste when sequences
don't fill their last block.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

from tabulate import tabulate


# ---------------------------------------------------------------------------
# Model configurations for common architectures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Transformer model parameters relevant to KV cache sizing."""
    name: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int          # GQA: fewer KV heads than query heads
    hidden_size: int
    head_dim: int              # usually hidden_size // num_attention_heads
    max_position_embeddings: int
    vocab_size: int = 0        # for weight memory estimation

    @property
    def gqa_ratio(self) -> int:
        """Number of query heads per KV head (Grouped Query Attention)."""
        return self.num_attention_heads // self.num_kv_heads

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_attention_heads

    @property
    def is_mqa(self) -> bool:
        return self.num_kv_heads == 1


# Pre-defined configs for popular models
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "meta-llama/Llama-3.1-8B-Instruct": ModelConfig(
        name="Llama-3.1-8B",
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,           # GQA with 4:1 ratio
        hidden_size=4096,
        head_dim=128,
        max_position_embeddings=131072,
        vocab_size=128256,
    ),
    "meta-llama/Llama-3.1-70B-Instruct": ModelConfig(
        name="Llama-3.1-70B",
        num_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,           # GQA with 8:1 ratio
        hidden_size=8192,
        head_dim=128,
        max_position_embeddings=131072,
        vocab_size=128256,
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": ModelConfig(
        name="Mistral-7B-v0.3",
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,           # GQA
        hidden_size=4096,
        head_dim=128,
        max_position_embeddings=32768,
        vocab_size=32768,
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        name="Qwen2.5-7B",
        num_layers=28,
        num_attention_heads=28,
        num_kv_heads=4,           # GQA with 7:1 ratio
        hidden_size=3584,
        head_dim=128,
        max_position_embeddings=131072,
        vocab_size=152064,
    ),
    "google/gemma-2-9b-it": ModelConfig(
        name="Gemma-2-9B",
        num_layers=42,
        num_attention_heads=16,
        num_kv_heads=8,
        hidden_size=3584,
        head_dim=256,
        max_position_embeddings=8192,
        vocab_size=256000,
    ),
}

DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,  # FP8 on H100+
    "int8": 1,
}


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

@dataclass
class KVCacheEstimate:
    """Memory estimate for a single configuration point."""
    model_name: str
    seq_len: int
    batch_size: int
    dtype: str
    num_kv_heads: int
    num_layers: int
    head_dim: int
    tp_degree: int

    # Computed fields
    per_token_bytes: int = 0
    per_sequence_bytes: int = 0
    total_bytes: int = 0
    total_gib: float = 0.0

    # PagedAttention fields
    block_size: int = 16
    blocks_per_sequence: int = 0
    wasted_bytes_per_sequence: int = 0
    fragmentation_pct: float = 0.0

    def __post_init__(self) -> None:
        dtype_bytes = DTYPE_BYTES[self.dtype]

        # KV heads are split across TP ranks
        effective_kv_heads = self.num_kv_heads // self.tp_degree

        # Per-token: layers × 2(K+V) × kv_heads × head_dim × dtype_bytes
        self.per_token_bytes = (
            self.num_layers * 2 * effective_kv_heads * self.head_dim * dtype_bytes
        )

        self.per_sequence_bytes = self.per_token_bytes * self.seq_len
        self.total_bytes = self.per_sequence_bytes * self.batch_size
        self.total_gib = self.total_bytes / (1024 ** 3)

        # PagedAttention block analysis
        self.blocks_per_sequence = math.ceil(self.seq_len / self.block_size)
        tokens_in_last_block = self.seq_len % self.block_size
        wasted_tokens = (self.block_size - tokens_in_last_block) if tokens_in_last_block else 0
        self.wasted_bytes_per_sequence = wasted_tokens * self.per_token_bytes
        allocated = self.blocks_per_sequence * self.block_size * self.per_token_bytes
        self.fragmentation_pct = (self.wasted_bytes_per_sequence / allocated * 100) if allocated else 0.0


def estimate_kv_cache(
    model: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    dtype: str = "float16",
    tp_degree: int = 1,
    block_size: int = 16,
) -> KVCacheEstimate:
    """Estimate KV cache memory for a given configuration.

    Args:
        model: Model architecture config.
        seq_len: Total sequence length (prompt + generated tokens).
        batch_size: Number of concurrent sequences.
        dtype: Data type for KV cache storage.
        tp_degree: Tensor parallelism degree (splits KV heads across GPUs).
        block_size: PagedAttention block size in tokens.

    Returns:
        KVCacheEstimate with detailed memory breakdown.

    Raises:
        ValueError: If tp_degree doesn't evenly divide num_kv_heads.
    """
    if model.num_kv_heads % tp_degree != 0:
        raise ValueError(
            f"tp_degree={tp_degree} must evenly divide "
            f"num_kv_heads={model.num_kv_heads}"
        )
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype '{dtype}'. Supported: {list(DTYPE_BYTES.keys())}")

    return KVCacheEstimate(
        model_name=model.name,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
        num_kv_heads=model.num_kv_heads,
        num_layers=model.num_layers,
        head_dim=model.head_dim,
        tp_degree=tp_degree,
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# GPU capacity analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_gib: float
    memory_bandwidth_gbps: float  # GB/s
    flops_fp16_tflops: float

GPU_SPECS: dict[str, GPUSpec] = {
    "A100-80GB": GPUSpec("A100-80GB", 80.0, 2039.0, 312.0),
    "A100-40GB": GPUSpec("A100-40GB", 40.0, 1555.0, 312.0),
    "H100-80GB": GPUSpec("H100-80GB", 80.0, 3350.0, 989.0),
    "L40S":      GPUSpec("L40S", 48.0, 864.0, 362.0),
    "RTX-4090":  GPUSpec("RTX-4090", 24.0, 1008.0, 330.0),
    "RTX-3090":  GPUSpec("RTX-3090", 24.0, 936.0, 142.0),
}


@dataclass
class CapacityAnalysis:
    """How many concurrent sequences a GPU can serve."""
    gpu: str
    model_name: str
    gpu_memory_gib: float
    model_weight_gib: float
    available_for_kv_gib: float
    max_sequences: int
    seq_len: int
    dtype: str
    kv_per_sequence_gib: float
    utilization_pct: float


def analyze_capacity(
    model: ModelConfig,
    gpu_name: str,
    seq_len: int,
    dtype: str = "float16",
    tp_degree: int = 1,
    model_weight_gib: Optional[float] = None,
    overhead_pct: float = 0.10,
) -> CapacityAnalysis:
    """Compute how many concurrent sequences fit on a GPU.

    Args:
        model: Model config.
        gpu_name: Key into GPU_SPECS.
        seq_len: Sequence length for KV cache.
        dtype: KV cache dtype.
        tp_degree: Tensor parallelism degree.
        model_weight_gib: Override for model weight size. If None, estimated
            from hidden_size × num_layers (rough approximation).
        overhead_pct: Fraction of GPU memory reserved for activations, CUDA
            context, framework overhead, etc.
    """
    gpu = GPU_SPECS[gpu_name]

    # Rough model weight estimate if not provided
    # Each layer: ~12 * hidden_size^2 params (attn: 4h^2, mlp: 8h^2)
    if model_weight_gib is None:
        dtype_bytes = DTYPE_BYTES.get(dtype, 2)
        params_per_layer = 12 * model.hidden_size ** 2
        total_params = params_per_layer * model.num_layers
        model_weight_gib = (total_params * dtype_bytes) / (1024 ** 3)

    # Per-GPU weight with TP
    weight_per_gpu = model_weight_gib / tp_degree

    # Available memory for KV cache
    overhead = gpu.memory_gib * overhead_pct
    available = gpu.memory_gib - weight_per_gpu - overhead

    # Single-sequence KV cache
    est = estimate_kv_cache(model, seq_len, batch_size=1, dtype=dtype, tp_degree=tp_degree)
    kv_per_seq_gib = est.total_gib

    max_seqs = int(available / kv_per_seq_gib) if kv_per_seq_gib > 0 else 0
    used = max_seqs * kv_per_seq_gib
    utilization = (used / available * 100) if available > 0 else 0.0

    return CapacityAnalysis(
        gpu=gpu_name,
        model_name=model.name,
        gpu_memory_gib=gpu.memory_gib,
        model_weight_gib=weight_per_gpu,
        available_for_kv_gib=round(available, 2),
        max_sequences=max_seqs,
        seq_len=seq_len,
        dtype=dtype,
        kv_per_sequence_gib=round(kv_per_seq_gib, 4),
        utilization_pct=round(utilization, 1),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_bytes(b: int) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.2f} GiB"
    if b >= 1024 ** 2:
        return f"{b / 1024**2:.2f} MiB"
    if b >= 1024:
        return f"{b / 1024:.2f} KiB"
    return f"{b} B"


def run_analysis(args: argparse.Namespace) -> None:
    model_key = args.model
    if model_key not in MODEL_REGISTRY:
        print(f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model = MODEL_REGISTRY[model_key]
    seq_lens = args.seq_lens
    batch_sizes = args.batch_sizes
    dtype = args.dtype
    tp = args.tp_degree

    print(f"\n{'='*70}")
    print(f"  KV Cache Analysis: {model.name}")
    print(f"  Layers: {model.num_layers} | KV Heads: {model.num_kv_heads} "
          f"(GQA {model.gqa_ratio}:1) | Head dim: {model.head_dim}")
    print(f"  Dtype: {dtype} | TP: {tp}")
    print(f"{'='*70}\n")

    # Per-token cost
    est_1 = estimate_kv_cache(model, seq_len=1, batch_size=1, dtype=dtype, tp_degree=tp)
    print(f"  Per-token KV cache cost: {format_bytes(est_1.per_token_bytes)}")
    print()

    # Matrix: seq_len × batch_size
    headers = ["seq_len"] + [f"bs={bs}" for bs in batch_sizes] + ["frag %"]
    rows = []

    for sl in seq_lens:
        row = [f"{sl:,}"]
        for bs in batch_sizes:
            est = estimate_kv_cache(model, sl, bs, dtype, tp)
            row.append(format_bytes(est.total_bytes))
        # Fragmentation is batch-independent (per-sequence property)
        est_frag = estimate_kv_cache(model, sl, 1, dtype, tp)
        row.append(f"{est_frag.fragmentation_pct:.1f}%")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="right"))

    # GPU capacity analysis
    print(f"\n{'─'*70}")
    print(f"  GPU Capacity (max concurrent sequences at seq_len={seq_lens[-1]:,})")
    print(f"{'─'*70}\n")

    cap_headers = ["GPU", "VRAM", "Weights", "Avail for KV", "Max Seqs", "KV/Seq"]
    cap_rows = []

    for gpu_name in ["RTX-4090", "A100-40GB", "A100-80GB", "H100-80GB"]:
        cap = analyze_capacity(model, gpu_name, seq_lens[-1], dtype, tp)
        cap_rows.append([
            gpu_name,
            f"{cap.gpu_memory_gib:.0f} GiB",
            f"{cap.model_weight_gib:.1f} GiB",
            f"{cap.available_for_kv_gib:.1f} GiB",
            str(cap.max_sequences),
            f"{cap.kv_per_sequence_gib:.4f} GiB",
        ])

    print(tabulate(cap_rows, headers=cap_headers, tablefmt="simple", stralign="right"))
    print()

    # Export if requested
    if args.output:
        results = []
        for sl in seq_lens:
            for bs in batch_sizes:
                est = estimate_kv_cache(model, sl, bs, dtype, tp)
                results.append(asdict(est))
        with open(args.output, "w") as f:
            json.dump({"model": model.name, "config": asdict(model), "estimates": results}, f, indent=2)
        print(f"  Results saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate KV cache memory for LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help=f"Model identifier. Available: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to analyze",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32],
        help="Batch sizes to analyze",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=list(DTYPE_BYTES.keys()),
        help="Data type for KV cache storage",
    )
    parser.add_argument(
        "--tp-degree", type=int, default=1,
        help="Tensor parallelism degree",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results",
    )
    run_analysis(parser.parse_args())


if __name__ == "__main__":
    main()
