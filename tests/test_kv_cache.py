"""Tests for KV cache memory estimation.

These tests verify the math by hand-computing expected values
for known model configurations. No GPU required.
"""

import pytest

from inference_lab.profiler.kv_cache import (
    ModelConfig,
    KVCacheEstimate,
    estimate_kv_cache,
    analyze_capacity,
    MODEL_REGISTRY,
    DTYPE_BYTES,
)


# ---------------------------------------------------------------------------
# Hand-computed reference model for verification
# ---------------------------------------------------------------------------

TINY_MODEL = ModelConfig(
    name="tiny-test",
    num_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,       # GQA 2:1
    hidden_size=256,
    head_dim=64,
    max_position_embeddings=1024,
)


class TestPerTokenCost:
    """Verify per-token KV cache bytes match hand computation."""

    def test_tiny_model_fp16(self):
        est = estimate_kv_cache(TINY_MODEL, seq_len=1, batch_size=1, dtype="float16")
        # per-token = layers × 2 × kv_heads × head_dim × dtype_bytes
        # = 2 × 2 × 2 × 64 × 2 = 1024 bytes
        assert est.per_token_bytes == 1024

    def test_tiny_model_fp32(self):
        est = estimate_kv_cache(TINY_MODEL, seq_len=1, batch_size=1, dtype="float32")
        # = 2 × 2 × 2 × 64 × 4 = 2048 bytes
        assert est.per_token_bytes == 2048

    def test_tiny_model_fp8(self):
        est = estimate_kv_cache(TINY_MODEL, seq_len=1, batch_size=1, dtype="float8_e4m3fn")
        # = 2 × 2 × 2 × 64 × 1 = 512 bytes
        assert est.per_token_bytes == 512

    def test_llama3_8b_per_token(self):
        """Verify Llama 3.1 8B per-token cost."""
        model = MODEL_REGISTRY["meta-llama/Llama-3.1-8B-Instruct"]
        est = estimate_kv_cache(model, seq_len=1, batch_size=1, dtype="float16")
        # 32 layers × 2 × 8 kv_heads × 128 head_dim × 2 bytes
        # = 32 × 2 × 8 × 128 × 2 = 131072 bytes = 128 KiB
        assert est.per_token_bytes == 131072


class TestSequenceScaling:
    """Verify memory scales linearly with sequence length and batch size."""

    def test_linear_with_seq_len(self):
        est_512 = estimate_kv_cache(TINY_MODEL, seq_len=512, batch_size=1)
        est_1024 = estimate_kv_cache(TINY_MODEL, seq_len=1024, batch_size=1)
        assert est_1024.per_sequence_bytes == est_512.per_sequence_bytes * 2

    def test_linear_with_batch_size(self):
        est_bs1 = estimate_kv_cache(TINY_MODEL, seq_len=512, batch_size=1)
        est_bs8 = estimate_kv_cache(TINY_MODEL, seq_len=512, batch_size=8)
        assert est_bs8.total_bytes == est_bs1.total_bytes * 8

    def test_seq_times_batch(self):
        est = estimate_kv_cache(TINY_MODEL, seq_len=100, batch_size=10)
        assert est.total_bytes == est.per_token_bytes * 100 * 10


class TestTensorParallelism:
    """Verify KV cache splits correctly across TP ranks."""

    def test_tp2_halves_memory(self):
        est_tp1 = estimate_kv_cache(TINY_MODEL, seq_len=1024, batch_size=1, tp_degree=1)
        est_tp2 = estimate_kv_cache(TINY_MODEL, seq_len=1024, batch_size=1, tp_degree=2)
        assert est_tp2.total_bytes == est_tp1.total_bytes // 2

    def test_tp_must_divide_kv_heads(self):
        with pytest.raises(ValueError, match="must evenly divide"):
            estimate_kv_cache(TINY_MODEL, seq_len=1024, tp_degree=3)  # 2 kv_heads % 3 != 0


class TestPagedAttention:
    """Verify block-level fragmentation calculations."""

    def test_no_fragmentation_when_aligned(self):
        est = estimate_kv_cache(TINY_MODEL, seq_len=128, block_size=16)
        # 128 / 16 = 8 blocks, perfectly aligned
        assert est.blocks_per_sequence == 8
        assert est.wasted_bytes_per_sequence == 0
        assert est.fragmentation_pct == 0.0

    def test_fragmentation_when_unaligned(self):
        est = estimate_kv_cache(TINY_MODEL, seq_len=100, block_size=16)
        # ceil(100/16) = 7 blocks = 112 slots, 12 wasted
        assert est.blocks_per_sequence == 7
        assert est.wasted_bytes_per_sequence == 12 * est.per_token_bytes
        expected_frag = 12 / 112 * 100
        assert abs(est.fragmentation_pct - expected_frag) < 0.01

    def test_single_token_fragmentation(self):
        est = estimate_kv_cache(TINY_MODEL, seq_len=1, block_size=16)
        # 1 block, 15 wasted tokens
        assert est.blocks_per_sequence == 1
        expected_frag = 15 / 16 * 100
        assert abs(est.fragmentation_pct - expected_frag) < 0.01


class TestGPUCapacity:
    """Verify capacity analysis makes reasonable estimates."""

    def test_rtx4090_llama8b(self):
        model = MODEL_REGISTRY["meta-llama/Llama-3.1-8B-Instruct"]
        cap = analyze_capacity(model, "RTX-4090", seq_len=2048, dtype="float16")
        # RTX 4090: 24 GiB. ~16 GiB for 8B model weights (fp16).
        # Available ~5-6 GiB. KV per seq ~0.25 GiB at 2048.
        # Should fit ~20 sequences.
        assert cap.max_sequences > 0
        assert cap.available_for_kv_gib > 0
        assert cap.kv_per_sequence_gib > 0

    def test_a100_fits_more(self):
        model = MODEL_REGISTRY["meta-llama/Llama-3.1-8B-Instruct"]
        cap_4090 = analyze_capacity(model, "RTX-4090", seq_len=2048)
        cap_a100 = analyze_capacity(model, "A100-80GB", seq_len=2048)
        assert cap_a100.max_sequences > cap_4090.max_sequences


class TestInvalidInputs:
    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            estimate_kv_cache(TINY_MODEL, seq_len=1024, dtype="int4")

    def test_all_registered_models_valid(self):
        """Smoke test: estimate works for every registered model."""
        for name, model in MODEL_REGISTRY.items():
            est = estimate_kv_cache(model, seq_len=1024, batch_size=1)
            assert est.per_token_bytes > 0
            assert est.total_bytes > 0
