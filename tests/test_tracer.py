"""Tests for request lifecycle tracing."""

import time
import pytest

from inference_lab.profiler.trace import (
    RequestTracer,
    RequestPhase,
    RequestTrace,
)


class TestRequestTracer:
    def test_basic_lifecycle(self):
        tracer = RequestTracer()
        rid = tracer.start_request(model="test", prompt_tokens=100)

        with tracer.trace_phase(rid, RequestPhase.QUEUED):
            time.sleep(0.01)

        with tracer.trace_phase(rid, RequestPhase.PREFILL):
            time.sleep(0.02)

        with tracer.trace_phase(rid, RequestPhase.DECODE):
            time.sleep(0.05)

        trace = tracer.complete_request(rid, generated_tokens=50)

        assert trace.prompt_tokens == 100
        assert trace.generated_tokens == 50
        assert trace.total_tokens == 150
        assert trace.total_latency_ms is not None
        assert trace.total_latency_ms > 0
        assert trace.queue_time_ms is not None
        assert trace.prefill_time_ms is not None
        assert trace.decode_time_ms is not None

    def test_ttft_computation(self):
        tracer = RequestTracer()
        rid = tracer.start_request(model="test", prompt_tokens=100)

        with tracer.trace_phase(rid, RequestPhase.QUEUED):
            time.sleep(0.01)

        with tracer.trace_phase(rid, RequestPhase.PREFILL):
            time.sleep(0.02)

        with tracer.trace_phase(rid, RequestPhase.DECODE):
            time.sleep(0.03)

        trace = tracer.complete_request(rid, generated_tokens=20)

        # TTFT = queue + prefill
        assert trace.time_to_first_token_ms is not None
        expected_ttft = trace.queue_time_ms + trace.prefill_time_ms
        assert abs(trace.time_to_first_token_ms - expected_ttft) < 0.1

    def test_itl_computation(self):
        tracer = RequestTracer()
        rid = tracer.start_request(model="test")

        with tracer.trace_phase(rid, RequestPhase.DECODE):
            time.sleep(0.05)  # 50ms for decode

        trace = tracer.complete_request(rid, generated_tokens=10)
        # ITL = decode_time / (generated - 1) = ~50 / 9 ≈ 5.5ms
        assert trace.inter_token_latency_ms is not None
        assert trace.inter_token_latency_ms > 0

    def test_active_count(self):
        tracer = RequestTracer()
        assert tracer.active_count == 0

        rid1 = tracer.start_request(model="test")
        rid2 = tracer.start_request(model="test")
        assert tracer.active_count == 2

        tracer.complete_request(rid1, generated_tokens=10)
        assert tracer.active_count == 1

        tracer.complete_request(rid2, generated_tokens=10)
        assert tracer.active_count == 0

    def test_fail_request(self):
        tracer = RequestTracer()
        rid = tracer.start_request(model="test")
        trace = tracer.fail_request(rid, error="OOM")

        assert trace is not None
        assert tracer.active_count == 0
        assert any(s.phase == RequestPhase.ERROR for s in trace.spans)

    def test_unknown_request_raises(self):
        tracer = RequestTracer()
        with pytest.raises(KeyError):
            tracer.complete_request("nonexistent")

    def test_latency_stats(self):
        tracer = RequestTracer()
        for _ in range(10):
            rid = tracer.start_request(model="test", prompt_tokens=100)
            with tracer.trace_phase(rid, RequestPhase.QUEUED):
                time.sleep(0.001)
            with tracer.trace_phase(rid, RequestPhase.PREFILL):
                time.sleep(0.002)
            with tracer.trace_phase(rid, RequestPhase.DECODE):
                time.sleep(0.005)
            tracer.complete_request(rid, generated_tokens=10)

        stats = tracer.get_latency_stats(last_n=10)
        assert stats["sample_size"] == 10
        assert "ttft" in stats
        assert "p50" in stats["ttft"]
        assert stats["ttft"]["p50"] > 0

    def test_max_completed_eviction(self):
        tracer = RequestTracer(max_completed=5)
        for _ in range(10):
            rid = tracer.start_request(model="test")
            tracer.complete_request(rid, generated_tokens=1)

        assert len(tracer.completed_traces) == 5
