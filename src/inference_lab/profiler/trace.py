"""Request lifecycle tracing for LLM inference.

Instruments each phase of a request through the inference pipeline:
    1. Received    → API server receives the request
    2. Queued      → Request enters the scheduler's waiting queue
    3. Prefill     → Prompt tokens are processed (compute-bound)
    4. Decode      → Tokens are generated autoregressively (memory-bound)
    5. Detokenize  → Output IDs are converted back to text
    6. Complete    → Response is sent to the client

This tracer captures timing, token counts, and queue depth at each phase,
enabling bottleneck identification across the serving pipeline.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Optional

import structlog

logger = structlog.get_logger(__name__)


class RequestPhase(str, Enum):
    RECEIVED = "received"
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    DETOKENIZE = "detokenize"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class SpanEvent:
    """A single phase transition in the request lifecycle."""
    phase: RequestPhase
    timestamp: float
    duration_ms: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RequestTrace:
    """Complete trace of a single inference request."""
    request_id: str
    model: str
    created_at: float
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    spans: list[SpanEvent] = field(default_factory=list)

    # Computed latency metrics (populated on completion)
    time_to_first_token_ms: Optional[float] = None  # TTFT
    inter_token_latency_ms: Optional[float] = None   # ITL (avg)
    total_latency_ms: Optional[float] = None
    queue_time_ms: Optional[float] = None
    prefill_time_ms: Optional[float] = None
    decode_time_ms: Optional[float] = None

    def add_span(self, phase: RequestPhase, duration_ms: float | None = None, **metadata: object) -> None:
        self.spans.append(SpanEvent(
            phase=phase,
            timestamp=time.monotonic(),
            duration_ms=duration_ms,
            metadata=metadata,
        ))

    def finalize(self) -> None:
        """Compute derived latency metrics from spans."""
        span_map: dict[RequestPhase, SpanEvent] = {}
        for span in self.spans:
            span_map[span.phase] = span

        if RequestPhase.RECEIVED in span_map and RequestPhase.COMPLETE in span_map:
            self.total_latency_ms = (
                (span_map[RequestPhase.COMPLETE].timestamp - span_map[RequestPhase.RECEIVED].timestamp) * 1000
            )

        if RequestPhase.QUEUED in span_map:
            self.queue_time_ms = span_map[RequestPhase.QUEUED].duration_ms

        if RequestPhase.PREFILL in span_map:
            self.prefill_time_ms = span_map[RequestPhase.PREFILL].duration_ms

        if RequestPhase.DECODE in span_map:
            self.decode_time_ms = span_map[RequestPhase.DECODE].duration_ms

        # TTFT = queue_time + prefill_time
        if self.queue_time_ms is not None and self.prefill_time_ms is not None:
            self.time_to_first_token_ms = self.queue_time_ms + self.prefill_time_ms

        # Average inter-token latency
        if self.decode_time_ms is not None and self.generated_tokens > 1:
            self.inter_token_latency_ms = self.decode_time_ms / (self.generated_tokens - 1)

        self.total_tokens = self.prompt_tokens + self.generated_tokens

    def summary(self) -> dict:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_ms": round(self.time_to_first_token_ms, 2) if self.time_to_first_token_ms else None,
            "itl_ms": round(self.inter_token_latency_ms, 2) if self.inter_token_latency_ms else None,
            "total_ms": round(self.total_latency_ms, 2) if self.total_latency_ms else None,
            "queue_ms": round(self.queue_time_ms, 2) if self.queue_time_ms else None,
            "prefill_ms": round(self.prefill_time_ms, 2) if self.prefill_time_ms else None,
            "decode_ms": round(self.decode_time_ms, 2) if self.decode_time_ms else None,
        }


class RequestTracer:
    """Manages active request traces with phase timing."""

    def __init__(self, max_completed: int = 10000):
        self._active: dict[str, RequestTrace] = {}
        self._completed: list[RequestTrace] = []
        self._max_completed = max_completed

    def start_request(self, model: str, prompt_tokens: int = 0) -> str:
        request_id = str(uuid.uuid4())[:12]
        trace = RequestTrace(
            request_id=request_id,
            model=model,
            created_at=time.time(),
            prompt_tokens=prompt_tokens,
        )
        trace.add_span(RequestPhase.RECEIVED)
        self._active[request_id] = trace

        logger.debug("request.started", request_id=request_id, prompt_tokens=prompt_tokens)
        return request_id

    @contextmanager
    def trace_phase(
        self, request_id: str, phase: RequestPhase, **metadata: object
    ) -> Generator[RequestTrace, None, None]:
        """Context manager that times a request phase.

        Usage:
            with tracer.trace_phase(rid, RequestPhase.PREFILL, tokens=512):
                # ... do prefill work ...
        """
        trace = self._active.get(request_id)
        if trace is None:
            raise KeyError(f"No active request with id={request_id}")

        start = time.monotonic()
        try:
            yield trace
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            trace.add_span(phase, duration_ms=duration_ms, **metadata)
            logger.debug(
                "request.phase",
                request_id=request_id,
                phase=phase.value,
                duration_ms=round(duration_ms, 2),
            )

    def complete_request(self, request_id: str, generated_tokens: int = 0) -> RequestTrace:
        trace = self._active.pop(request_id, None)
        if trace is None:
            raise KeyError(f"No active request with id={request_id}")

        trace.generated_tokens = generated_tokens
        trace.add_span(RequestPhase.COMPLETE)
        trace.finalize()

        self._completed.append(trace)
        if len(self._completed) > self._max_completed:
            self._completed = self._completed[-self._max_completed:]

        logger.info("request.complete", **trace.summary())
        return trace

    def fail_request(self, request_id: str, error: str) -> Optional[RequestTrace]:
        trace = self._active.pop(request_id, None)
        if trace is None:
            return None
        trace.add_span(RequestPhase.ERROR, metadata={"error": error})
        trace.finalize()
        self._completed.append(trace)
        logger.error("request.failed", request_id=request_id, error=error)
        return trace

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def completed_traces(self) -> list[RequestTrace]:
        return list(self._completed)

    def get_latency_stats(self, last_n: int = 100) -> dict:
        """Compute aggregate latency statistics over recent requests."""
        recent = self._completed[-last_n:]
        if not recent:
            return {}

        ttfts = [t.time_to_first_token_ms for t in recent if t.time_to_first_token_ms is not None]
        itls = [t.inter_token_latency_ms for t in recent if t.inter_token_latency_ms is not None]
        totals = [t.total_latency_ms for t in recent if t.total_latency_ms is not None]

        def percentiles(values: list[float]) -> dict:
            if not values:
                return {}
            s = sorted(values)
            n = len(s)
            return {
                "p50": round(s[n // 2], 2),
                "p95": round(s[int(n * 0.95)], 2),
                "p99": round(s[int(n * 0.99)], 2),
                "mean": round(sum(s) / n, 2),
                "min": round(s[0], 2),
                "max": round(s[-1], 2),
            }

        return {
            "sample_size": len(recent),
            "ttft": percentiles(ttfts),
            "itl": percentiles(itls),
            "total_latency": percentiles(totals),
        }
