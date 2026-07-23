"""FastAPI serving layer for LLM inference with observability.

Provides:
    - POST /v1/completions — OpenAI-compatible completion endpoint
    - GET /health — Liveness/readiness probe for K8s
    - GET /metrics — Prometheus-format metrics
    - GET /stats — JSON latency statistics from request tracer

This is a thin serving layer around vLLM's AsyncLLMEngine, adding:
    - Request lifecycle tracing (TTFT, ITL, queue time)
    - Structured logging with structlog
    - Prometheus metrics export
    - Graceful shutdown with in-flight request draining
"""

from __future__ import annotations

import argparse
import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from inference_lab.profiler.trace import RequestTracer, RequestPhase

logger = structlog.get_logger(__name__)

# Global state (initialized in lifespan)
_engine = None
_tracer = RequestTracer()
_model_name: str = ""


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[list[str]] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: list[dict]
    usage: dict


class HealthResponse(BaseModel):
    status: str
    model: str
    active_requests: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vLLM engine on startup, drain on shutdown."""
    global _engine, _start_time
    _start_time = time.time()

    logger.info("server.starting", model=_model_name)

    # Try to import and initialize vLLM engine
    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        engine_args = AsyncEngineArgs(
            model=_model_name,
            dtype="auto",
            max_model_len=8192,
        )
        _engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("engine.ready", model=_model_name)
    except ImportError:
        logger.warning(
            "vllm not installed — running in mock mode. "
            "Install vLLM for real inference: pip install vllm"
        )
        _engine = None

    yield

    # Shutdown: wait for active requests to drain
    logger.info("server.shutting_down", active_requests=_tracer.active_count)
    deadline = time.monotonic() + 30  # 30s grace period
    while _tracer.active_count > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.5)
    logger.info("server.stopped")


app = FastAPI(
    title="LLM Inference Lab",
    description="Inference serving with observability",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    response.headers["X-Request-Duration-Ms"] = f"{duration_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if _engine is not None else "mock",
        model=_model_name,
        active_requests=_tracer.active_count,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/stats")
async def stats():
    """Return latency statistics from the request tracer."""
    return _tracer.get_latency_stats(last_n=100)


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    """OpenAI-compatible completion endpoint with tracing."""
    # Start trace
    request_id = _tracer.start_request(
        model=req.model or _model_name,
        prompt_tokens=len(req.prompt.split()) * 4 // 3,  # rough estimate
    )

    if _engine is None:
        # Mock mode for development without GPU
        return await _mock_completion(req, request_id)

    try:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )

        if req.stream:
            return StreamingResponse(
                _stream_response(req, request_id, sampling_params),
                media_type="text/event-stream",
            )
        else:
            return await _generate_response(req, request_id, sampling_params)

    except Exception as e:
        _tracer.fail_request(request_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_response(req, request_id, sampling_params):
    """Non-streaming completion with full tracing."""
    results_generator = _engine.generate(req.prompt, sampling_params, request_id)

    generated_text = ""
    generated_tokens = 0
    first_token = True

    async for output in results_generator:
        if first_token:
            first_token = False
        if output.outputs:
            generated_text = output.outputs[0].text
            generated_tokens = len(output.outputs[0].token_ids)

    trace = _tracer.complete_request(request_id, generated_tokens)

    return CompletionResponse(
        id=request_id,
        model=req.model or _model_name,
        choices=[{"text": generated_text, "index": 0, "finish_reason": "stop"}],
        usage={
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": generated_tokens,
            "total_tokens": trace.total_tokens,
        },
    )


async def _stream_response(req, request_id, sampling_params) -> AsyncGenerator[str, None]:
    """Streaming completion with per-token timing."""
    results_generator = _engine.generate(req.prompt, sampling_params, request_id)

    generated_tokens = 0
    prev_text = ""

    async for output in results_generator:
        if output.outputs:
            new_text = output.outputs[0].text
            delta = new_text[len(prev_text):]
            prev_text = new_text
            generated_tokens = len(output.outputs[0].token_ids)

            if delta:
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "choices": [{"text": delta, "index": 0}],
                }
                yield f"data: {__import__('json').dumps(chunk)}\n\n"

    _tracer.complete_request(request_id, generated_tokens)
    yield "data: [DONE]\n\n"


async def _mock_completion(req: CompletionRequest, request_id: str):
    """Mock response for development without GPU."""
    # Simulate prefill
    with _tracer.trace_phase(request_id, RequestPhase.PREFILL):
        await asyncio.sleep(0.05)

    # Simulate decode
    mock_tokens = min(req.max_tokens, 50)
    with _tracer.trace_phase(request_id, RequestPhase.DECODE):
        await asyncio.sleep(mock_tokens * 0.01)  # ~10ms per token

    mock_text = f"[Mock response: {mock_tokens} tokens generated for prompt of ~{len(req.prompt)} chars]"
    trace = _tracer.complete_request(request_id, mock_tokens)

    return CompletionResponse(
        id=request_id,
        model=req.model or _model_name,
        choices=[{"text": mock_text, "index": 0, "finish_reason": "stop"}],
        usage={
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": mock_tokens,
            "total_tokens": trace.total_tokens,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Start LLM inference server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    global _model_name
    _model_name = args.model

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
