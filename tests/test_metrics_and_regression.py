"""Tests for benchmark metrics and regression detection."""

import pytest
import json
import tempfile
from pathlib import Path

from inference_lab.bench.metrics import (
    RequestResult,
    LatencyStats,
    BenchmarkReport,
)
from inference_lab.bench.regression import (
    detect_regressions,
    RegressionConfig,
    RegressionSeverity,
    MetricComparison,
)
from inference_lab.bench.workloads import (
    WorkloadConfig,
    DistributionType,
    generate_workload,
    workload_stats,
)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_from_values_basic(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = LatencyStats.from_values("test", values)
        assert stats.count == 5
        assert stats.min == 10.0
        assert stats.max == 50.0
        assert stats.mean == 30.0
        assert stats.p50 == 30.0

    def test_empty_values(self):
        stats = LatencyStats.from_values("test", [])
        assert stats.count == 0
        assert stats.mean == 0

    def test_single_value(self):
        stats = LatencyStats.from_values("test", [42.0])
        assert stats.count == 1
        assert stats.p50 == 42.0
        assert stats.p99 == 42.0


class TestRequestResult:
    def test_total_tokens(self):
        r = RequestResult(
            request_id="1",
            prompt_tokens=100,
            generated_tokens=50,
            ttft_ms=10.0,
            total_latency_ms=100.0,
            itl_ms=2.0,
        )
        assert r.total_tokens == 150

    def test_decode_throughput(self):
        r = RequestResult(
            request_id="1",
            prompt_tokens=100,
            generated_tokens=50,
            ttft_ms=10.0,
            total_latency_ms=1000.0,
            itl_ms=2.0,
            decode_time_ms=500.0,  # 500ms for 50 tokens = 100 tok/s
        )
        assert r.decode_throughput_tps == 100.0


class TestBenchmarkReport:
    def _make_results(self, n=10) -> list[RequestResult]:
        return [
            RequestResult(
                request_id=str(i),
                prompt_tokens=100,
                generated_tokens=50,
                ttft_ms=10.0 + i,
                total_latency_ms=100.0 + i * 5,
                itl_ms=2.0 + i * 0.1,
            )
            for i in range(n)
        ]

    def test_from_results(self):
        results = self._make_results(10)
        report = BenchmarkReport.from_results(
            results=results,
            model="test-model",
            backend="vllm",
            dtype="float16",
            tp_degree=1,
            gpu="A100",
            timestamp="2024-01-01",
            concurrency=4,
            total_duration_s=10.0,
        )
        assert report.successful_requests == 10
        assert report.failed_requests == 0
        assert report.success_rate == 100.0
        assert report.total_prompt_tokens == 1000
        assert report.total_generated_tokens == 500

    def test_save_and_load(self, tmp_path):
        results = self._make_results(5)
        report = BenchmarkReport.from_results(
            results=results,
            model="test", backend="vllm", dtype="fp16",
            tp_degree=1, gpu="A100", timestamp="2024-01-01",
            concurrency=1, total_duration_s=5.0,
        )

        path = tmp_path / "report.json"
        report.save(path)

        loaded = BenchmarkReport.load(path)
        assert loaded.model == "test"
        assert loaded.successful_requests == 5
        assert loaded.ttft.count == 5


# ---------------------------------------------------------------------------
# Regression detection tests
# ---------------------------------------------------------------------------

def _make_report(
    decode_tps=1000.0,
    ttft_p50=15.0,
    ttft_p95=25.0,
    itl_p50=5.0,
    itl_p95=8.0,
    e2e_p95=200.0,
    success_rate=100.0,
) -> BenchmarkReport:
    """Create a minimal BenchmarkReport for regression testing."""
    return BenchmarkReport(
        model="test",
        backend="vllm",
        dtype="fp16",
        tp_degree=1,
        gpu="A100",
        timestamp="2024-01-01",
        num_requests=100,
        concurrency=4,
        prompt_len_mean=512,
        output_len_mean=128,
        total_duration_s=10.0,
        total_prompt_tokens=51200,
        total_generated_tokens=12800,
        requests_per_second=10.0,
        prompt_throughput_tps=5120.0,
        decode_throughput_tps=decode_tps,
        successful_requests=100,
        failed_requests=0,
        success_rate=success_rate,
        ttft=LatencyStats("ttft", 100, ttft_p50, 3.0, 10.0, 12.0, ttft_p50, 20.0, 22.0, ttft_p95, 30.0, 35.0),
        itl=LatencyStats("itl", 100, itl_p50, 1.0, 3.0, 4.0, itl_p50, 6.0, 7.0, itl_p95, 9.0, 10.0),
        e2e=LatencyStats("e2e", 100, 150.0, 20.0, 100.0, 120.0, 150.0, 180.0, 190.0, e2e_p95, 250.0, 300.0),
    )


class TestRegressionDetection:
    def test_no_regression(self):
        baseline = _make_report()
        current = _make_report()  # identical
        comparisons = detect_regressions(baseline, current)
        assert all(c.severity == RegressionSeverity.OK for c in comparisons)

    def test_throughput_regression(self):
        baseline = _make_report(decode_tps=1000.0)
        current = _make_report(decode_tps=900.0)  # 10% drop
        config = RegressionConfig(decode_throughput_threshold_pct=5.0)
        comparisons = detect_regressions(baseline, current, config)

        tps_comp = [c for c in comparisons if c.name == "decode_throughput_tps"][0]
        assert tps_comp.is_regression
        assert tps_comp.change_pct == pytest.approx(-10.0, abs=0.1)

    def test_latency_regression(self):
        baseline = _make_report(ttft_p95=25.0)
        current = _make_report(ttft_p95=30.0)  # 20% increase
        config = RegressionConfig(ttft_p95_threshold_pct=15.0)
        comparisons = detect_regressions(baseline, current, config)

        ttft_comp = [c for c in comparisons if c.name == "ttft_p95_ms"][0]
        assert ttft_comp.is_regression

    def test_latency_within_threshold(self):
        baseline = _make_report(ttft_p95=25.0)
        current = _make_report(ttft_p95=26.0)  # 4% increase
        config = RegressionConfig(ttft_p95_threshold_pct=15.0)
        comparisons = detect_regressions(baseline, current, config)

        ttft_comp = [c for c in comparisons if c.name == "ttft_p95_ms"][0]
        assert not ttft_comp.is_regression

    def test_improvement_not_flagged(self):
        baseline = _make_report(decode_tps=1000.0)
        current = _make_report(decode_tps=1200.0)  # 20% improvement
        comparisons = detect_regressions(baseline, current)

        tps_comp = [c for c in comparisons if c.name == "decode_throughput_tps"][0]
        assert tps_comp.severity == RegressionSeverity.OK

    def test_warning_half_threshold(self):
        baseline = _make_report(decode_tps=1000.0)
        current = _make_report(decode_tps=965.0)  # 3.5% drop, threshold 5%
        config = RegressionConfig(decode_throughput_threshold_pct=5.0)
        comparisons = detect_regressions(baseline, current, config)

        tps_comp = [c for c in comparisons if c.name == "decode_throughput_tps"][0]
        assert tps_comp.severity == RegressionSeverity.WARNING


# ---------------------------------------------------------------------------
# Workload tests
# ---------------------------------------------------------------------------

class TestWorkloadGeneration:
    def test_fixed_distribution(self):
        config = WorkloadConfig(
            num_requests=10,
            prompt_distribution=DistributionType.FIXED,
            prompt_fixed=512,
            output_distribution=DistributionType.FIXED,
            output_fixed=128,
        )
        workload = generate_workload(config)
        assert len(workload) == 10
        assert all(w.prompt_tokens == 512 for w in workload)
        assert all(w.max_output_tokens == 128 for w in workload)

    def test_uniform_distribution_bounds(self):
        config = WorkloadConfig(
            num_requests=100,
            prompt_distribution=DistributionType.UNIFORM,
            prompt_min=100,
            prompt_max=200,
        )
        workload = generate_workload(config)
        for w in workload:
            assert 100 <= w.prompt_tokens <= 200

    def test_deterministic_with_seed(self):
        config = WorkloadConfig(num_requests=50, seed=42)
        w1 = generate_workload(config)
        w2 = generate_workload(config)
        assert [w.prompt_tokens for w in w1] == [w.prompt_tokens for w in w2]

    def test_poisson_arrivals(self):
        config = WorkloadConfig(num_requests=20, arrival_rate=10.0)
        workload = generate_workload(config)
        # Times should be monotonically increasing
        times = [w.scheduled_at for w in workload]
        assert times == sorted(times)
        assert times[-1] > 0  # not all zero

    def test_workload_stats(self):
        config = WorkloadConfig(num_requests=50)
        workload = generate_workload(config)
        stats = workload_stats(workload)
        assert stats["num_requests"] == 50
        assert stats["prompt_tokens"]["count"] == 50
        assert stats["total_tokens"] > 0
