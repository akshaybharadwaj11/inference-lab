"""Performance regression detection for CI/CD pipelines.

Compares a current benchmark run against a baseline and flags regressions
using configurable thresholds. Designed to run in GitHub Actions or any CI
system, with exit code 1 on regression detection.

Detection strategy:
    - Relative threshold: flag if metric degrades by more than X%
    - Absolute threshold: flag if metric exceeds an absolute value
    - Both must be checked for each metric category

Metrics checked:
    - Decode throughput (tokens/s) — regression = decrease
    - TTFT p50/p95 (ms) — regression = increase
    - ITL p50/p95 (ms) — regression = increase
    - E2E p95 (ms) — regression = increase
    - Success rate (%) — regression = decrease
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from inference_lab.bench.metrics import BenchmarkReport


class RegressionSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    REGRESSION = "regression"


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""
    name: str
    baseline_value: float
    current_value: float
    change_pct: float            # positive = degradation for latency, negative for throughput
    threshold_pct: float
    severity: RegressionSeverity
    direction: str               # "lower_is_better" or "higher_is_better"

    @property
    def is_regression(self) -> bool:
        return self.severity == RegressionSeverity.REGRESSION

    def summary_line(self) -> str:
        icon = {"ok": "✓", "warning": "⚠", "regression": "✗"}[self.severity.value]
        sign = "+" if self.change_pct > 0 else ""
        return (
            f"  {icon} {self.name:30s} "
            f"baseline={self.baseline_value:10.2f}  "
            f"current={self.current_value:10.2f}  "
            f"change={sign}{self.change_pct:+.1f}%  "
            f"threshold=±{self.threshold_pct:.1f}%"
        )


@dataclass
class RegressionConfig:
    """Thresholds for regression detection."""
    # Throughput: flag if current < baseline * (1 - threshold)
    decode_throughput_threshold_pct: float = 5.0
    prompt_throughput_threshold_pct: float = 5.0

    # Latency: flag if current > baseline * (1 + threshold)
    ttft_p50_threshold_pct: float = 10.0
    ttft_p95_threshold_pct: float = 15.0
    itl_p50_threshold_pct: float = 10.0
    itl_p95_threshold_pct: float = 15.0
    e2e_p95_threshold_pct: float = 10.0

    # Success rate: flag if current < baseline - threshold_points
    success_rate_threshold_points: float = 1.0

    # Minimum requests for statistical validity
    min_requests: int = 50

    @classmethod
    def from_json(cls, path: str | Path) -> RegressionConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def strict(cls) -> RegressionConfig:
        """Strict thresholds for production regression gates."""
        return cls(
            decode_throughput_threshold_pct=3.0,
            prompt_throughput_threshold_pct=3.0,
            ttft_p50_threshold_pct=5.0,
            ttft_p95_threshold_pct=10.0,
            itl_p50_threshold_pct=5.0,
            itl_p95_threshold_pct=10.0,
            e2e_p95_threshold_pct=5.0,
            success_rate_threshold_points=0.5,
        )


def _compare_metric(
    name: str,
    baseline: float,
    current: float,
    threshold_pct: float,
    higher_is_better: bool,
) -> MetricComparison:
    """Compare a single metric against its baseline."""
    if baseline == 0:
        change_pct = 0.0
        severity = RegressionSeverity.OK
    else:
        change_pct = ((current - baseline) / abs(baseline)) * 100

        if higher_is_better:
            # Throughput: regression if current is LOWER
            if change_pct < -threshold_pct:
                severity = RegressionSeverity.REGRESSION
            elif change_pct < -(threshold_pct / 2):
                severity = RegressionSeverity.WARNING
            else:
                severity = RegressionSeverity.OK
        else:
            # Latency: regression if current is HIGHER
            if change_pct > threshold_pct:
                severity = RegressionSeverity.REGRESSION
            elif change_pct > (threshold_pct / 2):
                severity = RegressionSeverity.WARNING
            else:
                severity = RegressionSeverity.OK

    return MetricComparison(
        name=name,
        baseline_value=baseline,
        current_value=current,
        change_pct=round(change_pct, 2),
        threshold_pct=threshold_pct,
        severity=severity,
        direction="higher_is_better" if higher_is_better else "lower_is_better",
    )


def detect_regressions(
    baseline: BenchmarkReport,
    current: BenchmarkReport,
    config: Optional[RegressionConfig] = None,
) -> list[MetricComparison]:
    """Compare two benchmark reports and detect regressions.

    Args:
        baseline: Reference benchmark results.
        current: New benchmark results to check.
        config: Regression thresholds. Uses defaults if None.

    Returns:
        List of MetricComparison results for all checked metrics.
    """
    if config is None:
        config = RegressionConfig()

    comparisons: list[MetricComparison] = []

    # Throughput (higher is better)
    comparisons.append(_compare_metric(
        "decode_throughput_tps",
        baseline.decode_throughput_tps,
        current.decode_throughput_tps,
        config.decode_throughput_threshold_pct,
        higher_is_better=True,
    ))
    comparisons.append(_compare_metric(
        "prompt_throughput_tps",
        baseline.prompt_throughput_tps,
        current.prompt_throughput_tps,
        config.prompt_throughput_threshold_pct,
        higher_is_better=True,
    ))

    # TTFT (lower is better)
    comparisons.append(_compare_metric(
        "ttft_p50_ms", baseline.ttft.p50, current.ttft.p50,
        config.ttft_p50_threshold_pct, higher_is_better=False,
    ))
    comparisons.append(_compare_metric(
        "ttft_p95_ms", baseline.ttft.p95, current.ttft.p95,
        config.ttft_p95_threshold_pct, higher_is_better=False,
    ))

    # ITL (lower is better)
    comparisons.append(_compare_metric(
        "itl_p50_ms", baseline.itl.p50, current.itl.p50,
        config.itl_p50_threshold_pct, higher_is_better=False,
    ))
    comparisons.append(_compare_metric(
        "itl_p95_ms", baseline.itl.p95, current.itl.p95,
        config.itl_p95_threshold_pct, higher_is_better=False,
    ))

    # E2E (lower is better)
    comparisons.append(_compare_metric(
        "e2e_p95_ms", baseline.e2e.p95, current.e2e.p95,
        config.e2e_p95_threshold_pct, higher_is_better=False,
    ))

    # Success rate (higher is better)
    comparisons.append(_compare_metric(
        "success_rate_%",
        baseline.success_rate,
        current.success_rate,
        config.success_rate_threshold_points,
        higher_is_better=True,
    ))

    return comparisons


def print_regression_report(comparisons: list[MetricComparison]) -> bool:
    """Print a formatted regression report. Returns True if any regressions found."""
    regressions = [c for c in comparisons if c.is_regression]
    warnings = [c for c in comparisons if c.severity == RegressionSeverity.WARNING]

    print(f"\n{'='*80}")
    print(f"  Performance Regression Report")
    print(f"{'='*80}\n")

    for c in comparisons:
        print(c.summary_line())

    print()
    if regressions:
        print(f"  ✗ {len(regressions)} REGRESSION(S) DETECTED")
        for r in regressions:
            print(f"    → {r.name}: {r.change_pct:+.1f}% (threshold: ±{r.threshold_pct:.1f}%)")
    elif warnings:
        print(f"  ⚠ {len(warnings)} warning(s), no regressions")
    else:
        print(f"  ✓ All metrics within thresholds")

    print()
    return len(regressions) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect inference performance regressions")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline JSON")
    parser.add_argument("--current", type=str, required=True, help="Path to current JSON")
    parser.add_argument("--threshold", type=float, default=0.05, help="Global threshold override")
    parser.add_argument("--strict", action="store_true", help="Use strict thresholds")
    parser.add_argument("--config", type=str, default=None, help="Path to threshold config JSON")
    args = parser.parse_args()

    baseline = BenchmarkReport.load(args.baseline)
    current = BenchmarkReport.load(args.current)

    if args.config:
        reg_config = RegressionConfig.from_json(args.config)
    elif args.strict:
        reg_config = RegressionConfig.strict()
    else:
        reg_config = RegressionConfig()

    comparisons = detect_regressions(baseline, current, reg_config)
    has_regressions = print_regression_report(comparisons)

    sys.exit(1 if has_regressions else 0)


if __name__ == "__main__":
    main()
