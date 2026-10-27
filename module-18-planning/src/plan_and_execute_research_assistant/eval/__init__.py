"""Evaluation package for Module 18 Plan-and-Execute Research Assistant."""

from __future__ import annotations

from .dataset import (
    BENCHMARK_GOALS,
    create_10_goals_dataset,
    get_10_benchmark_goals,
)
from .metrics import (
    BenchmarkSummaryStats,
    ComparativeBenchmarkReport,
    SystemEvalResult,
)
from .runner import (
    deterministic_mock_planned_target,
    deterministic_mock_reactive_target,
    run_comparative_benchmark,
)

__all__ = [
    "BENCHMARK_GOALS",
    "BenchmarkSummaryStats",
    "ComparativeBenchmarkReport",
    "SystemEvalResult",
    "create_10_goals_dataset",
    "deterministic_mock_planned_target",
    "deterministic_mock_reactive_target",
    "get_10_benchmark_goals",
    "run_comparative_benchmark",
]
