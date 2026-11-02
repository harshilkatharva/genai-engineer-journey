"""Benchmark and evaluation suite for Self-Improving Report Writer."""

from .blinded_judge import BlindedJudge
from .dataset import BenchmarkTopic, get_benchmark_dataset, get_topic_by_id
from .metrics import (
    ApproachSummaryMetrics,
    ComparativeEvaluationSummary,
    compute_comparative_summary,
)
from .runners import (
    ApproachRunResult,
    ApproachType,
    run_approach_a,
    run_approach_b,
    run_approach_c,
)

__all__ = [
    "ApproachRunResult",
    "ApproachSummaryMetrics",
    "ApproachType",
    "BenchmarkTopic",
    "BlindedJudge",
    "ComparativeEvaluationSummary",
    "compute_comparative_summary",
    "get_benchmark_dataset",
    "get_topic_by_id",
    "run_approach_a",
    "run_approach_b",
    "run_approach_c",
]
