"""Benchmark and evaluation suite for Self-Improving Report Writer."""

from self_improving_report_writer.core.rate_limit_controller import (
    EvaluationRateLimitController,
    extract_retry_delay,
    get_shared_rate_limit_controller,
    is_rate_limit_error,
    reset_shared_rate_limit_controller,
)

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
    "EvaluationRateLimitController",
    "compute_comparative_summary",
    "extract_retry_delay",
    "get_benchmark_dataset",
    "get_shared_rate_limit_controller",
    "get_topic_by_id",
    "is_rate_limit_error",
    "reset_shared_rate_limit_controller",
    "run_approach_a",
    "run_approach_b",
    "run_approach_c",
]
