"""Core package: configuration, exceptions, and run budget management."""

from self_improving_report_writer.core.budget import BudgetSummary, RunBudget
from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.core.exceptions import (
    BudgetExceededError,
    EvaluationRateLimitError,
    GenerationError,
    JudgingError,
    ReflectionError,
    ReportWriterError,
    RetrievalError,
    VerificationError,
)
from self_improving_report_writer.core.rate_limit_controller import (
    EvaluationRateLimitController,
    extract_retry_delay,
    get_shared_rate_limit_controller,
    is_rate_limit_error,
    reset_shared_rate_limit_controller,
)

__all__ = [
    "BudgetExceededError",
    "BudgetSummary",
    "EvaluationRateLimitController",
    "EvaluationRateLimitError",
    "GenerationError",
    "JudgingError",
    "ReflectionError",
    "ReportWriterConfig",
    "ReportWriterError",
    "RetrievalError",
    "RunBudget",
    "VerificationError",
    "extract_retry_delay",
    "get_default_config",
    "get_shared_rate_limit_controller",
    "is_rate_limit_error",
    "reset_shared_rate_limit_controller",
]
