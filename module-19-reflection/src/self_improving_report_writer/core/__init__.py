"""Core package: configuration, exceptions, and run budget management."""

from self_improving_report_writer.core.budget import BudgetSummary, RunBudget
from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.core.exceptions import (
    BudgetExceededError,
    GenerationError,
    JudgingError,
    ReflectionError,
    ReportWriterError,
    RetrievalError,
    VerificationError,
)

__all__ = [
    "BudgetExceededError",
    "BudgetSummary",
    "GenerationError",
    "JudgingError",
    "ReflectionError",
    "ReportWriterConfig",
    "ReportWriterError",
    "RetrievalError",
    "RunBudget",
    "VerificationError",
    "get_default_config",
]
