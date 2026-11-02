"""Exception hierarchy for Self-Improving Report Writer."""

from __future__ import annotations


class ReportWriterError(Exception):
    """Base exception for all report writer errors."""


class RetrievalError(ReportWriterError):
    """Raised when evidence retrieval fails or yields insufficient sources."""


class GenerationError(ReportWriterError):
    """Raised when structured report generation fails or returns malformed output."""


class VerificationError(ReportWriterError):
    """Raised when deterministic or semantic verification encounters fatal inconsistency."""


class JudgingError(ReportWriterError):
    """Raised when rubric judging fails or returns unparseable evaluation."""


class BudgetExceededError(ReportWriterError):
    """Raised when token, cost, or wall-clock budget is breached."""


class ReflectionError(ReportWriterError):
    """Raised when reflection orchestration encounters unrecoverable failure."""
