from __future__ import annotations


class EvaluationError(Exception):
    """Base exception for all evaluation-related errors."""


class EvaluationDatasetError(EvaluationError):
    """Raised when an evaluation dataset is invalid, missing, or malformed."""


class MetricComputationError(EvaluationError):
    """Raised when an error occurs during metric calculation."""


class JudgeEvaluationError(EvaluationError):
    """Raised when a model-graded evaluation fails or returns an unparseable verdict."""
