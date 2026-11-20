"""Typed domain exceptions for Parallel Research Team."""

from __future__ import annotations


class ResearchTeamError(Exception):
    """Base exception for all Parallel Research Team errors."""


class BudgetExceededError(ResearchTeamError):
    """Raised when cumulative token, cost, or timeout limits are breached."""


class DecompositionError(ResearchTeamError):
    """Raised when research topic decomposition fails or produces invalid briefs."""


class WorkerExecutionError(ResearchTeamError):
    """Raised when a research worker encounters an unrecoverable execution failure."""


class WorkerTimeoutError(WorkerExecutionError):
    """Raised when a worker exceeds its per-worker wall-clock deadline."""


class WorkerOutputValidationError(ResearchTeamError):
    """Raised when a worker's output fails structural or citation integrity checks."""


class SynthesisError(ResearchTeamError):
    """Raised when final report synthesis from pooled evidence fails."""


class VerificationError(ResearchTeamError):
    """Raised when the synthesized report fails Gate 1 or Gate 2 verification."""


class CheckpointError(ResearchTeamError):
    """Raised when saving, loading, or parsing a checkpoint fails."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a requested checkpoint run ID is not found on disk."""


__all__ = [
    "BudgetExceededError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "DecompositionError",
    "ResearchTeamError",
    "SynthesisError",
    "VerificationError",
    "WorkerExecutionError",
    "WorkerOutputValidationError",
    "WorkerTimeoutError",
]
