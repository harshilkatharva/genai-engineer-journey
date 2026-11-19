"""Scheduling and concurrency coordination package for parallel research workers."""

from .scheduler import ParallelResearchScheduler
from .validator import WorkerOutputValidator

__all__ = [
    "ParallelResearchScheduler",
    "WorkerOutputValidator",
]
