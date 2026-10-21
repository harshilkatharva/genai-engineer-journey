from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...models.evaluation_model import EvalOutput, EvalSample, MetricResult


@runtime_checkable
class Metric(Protocol):
    """Protocol for all evaluation metrics (deterministic and model-graded)."""

    name: str

    async def evaluate(self, sample: EvalSample, output: EvalOutput) -> MetricResult:
        """Compute metric for the given sample and system output."""
        ...
