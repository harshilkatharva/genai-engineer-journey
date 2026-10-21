from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .eval_output import EvalOutput


class MetricResult(BaseModel):
    """Evaluation result for a single metric on a single sample."""

    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool | None = None
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SampleEvalResult(BaseModel):
    """Aggregated evaluation results for a single test sample."""

    sample_id: str | int
    query: str
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    output: EvalOutput
    passed: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    """Statistical summary for an evaluation metric across a dataset."""

    metric_name: str
    mean: float
    p50: float
    p95: float
    min: float
    max: float
    pass_rate: float
    total: int


class EvalDiff(BaseModel):
    """Comparison of two evaluation reports for regression detection."""

    metric_diffs: dict[str, float] = Field(default_factory=dict)
    regressions: list[dict[str, Any]] = Field(default_factory=list)
    is_regression: bool = False
