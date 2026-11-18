"""Consolidated run outcome and metrics models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .checkpoint import CheckpointState, RunStatus
from .decomposition import ResearchDecomposition
from .report import Report


class ExecutionMetrics(BaseModel):
    """Resource consumption and performance metrics for a research run."""

    total_tokens: int = Field(default=0, ge=0, description="Cumulative tokens consumed")
    total_cost_usd: float = Field(default=0.0, ge=0.0, description="Total USD expenditure")
    duration_ms: float = Field(default=0.0, ge=0.0, description="Total duration in milliseconds")
    worker_breakdown: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Resource and latency breakdown per worker brief"
    )
    total_workers: int = Field(default=0, ge=0)
    completed_workers: int = Field(default=0, ge=0)
    partial_workers: int = Field(default=0, ge=0)
    failed_workers: int = Field(default=0, ge=0)
    timed_out_workers: int = Field(default=0, ge=0)


class ResearchTeamRunResult(BaseModel):
    """Final output returned by the ParallelResearchTeam orchestrator."""

    run_id: str = Field(description="Unique run execution identifier")
    status: RunStatus = Field(description="Final execution lifecycle status")
    report: Report | None = Field(
        default=None, description="Final synthesized and verified research report"
    )
    decomposition: ResearchDecomposition | None = Field(
        default=None, description="Decomposition plan and assigned worker briefs"
    )
    metrics: ExecutionMetrics = Field(
        default_factory=ExecutionMetrics, description="Detailed run telemetry and cost metrics"
    )
    checkpoint_state: CheckpointState | None = Field(
        default=None, description="Final execution checkpoint state"
    )
    error_message: str | None = Field(default=None, description="Error details if execution failed")
