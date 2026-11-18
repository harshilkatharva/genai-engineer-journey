"""Domain models for execution state checkpointing and recovery."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .brief import EffortLevel
from .decomposition import ResearchDecomposition
from .worker import WorkerExecutionResult


class RunStatus(StrEnum):
    """End-to-end execution lifecycle status for the Parallel Research Team."""

    INITIALIZING = "initializing"
    DECOMPOSING = "decomposing"
    EXECUTING = "executing"
    VALIDATING = "validating"
    SYNTHESIZING = "synthesizing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CheckpointState(BaseModel):
    """Serializable snapshot of the research team run for durable resume."""

    run_id: str = Field(description="Unique run execution identifier")
    user_id: str = Field(default="default_user", description="Owning user identifier")
    topic: str = Field(description="Original research question or topic")
    status: RunStatus = Field(description="Current or terminal execution status")
    effort_level: EffortLevel = Field(
        default=EffortLevel.PARALLEL_DECOMPOSED,
        description="Scaling classification: direct or parallel",
    )
    decomposition: ResearchDecomposition | None = Field(
        default=None, description="Decomposition plan and generated worker briefs"
    )
    worker_results: dict[str, WorkerExecutionResult] = Field(
        default_factory=dict,
        description="Execution results keyed by brief_id",
    )
    authoritative_sources: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Pooled authoritative sources keyed by chunk_id or composite key",
    )
    used_tokens: int = Field(default=0, ge=0, description="Cumulative tokens consumed so far")
    used_cost_usd: float = Field(default=0.0, ge=0.0, description="Cumulative USD expenditure")
    elapsed_seconds: float = Field(
        default=0.0, ge=0.0, description="Wall-clock elapsed execution seconds"
    )
    final_report: dict[str, Any] | None = Field(
        default=None, description="Serialized final Report model if synthesized"
    )
    error_message: str | None = Field(
        default=None, description="Detailed error description if run failed"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
