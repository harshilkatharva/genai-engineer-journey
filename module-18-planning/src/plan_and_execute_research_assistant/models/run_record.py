"""Run record and execution telemetry models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .enums import RunStatus, StepStatus
from .plan import Plan
from .verification import StepVerificationResult


class StepExecutionRecord(BaseModel):
    """Detailed telemetry record for an individual step attempt or execution."""

    step_id: str = Field(description="Step identifier")
    plan_version: int = Field(ge=1, description="Plan version in which this step executed")
    status: StepStatus = Field(description="Final status of this step attempt")
    attempts: int = Field(default=1, ge=1, description="Attempt number for this step execution")
    depends_on: list[str] = Field(
        default_factory=list, description="Dependencies declared for this step"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Execution start timestamp"
    )
    completed_at: datetime | None = Field(
        default=None, description="Execution completion timestamp"
    )
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed during this attempt")
    cost_usd: float = Field(
        default=0.0, ge=0.0, description="USD cost incurred during this attempt"
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Duration in milliseconds for this step execution"
    )
    result_summary: str | None = Field(
        default=None, description="Outcome summary produced by executor"
    )
    verification: StepVerificationResult | None = Field(
        default=None, description="Verification outcome for this step attempt"
    )
    error_message: str | None = Field(
        default=None, description="Error message if execution or verification failed"
    )


class RunRecord(BaseModel):
    """Complete, serializable run history capturing all plan versions, step executions, and telemetry."""

    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique UUID identifier for this run"
    )
    goal: str = Field(description="User goal submitted for execution")
    user_id: str = Field(description="User identifier owning this run")
    status: RunStatus = Field(
        default=RunStatus.INITIALIZING, description="Final or current status of the run"
    )
    final_deliverable: str | None = Field(
        default=None,
        description="Consolidated final answer or deliverable synthesized for the user",
    )
    plan_versions: list[Plan] = Field(
        default_factory=list, description="Append-only history of all plan versions generated"
    )
    step_executions: dict[str, list[StepExecutionRecord]] = Field(
        default_factory=dict, description="Step executions keyed by step_id"
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Aggregate tokens consumed across all phases"
    )
    total_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Aggregate USD cost incurred across all phases"
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Total wall-clock duration of the run in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Run initialization timestamp"
    )
    completed_at: datetime | None = Field(
        default=None, description="Run completion or termination timestamp"
    )
    error_message: str | None = Field(
        default=None, description="Error message if run terminated in failure"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata or telemetry tags"
    )
