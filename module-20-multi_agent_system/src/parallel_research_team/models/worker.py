"""Domain models for worker execution lifecycle, status, and deliverables."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class WorkerStatus(StrEnum):
    """Execution status of an individual research worker."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class WorkerResearchOutput(BaseModel):
    """Structured deliverable produced by a research worker upon brief execution."""

    brief_id: str = Field(description="Identifier of the executed brief")
    findings_summary: str = Field(
        description="Consolidated synthesis of evidence gathered for this brief"
    )
    claims: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted factual claims linked to evidence"
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved raw source chunks with source_id and chunk_id"
    )
    search_queries_used: list[str] = Field(
        default_factory=list, description="Search queries executed during tool calls"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Gaps, unreachable sources, or uncertainties encountered"
    )

    @field_validator("brief_id", "findings_summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or pure whitespace")
        return trimmed


class WorkerExecutionResult(BaseModel):
    """Complete execution record for an individual worker attempt."""

    brief_id: str = Field(description="Identifier of the executed brief")
    status: WorkerStatus = Field(description="Final execution lifecycle status of the worker")
    output: WorkerResearchOutput | None = Field(
        default=None, description="Deliverable produced if execution reached completion or partial"
    )
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed by this worker")
    cost_usd: float = Field(default=0.0, ge=0.0, description="USD cost incurred by this worker")
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Execution duration in milliseconds"
    )
    error_message: str | None = Field(
        default=None, description="Detailed error description if worker failed or timed out"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Execution start timestamp"
    )
    completed_at: datetime | None = Field(
        default=None, description="Execution completion timestamp"
    )
