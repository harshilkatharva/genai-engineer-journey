"""Domain models for research question decomposition and planning."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .brief import EffortLevel, WorkerBrief


class ResearchDecomposition(BaseModel):
    """Decomposition plan produced by the Lead Agent."""

    topic: str = Field(description="The original user research question or prompt")
    effort_level: EffortLevel = Field(
        description="Scaling classification: DIRECT (1 worker) vs PARALLEL_DECOMPOSED (2-5 workers)"
    )
    rationale: str = Field(
        description="Reasoning behind the decomposition and effort scaling decision"
    )
    worker_count: int = Field(ge=1, le=10, description="Number of worker briefs generated (1 to 5)")
    worker_briefs: list[WorkerBrief] = Field(
        min_length=1, description="List of typed worker briefs assigned to workers"
    )
    input_tokens: int = Field(
        default=0, ge=0, description="Input tokens consumed during decomposition"
    )
    output_tokens: int = Field(
        default=0, ge=0, description="Output tokens consumed during decomposition"
    )

    @field_validator("topic", "rationale")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or pure whitespace")
        return trimmed

    @field_validator("worker_briefs")
    @classmethod
    def validate_brief_counts(cls, v: list[WorkerBrief], info: Any = None) -> list[WorkerBrief]:
        if not v:
            raise ValueError("ResearchDecomposition must have at least one WorkerBrief")
        ids = [b.brief_id for b in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate brief_id detected in worker_briefs")
        return v
