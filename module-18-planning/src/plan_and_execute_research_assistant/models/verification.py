"""Step verification models for programmatic and model-based validation."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class StepJudgeVerdict(BaseModel):
    """Structured verdict emitted by model-based evaluation judge."""

    is_satisfied: bool = Field(
        description="Whether the step output completely satisfies the explicit success criteria"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0 indicating degree of criteria satisfaction",
    )
    reasoning: str = Field(description="Objective justification explaining the verdict")
    missing_criteria: list[str] = Field(
        default_factory=list,
        description="Specific criteria points that were unfulfilled or missing from the output",
    )


class StepVerificationResult(BaseModel):
    """Consolidated outcome of both programmatic and model-based step verification."""

    step_id: str = Field(description="Identifier of the step verified")
    passed: bool = Field(
        description="Overall verification pass status (true only if both programmatic and model pass)"
    )
    programmatic_passed: bool = Field(
        description="Whether programmatic checks (status, non-empty text, format) succeeded"
    )
    model_passed: bool = Field(
        description="Whether model judge evaluated the deliverable as satisfying success criteria"
    )
    score: float = Field(
        ge=0.0, le=1.0, description="Verification confidence score (from model judge)"
    )
    reasoning: str = Field(description="Explanatory reasoning or failure diagnosis")
    missing_criteria: list[str] = Field(
        default_factory=list, description="Specific unfulfilled criteria points if failed"
    )
    tokens_used: int = Field(
        default=0, ge=0, description="Tokens consumed during model verification"
    )
    cost_usd: float = Field(
        default=0.0, ge=0.0, description="USD cost incurred during model verification"
    )
    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when verification completed",
    )
