"""Pydantic response schemas for LLM structured output evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from self_improving_report_writer.models.rubric import DimensionScore


class LLMJudgeEvaluation(BaseModel):
    """Raw structured output expected from the LLM Judge before deterministic mathematical rollup."""

    dimension_scores: list[DimensionScore] = Field(
        description="Individual score and evaluation for each rubric dimension"
    )
    overall_feedback: str = Field(
        description="Holistic evaluation summarizing report quality, evidence strength, and key deficits"
    )


class ClaimEntailmentEvaluation(BaseModel):
    """Structured entailment verdict for a single claim evaluated against its cited source chunks."""

    entailment: Literal["ENTAILS", "NEUTRAL", "CONTRADICTS"] = Field(
        description="Entailment relationship: ENTAILS (supported), NEUTRAL (insufficient/unsupported), CONTRADICTS (refuted)"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Entailment score: 1.0 for ENTAILS, 0.5 for NEUTRAL, 0.0 for CONTRADICTS",
    )
    reasoning: str = Field(
        description="Clear, concise rationale detailing why the context entails, lacks support for, or contradicts the claim"
    )


class SingleClaimEntailmentVerdict(BaseModel):
    """Structured entailment verdict for a single claim evaluated within a batch."""

    claim_id: str = Field(description="Exact ID of the claim being evaluated")
    entailment: Literal["ENTAILS", "NEUTRAL", "CONTRADICTS"] = Field(
        description="Entailment relationship: ENTAILS (supported), NEUTRAL (insufficient/unsupported), CONTRADICTS (refuted)"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Entailment score: 1.0 for ENTAILS, 0.5 for NEUTRAL, 0.0 for CONTRADICTS",
    )
    reasoning: str = Field(
        description="Clear, concise rationale detailing why the context entails, lacks support for, or contradicts the claim"
    )


class SectionClaimsEntailmentEvaluation(BaseModel):
    """Structured batch evaluation response containing individual verdicts for multiple claims."""

    evaluations: list[SingleClaimEntailmentVerdict] = Field(
        default_factory=list,
        description="List of entailment verdicts for all evaluated claims in the section",
    )


class ProseConsistencyEvaluation(BaseModel):
    """Evaluation of whether section prose contains factual statements missing from structured claims."""

    unstructured_claims: list[str] = Field(
        default_factory=list,
        description="List of substantive factual statements asserted in prose that are not represented in the structured claims",
    )
    reasoning: str = Field(
        description="Explanation of whether the prose introduces ungrounded factual assertions"
    )
