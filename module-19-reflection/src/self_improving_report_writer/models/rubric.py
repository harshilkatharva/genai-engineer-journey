"""Domain models for rubric configuration, dimension scores, and judge verdicts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from self_improving_report_writer.models.verification import (
    VerificationIssue,
)


class RubricCriterion(BaseModel):
    """An individual evaluation dimension within the rubric."""

    name: str = Field(description="Name of the evaluation dimension")
    weight: float = Field(ge=0.0, le=1.0, description="Mathematical weight in composite score")
    description: str = Field(description="Rubric description and scoring instructions")
    min_pass_score: float = Field(
        default=0.80, ge=0.0, le=1.0, description="Minimum acceptable passing score"
    )

    @field_validator("name", "description")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty")
        return trimmed


class RubricConfig(BaseModel):
    """Validated rubric configuration with strict mathematical weight normalization."""

    criteria: list[RubricCriterion] = Field(
        description="List of rubric criteria whose weights must sum strictly to 1.0"
    )
    acceptance_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Minimum composite score for report acceptance"
    )

    @field_validator("criteria")
    @classmethod
    def validate_weights_sum_to_one(cls, v: list[RubricCriterion]) -> list[RubricCriterion]:
        if not v:
            raise ValueError("Rubric must contain at least one criterion")
        total_weight = sum(c.weight for c in v)
        if not (0.999 <= total_weight <= 1.001):
            raise ValueError(f"Rubric criteria weights must sum to 1.0 (got {total_weight:.4f})")
        # Check duplicate names
        names = [c.name.strip().lower() for c in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate criterion name detected in rubric configuration")
        return v


def get_default_rubric_config() -> RubricConfig:
    """Return the confirmed 7-dimension calibrated rubric configuration."""
    return RubricConfig(
        criteria=[
            RubricCriterion(
                name="factual_accuracy",
                weight=0.25,
                description="Absence of hallucinations, false claims, or fabricated statistics.",
                min_pass_score=0.85,
            ),
            RubricCriterion(
                name="evidence_grounding",
                weight=0.20,
                description="Degree to which all factual claims and prose statements are supported by sources.",
                min_pass_score=0.85,
            ),
            RubricCriterion(
                name="citation_correctness",
                weight=0.15,
                description="Precision of source attribution, valid chunk pointers, and accurate quotation excerpts.",
                min_pass_score=0.90,
            ),
            RubricCriterion(
                name="completeness",
                weight=0.15,
                description="Comprehensive topic coverage, balanced perspectives, and caveats/limitations.",
                min_pass_score=0.80,
            ),
            RubricCriterion(
                name="relevance",
                weight=0.10,
                description="Strict adherence to the research prompt without off-topic filler or padding.",
                min_pass_score=0.80,
            ),
            RubricCriterion(
                name="clarity_organization",
                weight=0.10,
                description="Logical section hierarchy, executive tone, synthesis, and readability.",
                min_pass_score=0.80,
            ),
            RubricCriterion(
                name="logical_consistency",
                weight=0.05,
                description="Internal coherence without self-contradictory assertions between sections.",
                min_pass_score=0.85,
            ),
        ],
        acceptance_threshold=0.85,
    )


class DimensionScore(BaseModel):
    """Detailed score and feedback for an individual rubric dimension."""

    dimension_name: str = Field(description="Name of the rubric dimension")
    score: float = Field(ge=0.0, le=1.0, description="Normalized score from 0.0 to 1.0")
    passed: bool = Field(description="Whether score meets dimension min_pass_score")
    reasoning: str = Field(description="Objective, evidence-based critique and rationale")


class ReportJudgeVerdict(BaseModel):
    """Normalized structured evaluation emitted by the RubricJudge."""

    composite_score: float = Field(
        ge=0.0, le=1.0, description="Weighted composite quality score across all dimensions"
    )
    passed: bool = Field(
        description="Whether composite score and critical dimensions meet acceptance thresholds"
    )
    dimension_scores: dict[str, DimensionScore] = Field(
        description="Map of dimension name to detailed dimension score"
    )
    overall_feedback: str = Field(
        description="Executive summary of strengths, defects, and improvement opportunities"
    )
    is_self_judged: bool = Field(
        default=False,
        description="Flag indicating whether evaluation was conducted by the generator model fallback",
    )
    judge_model: str = Field(
        default="gemini-2.5-pro", description="Model identifier used for evaluation"
    )
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class ClaimGroundingResult(BaseModel):
    """Grounding evaluation for an individual claim."""

    claim_id: str = Field(description="Evaluated claim ID")
    section_id: str = Field(description="Section ID of the claim")
    entailment: Literal["ENTAILS", "NEUTRAL", "CONTRADICTS"] = Field(
        description="Semantic entailment verdict"
    )
    score: float = Field(ge=0.0, le=1.0, description="Grounding score (1.0, 0.5, or 0.0)")
    reasoning: str = Field(description="Explanation of whether the source entails the claim")
    cited_chunk_ids: list[str] = Field(
        default_factory=list, description="Referenced chunk IDs evaluated"
    )


class SemanticGroundingReport(BaseModel):
    """Comprehensive Gate 2 semantic grounding evaluation report."""

    passed: bool = Field(description="Whether all factual claims are satisfactorily grounded")
    all_claims_grounded: bool = Field(
        description="Whether zero contradiction or insufficient grounding defects were detected"
    )
    grounding_score: float = Field(
        ge=0.0, le=1.0, description="Ratio of supported to total factual claims"
    )
    claim_evaluations: list[ClaimGroundingResult] = Field(
        default_factory=list, description="Per-claim entailment results"
    )
    issues: list[VerificationIssue] = Field(
        default_factory=list, description="List of detected semantic grounding issues"
    )
    critical_count: int = Field(default=0, ge=0, description="Count of critical severity defects")
    major_count: int = Field(default=0, ge=0, description="Count of major severity defects")
    minor_count: int = Field(default=0, ge=0, description="Count of minor severity defects")
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
