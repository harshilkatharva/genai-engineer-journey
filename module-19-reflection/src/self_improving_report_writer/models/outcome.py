"""Domain models for acceptance gate outcomes, terminal report statuses, and human review workflows."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from self_improving_report_writer.models.critique import CritiqueIssue
from self_improving_report_writer.models.report import Report
from self_improving_report_writer.models.rubric import (
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DeterministicVerificationReport,
)


class ReportStatus(StrEnum):
    """Explicit terminal status of a research report deliverable."""

    ACCEPTED = "ACCEPTED"
    UNRESOLVED = "UNRESOLVED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    HUMAN_REJECTED = "HUMAN_REJECTED"


class ReportCandidate(BaseModel):
    """Immutable snapshot of a report candidate and all associated evaluations for a round."""

    round_number: int = Field(
        ge=0, description="Reflection round index (0 for baseline, 1+ for revisions)"
    )
    report: Report = Field(description="The generated or revised research report")
    deterministic_report: DeterministicVerificationReport = Field(
        description="Gate 1 deterministic verification results"
    )
    grounding_report: SemanticGroundingReport = Field(
        description="Gate 2 semantic grounding verification results"
    )
    judge_verdict: ReportJudgeVerdict = Field(
        description="Rubric judge evaluation and composite score"
    )
    tokens_used: int = Field(
        default=0, ge=0, description="Tokens consumed during generation/revision"
    )
    cost_usd: float = Field(default=0.0, ge=0.0, description="Financial cost incurred in USD")

    @property
    def total_critical_count(self) -> int:
        """Combined critical defects across Gate 1 and Gate 2."""
        return self.deterministic_report.critical_count + self.grounding_report.critical_count

    @property
    def total_major_count(self) -> int:
        """Combined major defects across Gate 1 and Gate 2."""
        return self.deterministic_report.major_count + self.grounding_report.major_count

    @property
    def grounding_subscore(self) -> float:
        """Sum of factual_accuracy and evidence_grounding dimension scores."""
        scores = self.judge_verdict.dimension_scores
        fa = scores.get("factual_accuracy")
        eg = scores.get("evidence_grounding")
        fa_val = fa.score if fa else 0.0
        eg_val = eg.score if eg else 0.0
        return round(fa_val + eg_val, 4)


class HumanReviewPacket(BaseModel):
    """Immutable packet containing all necessary context for human operator review (Section 6.2)."""

    run_id: str = Field(description="Unique identifier for the report generation run")
    topic: str = Field(description="Original research topic")
    best_candidate: Report = Field(description="The highest-ranking report candidate produced")
    best_candidate_version: int = Field(ge=1, description="Version index of best candidate")
    total_rounds_executed: int = Field(ge=0, description="Total reflection rounds executed")
    terminal_reason: str = Field(description="Reason the automated reflection loop stopped")
    deterministic_passed: bool = Field(
        description="Whether Gate 1 deterministic verification passed"
    )
    critical_defect_count: int = Field(ge=0, description="Remaining unresolved critical defects")
    major_defect_count: int = Field(ge=0, description="Remaining unresolved major defects")
    rubric_scores: dict[str, float] = Field(
        default_factory=dict, description="Final rubric dimension scores"
    )
    unresolved_critiques: list[CritiqueIssue] = Field(
        default_factory=list, description="Prioritized list of unresolved critique defects"
    )
    requires_human_action: bool = Field(
        default=True, description="Whether operator action is required before delivery"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HumanReviewDecision(BaseModel):
    """Structured decision submitted by an operator reviewing an unresolved report."""

    run_id: str = Field(description="Matching run identifier of the review packet")
    action: Literal["APPROVE_OVERRIDE", "REQUEST_REVISION", "REJECT"] = Field(
        description="Operator action: approve despite defects, request human-guided revision, or reject"
    )
    reviewer_notes: str = Field(description="Reviewer rationale, corrective guidance, or feedback")
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AcceptanceResult(BaseModel):
    """Outcome of the automated AcceptanceGate evaluation."""

    status: ReportStatus = Field(description="Terminal status: ACCEPTED or UNRESOLVED")
    is_accepted: bool = Field(description="Whether report meets all machine acceptance criteria")
    best_candidate: ReportCandidate = Field(description="The evaluated candidate")
    rationale: str = Field(
        description="Summary explanation of the acceptance or unresolved verdict"
    )
    unresolved_reasons: list[str] = Field(
        default_factory=list, description="Detailed list of failed acceptance conditions"
    )
    review_packet: HumanReviewPacket | None = Field(
        default=None, description="Human review packet if status is UNRESOLVED"
    )


class HumanReviewOutcome(BaseModel):
    """Result of processing an operator decision via HumanReviewGateway."""

    run_id: str = Field(description="Run identifier")
    status: ReportStatus = Field(description="Post-review terminal status")
    final_report: Report = Field(description="Final report deliverable")
    decision: HumanReviewDecision = Field(description="The operator decision applied")
    revised_candidate: ReportCandidate | None = Field(
        default=None, description="New candidate produced if revision was requested"
    )
    summary: str = Field(description="Human-readable outcome summary")
