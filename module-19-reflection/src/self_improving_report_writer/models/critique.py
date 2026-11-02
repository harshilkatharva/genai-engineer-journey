"""Domain models for structured critiques, issues, severities, and categories."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class IssueSeverity(StrEnum):
    """Classification of issue severity in critiques."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class IssueCategory(StrEnum):
    """Taxonomy of critique issue categories."""

    SCHEMA = "schema"
    CITATION_INTEGRITY = "citation_integrity"
    FACTUAL_GROUNDING = "factual_grounding"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CLARITY_ORGANIZATION = "clarity_organization"
    LOGICAL_CONSISTENCY = "logical_consistency"


class CritiqueIssue(BaseModel):
    """An individual actionable defect identified during verification or rubric judging."""

    issue_id: str = Field(description="Unique issue identifier, e.g. 'ISSUE-001'")
    category: IssueCategory = Field(description="Taxonomy category of the deficiency")
    severity: IssueSeverity = Field(description="Severity classification")
    affected_section_id: str | None = Field(
        default=None, description="Identifier of the affected section"
    )
    affected_claim_id: str | None = Field(
        default=None, description="Identifier of the affected claim"
    )
    explanation: str = Field(description="Detailed explanation of the deficiency")
    supporting_source_ids: list[str] = Field(
        default_factory=list, description="IDs of sources containing corrective evidence"
    )
    suggested_correction: str = Field(description="Concrete, actionable revision recommendation")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the validity of critique"
    )
    is_resolved: bool = Field(
        default=False, description="Whether issue was successfully addressed in revision"
    )

    @field_validator("issue_id", "explanation", "suggested_correction")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or pure whitespace")
        return trimmed


class Critique(BaseModel):
    """Structured critique collection emitted during a reflection round."""

    round_number: int = Field(
        ge=0, description="Current reflection round number (0-indexed or 1-indexed)"
    )
    issues: list[CritiqueIssue] = Field(
        default_factory=list, description="Actionable critique issues prioritized by severity"
    )
    summary: str = Field(
        default="", description="Executive summary of defects and required revision goals"
    )
    should_revise: bool = Field(
        default=True, description="Whether another revision round is justified based on defects"
    )

    @property
    def critical_count(self) -> int:
        """Count of critical severity issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def major_count(self) -> int:
        """Count of major severity issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.MAJOR)

    @property
    def minor_count(self) -> int:
        """Count of minor severity issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.MINOR)
