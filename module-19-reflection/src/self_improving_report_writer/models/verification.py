"""Domain models for deterministic and grounding verification reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DefectSeverity(StrEnum):
    """Classification of defect severity."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class DefectCategory(StrEnum):
    """Classification of defect category."""

    SCHEMA = "schema"
    CITATION_INTEGRITY = "citation_integrity"
    FACTUAL_GROUNDING = "factual_grounding"
    COMPLETENESS = "completeness"
    ORGANIZATION_CLARITY = "organization_clarity"


class VerificationIssue(BaseModel):
    """An individual defect identified during deterministic verification."""

    issue_id: str = Field(description="Unique defect identifier, e.g. 'DEFECT-001'")
    category: DefectCategory = Field(description="Category of the defect")
    severity: DefectSeverity = Field(description="Severity classification")
    message: str = Field(description="Descriptive explanation of the defect")
    section_id: str | None = Field(default=None, description="Affected section ID if applicable")
    claim_id: str | None = Field(default=None, description="Affected claim ID if applicable")
    citation_id: str | None = Field(default=None, description="Affected citation ID if applicable")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary diagnostic details"
    )


class PassageMatchResult(BaseModel):
    """Result of deterministic substring containment check for a quotation citation."""

    citation_id: str = Field(description="Evaluated citation ID")
    source_id: str = Field(description="Referenced source ID")
    chunk_id: str = Field(description="Referenced chunk ID")
    is_match: bool = Field(description="Whether the passage was found in the source chunk")
    match_type: Literal["exact", "normalized", "missing"] = Field(
        description="Type of match: exact string, normalized substring, or missing"
    )
    quoted_passage: str = Field(description="Original quoted passage from the citation")
    normalized_passage: str = Field(description="Normalized quoted passage")
    normalized_source_chunk: str = Field(description="Normalized source chunk text")


class DeterministicVerificationReport(BaseModel):
    """Comprehensive outcome of Gate 1 deterministic verification."""

    passed: bool = Field(description="Whether all deterministic verification checks passed")
    schema_valid: bool = Field(description="Whether report schema and structural checks passed")
    citations_valid: bool = Field(
        description="Whether all citation references point to valid source chunks"
    )
    passages_matched: bool = Field(
        description="Whether all quotation passages matched their source chunks"
    )
    total_claims: int = Field(default=0, ge=0, description="Total claims evaluated")
    total_citations: int = Field(default=0, ge=0, description="Total citations evaluated")
    passage_matches: list[PassageMatchResult] = Field(
        default_factory=list, description="Detailed passage matching results"
    )
    issues: list[VerificationIssue] = Field(
        default_factory=list, description="List of all detected issues"
    )
    critical_count: int = Field(default=0, ge=0, description="Count of critical severity defects")
    major_count: int = Field(default=0, ge=0, description="Count of major severity defects")
    minor_count: int = Field(default=0, ge=0, description="Count of minor severity defects")
