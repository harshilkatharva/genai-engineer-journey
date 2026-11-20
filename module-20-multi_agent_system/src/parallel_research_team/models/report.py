"""Module 20 native research report domain models.

Maintains structural compatibility with Module 19 verification gates (DeterministicVerifier
and SemanticGroundingVerifier) while adding Module 20 specific research metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CitationType(StrEnum):
    """Classification of citation evidence representation."""

    QUOTATION = "quotation"
    PARAPHRASE = "paraphrase"


class SourceReference(BaseModel):
    """Metadata describing an immutable evidence chunk in the authoritative pool."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Unique identifier of the source document or URL hash")
    chunk_id: str = Field(description="Unique identifier of the specific retrieved chunk")
    title: str = Field(description="Title of the source publication or web document")
    content: str = Field(description="Raw text content of the retrieved chunk (immutable)")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Retrieval similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Preserved source metadata")

    @field_validator("source_id", "chunk_id", "title", "content")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or pure whitespace")
        return trimmed


class Citation(BaseModel):
    """Explicit citation linking a factual claim to an authoritative source chunk."""

    citation_id: str = Field(
        description="Unique citation identifier within the report (e.g. 'cite_1')"
    )
    source_id: str = Field(description="Referenced source document identifier")
    chunk_id: str = Field(description="Referenced source chunk identifier")
    citation_type: CitationType = Field(
        default=CitationType.QUOTATION,
        description="Whether evidence is an exact quotation or paraphrased statement",
    )
    quoted_passage: str | None = Field(
        default=None,
        description="Verbatim excerpt from the source chunk. Mandatory if citation_type is QUOTATION.",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in citation relevance"
    )

    @field_validator("citation_id", "source_id", "chunk_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Identifier fields cannot be empty or pure whitespace")
        return trimmed

    @model_validator(mode="after")
    def validate_quotation_passage(self) -> Citation:
        if self.citation_type == CitationType.QUOTATION and (
            not self.quoted_passage or not self.quoted_passage.strip()
        ):
            raise ValueError("Citations of type QUOTATION must include non-empty quoted_passage")
        return self


class Claim(BaseModel):
    """An individual factual claim requiring grounding and citation."""

    claim_id: str = Field(description="Unique claim identifier, e.g. 'claim_1'")
    section_id: str = Field(description="Identifier of the section containing this claim")
    text: str = Field(description="Factual proposition asserted in the report")
    citations: list[Citation] = Field(
        default_factory=list, description="Citations supporting this factual assertion"
    )
    is_factual: bool = Field(
        default=True, description="Whether statement asserts a verifiable factual claim"
    )

    @field_validator("claim_id", "section_id", "text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Claim fields cannot be empty or pure whitespace")
        return trimmed


class ReportSection(BaseModel):
    """A distinct structural section of the research report with linked claims."""

    section_id: str = Field(description="Unique section identifier, e.g. 'sec_findings'")
    title: str = Field(description="Section heading, e.g. 'Key Findings'")
    content: str = Field(
        description="Full textual prose of the section with inline citation references"
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="Structured claims extracted from or substantiating the prose",
    )

    @field_validator("section_id", "title", "content")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Section fields cannot be empty or pure whitespace")
        return trimmed


class BriefCoverage(BaseModel):
    """Mapping of worker brief coverage integrated into the synthesized report."""

    brief_id: str = Field(description="Unique brief identifier")
    coverage_summary: str = Field(
        description="Summary of brief findings integrated into the report"
    )

    @field_validator("brief_id", "coverage_summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or pure whitespace")
        return trimmed


def _default_research_team_version() -> str:
    """Dynamically retrieve the research team version from active configuration."""
    from ..core.config import get_research_team_config

    return get_research_team_config().system_version


class Report(BaseModel):
    """Complete structured research report synthesized from parallel worker findings.

    Fully compatible with Module 19's DeterministicVerifier and SemanticGroundingVerifier,
    while adding Module 20 specific coverage and limitation tracking fields.
    """

    topic: str = Field(description="The user's original research topic or prompt")
    title: str = Field(description="Descriptive report title")
    executive_summary: ReportSection = Field(
        description="Executive summary section with prose content and structured claims"
    )
    sections: list[ReportSection] = Field(
        description="Core body sections detailing findings, evidence, and analysis"
    )
    conclusion: ReportSection = Field(
        description="Final synthesis section with prose content and structured claims"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Unresolved questions, data gaps, and documented worker failures",
    )
    version: int = Field(default=1, ge=1, description="Monotonically increasing report version")
    research_team_version: str = Field(
        default_factory=_default_research_team_version,
        description="Module 20 research team generator version",
    )
    brief_coverages: list[BriefCoverage] = Field(
        default_factory=list,
        description="List of brief coverage summaries in the synthesized report",
    )
    unresolved_limitations: list[str] = Field(
        default_factory=list,
        description="Explicit gaps from failed or timed-out worker briefs",
    )
    created_by: str = Field(
        default="parallel_research_team_lead",
        description="Author entity of this report",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def brief_coverages_dict(self) -> dict[str, str]:
        """Convenience dictionary representation of brief coverage mappings."""
        return {bc.brief_id: bc.coverage_summary for bc in self.brief_coverages}

    @field_validator("brief_coverages", mode="before")
    @classmethod
    def coerce_brief_coverages(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return [
                BriefCoverage(brief_id=str(k), coverage_summary=str(val)) for k, val in v.items()
            ]
        return v

    @field_validator("topic", "title")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Topic and title cannot be empty or pure whitespace")
        return trimmed

    @model_validator(mode="after")
    def validate_structural_integrity(self) -> Report:
        """Validate uniqueness of section IDs, claim IDs, citation IDs, and claim-section bindings."""
        all_sections = [self.executive_summary, *self.sections, self.conclusion]
        section_ids = [s.section_id for s in all_sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Duplicate section_id detected across report sections")

        seen_claims: set[str] = set()
        seen_citations: set[str] = set()

        for sec in all_sections:
            for claim in sec.claims:
                if claim.claim_id in seen_claims:
                    raise ValueError(f"Duplicate claim_id detected: '{claim.claim_id}'")
                seen_claims.add(claim.claim_id)

                if claim.section_id != sec.section_id:
                    raise ValueError(
                        f"Claim '{claim.claim_id}' section_id '{claim.section_id}' does not match "
                        f"parent section '{sec.section_id}'"
                    )

                for citation in claim.citations:
                    if citation.citation_id in seen_citations:
                        raise ValueError(
                            f"Duplicate citation_id detected: '{citation.citation_id}'"
                        )
                    seen_citations.add(citation.citation_id)

        return self
