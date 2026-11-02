"""Core domain models for research reports, sections, claims, and citations."""

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
    """Metadata describing an immutable evidence chunk in the application-controlled pool."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="Unique identifier of the source document or URL")
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
    """Explicit citation linking a factual claim to a retrieved source chunk."""

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
        default=1.0, ge=0.0, le=1.0, description="Generator confidence in citation relevance"
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


class Report(BaseModel):
    """Complete structured research report covering all sections and claims."""

    topic: str = Field(description="The user's original research topic or prompt")
    title: str = Field(description="Compelling, descriptive report title")
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
        default_factory=list, description="Unresolved questions, data gaps, and study caveats"
    )
    version: int = Field(default=1, ge=1, description="Monotonically increasing revision version")
    created_by: str = Field(
        default="generator_llm",
        description="Origin: generator_llm, revision_llm, human_revision",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

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
            for c in sec.claims:
                if c.section_id != sec.section_id:
                    raise ValueError(
                        f"Claim '{c.claim_id}' has section_id='{c.section_id}' "
                        f"which does not match parent section_id='{sec.section_id}'"
                    )
                if c.claim_id in seen_claims:
                    raise ValueError(f"Duplicate claim_id detected: '{c.claim_id}'")
                seen_claims.add(c.claim_id)

                for cite in c.citations:
                    if cite.citation_id in seen_citations:
                        raise ValueError(f"Duplicate citation_id detected: '{cite.citation_id}'")
                    seen_citations.add(cite.citation_id)

        return self
