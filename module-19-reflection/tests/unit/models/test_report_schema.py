"""Unit tests for Report domain models and schema validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_improving_report_writer.models.report import (
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)


def test_valid_report_creation(valid_report: Report) -> None:
    """A valid report should initialize with all fields properly populated."""
    assert valid_report.topic.startswith("Quantum Annealing")
    assert valid_report.version == 1
    assert valid_report.created_by == "generator_llm"
    assert len(valid_report.sections) == 1
    assert valid_report.executive_summary.section_id == "sec_exec"
    assert valid_report.conclusion.section_id == "sec_concl"


def test_source_reference_immutability(sample_sources: list[SourceReference]) -> None:
    """SourceReference should be frozen and prevent accidental mutation."""
    src = sample_sources[0]
    with pytest.raises(ValidationError):
        src.content = "mutated content"  # type: ignore[misc]


def test_source_reference_empty_fields_rejected() -> None:
    """SourceReference should reject empty string fields."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        SourceReference(source_id="   ", chunk_id="c1", title="Title", content="Content")


def test_citation_quotation_requires_quoted_passage() -> None:
    """Citations of type QUOTATION must provide a non-empty quoted_passage."""
    with pytest.raises(
        ValidationError, match="Citations of type QUOTATION must include non-empty quoted_passage"
    ):
        Citation(
            citation_id="c1",
            source_id="s1",
            chunk_id="ch1",
            citation_type=CitationType.QUOTATION,
            quoted_passage=None,
        )

    with pytest.raises(
        ValidationError, match="Citations of type QUOTATION must include non-empty quoted_passage"
    ):
        Citation(
            citation_id="c1",
            source_id="s1",
            chunk_id="ch1",
            citation_type=CitationType.QUOTATION,
            quoted_passage="   ",
        )


def test_citation_paraphrase_without_passage_allowed() -> None:
    """Citations of type PARAPHRASE do not require quoted_passage."""
    cite = Citation(
        citation_id="c1",
        source_id="s1",
        chunk_id="ch1",
        citation_type=CitationType.PARAPHRASE,
        quoted_passage=None,
    )
    assert cite.quoted_passage is None
    assert cite.citation_type == CitationType.PARAPHRASE


def test_duplicate_section_id_rejected(valid_report: Report) -> None:
    """Reports with duplicate section_ids must be rejected."""
    duplicate_sec = ReportSection(
        section_id="sec_findings",  # Duplicate with valid_report.sections[0]
        title="Duplicate Findings",
        content="Some content",
        claims=[],
    )
    with pytest.raises(ValidationError, match="Duplicate section_id detected"):
        Report(
            topic="Test Topic",
            title="Test Title",
            executive_summary=valid_report.executive_summary,
            sections=[valid_report.sections[0], duplicate_sec],
            conclusion=valid_report.conclusion,
        )


def test_duplicate_claim_id_rejected(valid_report: Report) -> None:
    """Reports with duplicate claim_ids across sections must be rejected."""
    dup_claim = Claim(
        claim_id="claim_find_1",  # Duplicate of claim in findings section
        section_id="sec_exec",
        text="A duplicate claim statement.",
        citations=[],
    )
    bad_exec = ReportSection(
        section_id="sec_exec",
        title="Exec",
        content="Exec content",
        claims=[valid_report.executive_summary.claims[0], dup_claim],
    )
    with pytest.raises(ValidationError, match="Duplicate claim_id detected: 'claim_find_1'"):
        Report(
            topic="Test Topic",
            title="Test Title",
            executive_summary=bad_exec,
            sections=valid_report.sections,
            conclusion=valid_report.conclusion,
        )


def test_duplicate_citation_id_rejected(valid_report: Report) -> None:
    """Reports with duplicate citation_ids must be rejected."""
    dup_cite = Citation(
        citation_id="cite_find_1",  # Duplicate of citation in findings section
        source_id="src_quantum_01",
        chunk_id="chunk_q01",
        citation_type=CitationType.PARAPHRASE,
    )
    bad_claim = Claim(
        claim_id="claim_exec_new",
        section_id="sec_exec",
        text="New statement",
        citations=[dup_cite],
    )
    bad_exec = ReportSection(
        section_id="sec_exec",
        title="Exec",
        content="Exec content",
        claims=[bad_claim],
    )
    with pytest.raises(ValidationError, match="Duplicate citation_id detected: 'cite_find_1'"):
        Report(
            topic="Test Topic",
            title="Test Title",
            executive_summary=bad_exec,
            sections=valid_report.sections,
            conclusion=valid_report.conclusion,
        )


def test_claim_section_id_mismatch_rejected(valid_report: Report) -> None:
    """A claim whose section_id does not match its containing section must be rejected."""
    mismatched_claim = Claim(
        claim_id="claim_mismatch",
        section_id="sec_findings",  # Mismatched with sec_exec
        text="Statement in exec",
        citations=[],
    )
    bad_exec = ReportSection(
        section_id="sec_exec",
        title="Exec",
        content="Exec content",
        claims=[mismatched_claim],
    )
    with pytest.raises(ValidationError, match="does not match parent section_id"):
        Report(
            topic="Test Topic",
            title="Test Title",
            executive_summary=bad_exec,
            sections=valid_report.sections,
            conclusion=valid_report.conclusion,
        )


def test_empty_string_validation_errors() -> None:
    """Empty strings should raise ValidationError on all domain models."""
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", section_id="s1", text="   ")

    with pytest.raises(ValidationError):
        ReportSection(section_id="s1", title="   ", content="content")

    with pytest.raises(ValidationError):
        ReportSection(section_id="s1", title="Title", content="   ")
