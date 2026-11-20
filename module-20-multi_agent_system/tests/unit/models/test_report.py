"""Unit tests for Module 20 Report models and Module 19 verification compatibility."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from parallel_research_team.models.report import (
    BriefCoverage,
    Citation,
    CitationType,
    Claim,
    Report,
    ReportSection,
    SourceReference,
)


def test_source_reference() -> None:
    src = SourceReference(
        source_id="src_01",
        chunk_id="chk_01",
        title="Title",
        content="Content text",
    )
    assert src.source_id == "src_01"
    assert src.score == 1.0

    with pytest.raises(ValidationError):
        SourceReference(source_id="", chunk_id="chk", title="T", content="C")


def test_citation_quotation_validation() -> None:
    # Valid quotation citation
    cite = Citation(
        citation_id="cite_01",
        source_id="src_01",
        chunk_id="chk_01",
        citation_type=CitationType.QUOTATION,
        quoted_passage="Exact passage",
    )
    assert cite.quoted_passage == "Exact passage"

    # Quotation without quoted_passage fails
    with pytest.raises(ValidationError, match="must include non-empty quoted_passage"):
        Citation(
            citation_id="cite_02",
            source_id="src_01",
            chunk_id="chk_01",
            citation_type=CitationType.QUOTATION,
            quoted_passage=None,
        )


def test_report_structural_integrity() -> None:
    cite = Citation(
        citation_id="cite_01",
        source_id="src_01",
        chunk_id="chk_01",
        citation_type=CitationType.QUOTATION,
        quoted_passage="HNSW delivers sub-5ms query latency",
    )
    claim = Claim(
        claim_id="claim_01",
        section_id="sec_exec",
        text="HNSW achieves rapid query performance.",
        citations=[cite],
    )
    sec_exec = ReportSection(
        section_id="sec_exec",
        title="Executive Summary",
        content="HNSW delivers sub-5ms query latency in benchmarks.",
        claims=[claim],
    )
    sec_body = ReportSection(
        section_id="sec_body",
        title="Comparative Findings",
        content="Detailed analysis...",
        claims=[],
    )
    sec_conclusion = ReportSection(
        section_id="sec_conclusion",
        title="Conclusion",
        content="Summary conclusion...",
        claims=[],
    )

    report = Report(
        topic="Vector Indexing",
        title="Comparative Vector Indexing Study",
        executive_summary=sec_exec,
        sections=[sec_body],
        conclusion=sec_conclusion,
        limitations=["Small memory test set"],
        brief_coverages=[
            BriefCoverage(brief_id="brief_01", coverage_summary="Covered in sec_body")
        ],
    )
    assert report.title == "Comparative Vector Indexing Study"
    assert report.research_team_version == "1.0.0"
    assert len(report.sections) == 1


def test_report_version_reads_from_config() -> None:
    from parallel_research_team.core.config import get_research_team_config

    cfg = get_research_team_config()
    orig_ver = cfg.system_version
    try:
        cfg.system_version = "2.1.0-beta"
        sec_exec = ReportSection(
            section_id="sec_exec",
            title="Overview",
            content="Content",
            claims=[],
        )
        sec_body = ReportSection(
            section_id="sec_body",
            title="Findings",
            content="Content",
            claims=[],
        )
        sec_concl = ReportSection(
            section_id="sec_concl",
            title="Conclusion",
            content="Content",
            claims=[],
        )
        rep = Report(
            topic="Test",
            title="Title",
            executive_summary=sec_exec,
            sections=[sec_body],
            conclusion=sec_concl,
        )
        assert rep.research_team_version == "2.1.0-beta"
    finally:
        cfg.system_version = orig_ver


def test_report_duplicate_section_id_fails() -> None:
    sec_exec = ReportSection(
        section_id="sec_dup",
        title="Executive Summary",
        content="Content",
        claims=[],
    )
    sec_body = ReportSection(
        section_id="sec_dup",  # Duplicate section ID
        title="Body",
        content="Content",
        claims=[],
    )
    sec_conclusion = ReportSection(
        section_id="sec_concl",
        title="Conclusion",
        content="Content",
        claims=[],
    )

    with pytest.raises(ValidationError, match="Duplicate section_id detected"):
        Report(
            topic="Topic",
            title="Title",
            executive_summary=sec_exec,
            sections=[sec_body],
            conclusion=sec_conclusion,
        )


def test_report_claim_section_mismatch_fails() -> None:
    claim = Claim(
        claim_id="c1",
        section_id="sec_other",  # Mismatch with sec_exec
        text="Factual claim",
    )
    sec_exec = ReportSection(
        section_id="sec_exec",
        title="Executive Summary",
        content="Content",
        claims=[claim],
    )
    sec_conclusion = ReportSection(
        section_id="sec_concl",
        title="Conclusion",
        content="Content",
        claims=[],
    )

    with pytest.raises(ValidationError, match="does not match parent section"):
        Report(
            topic="Topic",
            title="Title",
            executive_summary=sec_exec,
            sections=[],
            conclusion=sec_conclusion,
        )
