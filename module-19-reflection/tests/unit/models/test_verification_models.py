"""Unit tests for verification domain models (DeterministicVerificationReport, VerificationIssue, PassageMatchResult)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_improving_report_writer.models.verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    PassageMatchResult,
    VerificationIssue,
)


def test_verification_enums() -> None:
    """Test DefectSeverity and DefectCategory enum string values."""
    assert DefectSeverity.CRITICAL == "critical"
    assert DefectSeverity.MAJOR == "major"
    assert DefectSeverity.MINOR == "minor"

    assert DefectCategory.SCHEMA == "schema"
    assert DefectCategory.CITATION_INTEGRITY == "citation_integrity"
    assert DefectCategory.FACTUAL_GROUNDING == "factual_grounding"
    assert DefectCategory.COMPLETENESS == "completeness"
    assert DefectCategory.ORGANIZATION_CLARITY == "organization_clarity"


def test_verification_issue_valid() -> None:
    """Test valid VerificationIssue attributes and defaults."""
    issue = VerificationIssue(
        issue_id="DEFECT-001",
        category=DefectCategory.CITATION_INTEGRITY,
        severity=DefectSeverity.CRITICAL,
        message="Quoted passage not found in referenced source chunk.",
        section_id="sec_1",
        claim_id="claim_1",
        citation_id="cite_1",
        details={"chunk_id": "chunk_abc"},
    )
    assert issue.issue_id == "DEFECT-001"
    assert issue.category == DefectCategory.CITATION_INTEGRITY
    assert issue.severity == DefectSeverity.CRITICAL
    assert issue.section_id == "sec_1"
    assert issue.details["chunk_id"] == "chunk_abc"


def test_passage_match_result_valid() -> None:
    """Test valid PassageMatchResult with match_type exact, normalized, missing."""
    pm_exact = PassageMatchResult(
        citation_id="cite_1",
        source_id="src_1",
        chunk_id="chk_1",
        is_match=True,
        match_type="exact",
        quoted_passage="quantum advantage demonstrated",
        normalized_passage="quantum advantage demonstrated",
        normalized_source_chunk="...quantum advantage demonstrated in laboratory...",
    )
    assert pm_exact.is_match is True
    assert pm_exact.match_type == "exact"

    pm_norm = PassageMatchResult(
        citation_id="cite_2",
        source_id="src_1",
        chunk_id="chk_1",
        is_match=True,
        match_type="normalized",
        quoted_passage="Quantum  Advantage  Demonstrated",
        normalized_passage="quantum advantage demonstrated",
        normalized_source_chunk="...quantum advantage demonstrated...",
    )
    assert pm_norm.match_type == "normalized"

    pm_missing = PassageMatchResult(
        citation_id="cite_3",
        source_id="src_1",
        chunk_id="chk_1",
        is_match=False,
        match_type="missing",
        quoted_passage="not present anywhere",
        normalized_passage="not present anywhere",
        normalized_source_chunk="...quantum advantage demonstrated...",
    )
    assert pm_missing.is_match is False
    assert pm_missing.match_type == "missing"

    with pytest.raises(ValidationError):
        PassageMatchResult(
            citation_id="cite_4",
            source_id="src_1",
            chunk_id="chk_1",
            is_match=False,
            match_type="fuzzy",  # type: ignore[arg-type] # invalid literal
            quoted_passage="text",
            normalized_passage="text",
            normalized_source_chunk="text",
        )


def test_deterministic_verification_report_valid() -> None:
    """Test DeterministicVerificationReport properties and defect counts."""
    report = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
        total_claims=5,
        total_citations=3,
        passage_matches=[],
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
    )
    assert report.passed is True
    assert report.total_claims == 5
    assert report.total_citations == 3
    assert report.critical_count == 0


def test_deterministic_verification_report_counts_non_negative() -> None:
    """Test that count fields reject negative numbers."""
    with pytest.raises(ValidationError):
        DeterministicVerificationReport(
            passed=False,
            schema_valid=True,
            citations_valid=True,
            passages_matched=False,
            total_claims=-1,
        )

    with pytest.raises(ValidationError):
        DeterministicVerificationReport(
            passed=False,
            schema_valid=True,
            citations_valid=True,
            passages_matched=False,
            critical_count=-1,
        )
