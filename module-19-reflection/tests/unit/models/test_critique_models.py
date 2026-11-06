"""Unit tests for critique domain models (CritiqueIssue, Critique, IssueSeverity, IssueCategory)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)


def test_issue_enums() -> None:
    """Test IssueSeverity and IssueCategory enum string values."""
    assert IssueSeverity.CRITICAL == "critical"
    assert IssueSeverity.MAJOR == "major"
    assert IssueSeverity.MINOR == "minor"

    assert IssueCategory.SCHEMA == "schema"
    assert IssueCategory.CITATION_INTEGRITY == "citation_integrity"
    assert IssueCategory.FACTUAL_GROUNDING == "factual_grounding"
    assert IssueCategory.COMPLETENESS == "completeness"
    assert IssueCategory.RELEVANCE == "relevance"
    assert IssueCategory.CLARITY_ORGANIZATION == "clarity_organization"
    assert IssueCategory.LOGICAL_CONSISTENCY == "logical_consistency"


def test_critique_issue_valid() -> None:
    """Test valid CritiqueIssue creation and default attributes."""
    issue = CritiqueIssue(
        issue_id="ISSUE-001",
        category=IssueCategory.FACTUAL_GROUNDING,
        severity=IssueSeverity.MAJOR,
        affected_section_id="sec_exec",
        affected_claim_id="claim_1",
        explanation="The claimed speedup is not supported by source chunk.",
        suggested_correction="Revise claim to reflect 3.2x instead of 10x.",
    )
    assert issue.issue_id == "ISSUE-001"
    assert issue.category == IssueCategory.FACTUAL_GROUNDING
    assert issue.severity == IssueSeverity.MAJOR
    assert issue.confidence == 1.0
    assert issue.is_resolved is False
    assert issue.supporting_source_ids == []


def test_critique_issue_non_empty_validation() -> None:
    """Test that issue_id, explanation, and suggested_correction cannot be empty strings."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        CritiqueIssue(
            issue_id="   ",
            category=IssueCategory.SCHEMA,
            severity=IssueSeverity.CRITICAL,
            explanation="Some explanation",
            suggested_correction="Some correction",
        )

    with pytest.raises(ValidationError, match="cannot be empty"):
        CritiqueIssue(
            issue_id="ISSUE-002",
            category=IssueCategory.SCHEMA,
            severity=IssueSeverity.CRITICAL,
            explanation="",
            suggested_correction="Some correction",
        )

    with pytest.raises(ValidationError, match="cannot be empty"):
        CritiqueIssue(
            issue_id="ISSUE-002",
            category=IssueCategory.SCHEMA,
            severity=IssueSeverity.CRITICAL,
            explanation="Valid explanation",
            suggested_correction="",
        )


def test_critique_issue_confidence_bounds() -> None:
    """Test confidence bounds between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        CritiqueIssue(
            issue_id="ISSUE-003",
            category=IssueCategory.FACTUAL_GROUNDING,
            severity=IssueSeverity.MINOR,
            explanation="Valid",
            suggested_correction="Valid",
            confidence=1.1,
        )

    with pytest.raises(ValidationError):
        CritiqueIssue(
            issue_id="ISSUE-003",
            category=IssueCategory.FACTUAL_GROUNDING,
            severity=IssueSeverity.MINOR,
            explanation="Valid",
            suggested_correction="Valid",
            confidence=-0.1,
        )


def test_critique_counts_and_properties() -> None:
    """Test that Critique counts critical, major, and minor issues properly."""
    c_issue = CritiqueIssue(
        issue_id="C-01",
        category=IssueCategory.CITATION_INTEGRITY,
        severity=IssueSeverity.CRITICAL,
        explanation="Hallucinated chunk ID",
        suggested_correction="Use valid chunk",
    )
    m_issue_1 = CritiqueIssue(
        issue_id="M-01",
        category=IssueCategory.FACTUAL_GROUNDING,
        severity=IssueSeverity.MAJOR,
        explanation="Ungrounded claim",
        suggested_correction="Add citation",
    )
    m_issue_2 = CritiqueIssue(
        issue_id="M-02",
        category=IssueCategory.LOGICAL_CONSISTENCY,
        severity=IssueSeverity.MAJOR,
        explanation="Contradiction across sections",
        suggested_correction="Align conclusions",
    )
    min_issue = CritiqueIssue(
        issue_id="MIN-01",
        category=IssueCategory.CLARITY_ORGANIZATION,
        severity=IssueSeverity.MINOR,
        explanation="Awkward sentence flow",
        suggested_correction="Rephrase sentence",
    )

    critique = Critique(
        round_number=1,
        issues=[c_issue, m_issue_1, m_issue_2, min_issue],
        summary="Found 1 critical, 2 major, and 1 minor defects.",
        should_revise=True,
    )

    assert critique.critical_count == 1
    assert critique.major_count == 2
    assert critique.minor_count == 1
    assert critique.round_number == 1
    assert critique.should_revise is True


def test_critique_empty_issues() -> None:
    """Test empty Critique object properties."""
    critique = Critique(round_number=0, issues=[], should_revise=False)
    assert critique.critical_count == 0
    assert critique.major_count == 0
    assert critique.minor_count == 0
    assert critique.summary == ""
