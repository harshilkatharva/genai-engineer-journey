"""Unit tests for OscillationDetector: content hashing, cycle detection, and critique stagnation."""

from __future__ import annotations

from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.report import Report, ReportSection
from self_improving_report_writer.reflection.oscillation_detector import OscillationDetector


def _create_report_variant(text_suffix: str) -> Report:
    """Helper to create Report variants with different section texts."""
    return Report(
        topic="Solid State Batteries",
        title="Solid State Overview",
        executive_summary=ReportSection(
            section_id="sec_exec",
            title="Executive Summary",
            content=f"Executive summary content variant {text_suffix}",
            claims=[],
        ),
        sections=[
            ReportSection(
                section_id="sec_body",
                title="Body Section",
                content=f"Body analysis content variant {text_suffix}",
                claims=[],
            )
        ],
        conclusion=ReportSection(
            section_id="sec_conc",
            title="Conclusion",
            content=f"Conclusion content variant {text_suffix}",
            claims=[],
        ),
    )


def _create_critique(issues: list[CritiqueIssue]) -> Critique:
    """Helper to create Critique."""
    return Critique(
        round_number=0,
        issues=issues,
        summary="Test critique",
        should_revise=True,
    )


def test_no_oscillation_on_first_round() -> None:
    """First round candidate never triggers oscillation."""
    detector = OscillationDetector()
    report_r0 = _create_report_variant("v0")
    critique_r0 = _create_critique([])

    res = detector.check_oscillation(0, report_r0, critique_r0)

    assert res.has_oscillation is False
    assert res.oscillation_type == "none"
    assert len(detector.content_history) == 1
    assert len(detector.critique_history) == 1


def test_content_cycle_detection() -> None:
    """Detects oscillation when report returns to a previously generated version."""
    detector = OscillationDetector()

    report_a = _create_report_variant("A")
    report_b = _create_report_variant("B")

    issue_1 = CritiqueIssue(
        issue_id="I1",
        category=IssueCategory.COMPLETENESS,
        severity=IssueSeverity.MAJOR,
        explanation="Missing details",
        suggested_correction="Add details",
    )
    issue_2 = CritiqueIssue(
        issue_id="I2",
        category=IssueCategory.CLARITY_ORGANIZATION,
        severity=IssueSeverity.MINOR,
        explanation="Too verbose",
        suggested_correction="Shorten details",
    )

    # Round 0: Report A
    res0 = detector.check_oscillation(0, report_a, _create_critique([issue_1]))
    assert res0.has_oscillation is False

    # Round 1: Report B
    res1 = detector.check_oscillation(1, report_b, _create_critique([issue_2]))
    assert res1.has_oscillation is False

    # Round 2: Report A again (cyclical regression!)
    res2 = detector.check_oscillation(2, report_a, _create_critique([issue_1]))
    assert res2.has_oscillation is True
    assert res2.oscillation_type == "content_cycle"
    assert "Content oscillation cycle detected in round 2" in res2.details


def test_critique_stagnation_detection() -> None:
    """Detects stagnation when the issue set is unchanged across consecutive rounds."""
    detector = OscillationDetector()

    report_a = _create_report_variant("A")
    report_b = _create_report_variant("B")  # Text changed, but issues remained identical

    stagnant_issues = [
        CritiqueIssue(
            issue_id="I1",
            category=IssueCategory.FACTUAL_GROUNDING,
            severity=IssueSeverity.CRITICAL,
            affected_claim_id="claim_1",
            explanation="Unresolved contradiction",
            suggested_correction="Fix contradiction",
        )
    ]

    # Round 0: Report A with issue I1
    res0 = detector.check_oscillation(0, report_a, _create_critique(stagnant_issues))
    assert res0.has_oscillation is False

    # Round 1: Report B, but critique still has identical issue I1
    res1 = detector.check_oscillation(1, report_b, _create_critique(stagnant_issues))
    assert res1.has_oscillation is True
    assert res1.oscillation_type == "critique_stagnation"
    assert "Critique stagnation detected in round 1" in res1.details


def test_clean_progression_across_rounds() -> None:
    """Verifies that monotonic improvements across rounds do not trigger false positives."""
    detector = OscillationDetector()

    report_0 = _create_report_variant("step0")
    report_1 = _create_report_variant("step1")
    report_2 = _create_report_variant("step2")

    issues_r0 = [
        CritiqueIssue(
            issue_id="I1",
            category=IssueCategory.FACTUAL_GROUNDING,
            severity=IssueSeverity.CRITICAL,
            explanation="Fix contradiction",
            suggested_correction="Fix",
        ),
        CritiqueIssue(
            issue_id="I2",
            category=IssueCategory.COMPLETENESS,
            severity=IssueSeverity.MAJOR,
            explanation="Expand limitations",
            suggested_correction="Expand",
        ),
    ]

    issues_r1 = [
        CritiqueIssue(
            issue_id="I2",
            category=IssueCategory.COMPLETENESS,
            severity=IssueSeverity.MAJOR,
            explanation="Expand limitations",
            suggested_correction="Expand",
        )
    ]

    issues_r2: list[CritiqueIssue] = []

    res0 = detector.check_oscillation(0, report_0, _create_critique(issues_r0))
    res1 = detector.check_oscillation(1, report_1, _create_critique(issues_r1))
    res2 = detector.check_oscillation(2, report_2, _create_critique(issues_r2))

    assert res0.has_oscillation is False
    assert res1.has_oscillation is False
    assert res2.has_oscillation is False


def test_detector_reset() -> None:
    """Verifies that reset clears all historical fingerprints."""
    detector = OscillationDetector()
    report = _create_report_variant("base")
    critique = _create_critique([])

    detector.check_oscillation(0, report, critique)
    assert len(detector.content_history) == 1

    detector.reset()
    assert len(detector.content_history) == 0
    assert len(detector.critique_history) == 0

    # Checking identical report after reset does not trigger oscillation
    res = detector.check_oscillation(0, report, critique)
    assert res.has_oscillation is False


def test_normalization_insensitivity() -> None:
    """Whitespace variations do not create spurious hash differences."""
    report_1 = _create_report_variant("normal text")

    # Same content with varied spacing and tabs
    report_2 = Report(
        topic="Solid State Batteries",
        title="  Solid   State   Overview  ",
        executive_summary=ReportSection(
            section_id="sec_exec",
            title="Executive Summary",
            content="Executive  summary   content   variant   normal   text\n",
            claims=[],
        ),
        sections=[
            ReportSection(
                section_id="sec_body",
                title="Body Section",
                content="Body \t analysis content \n variant normal text",
                claims=[],
            )
        ],
        conclusion=ReportSection(
            section_id="sec_conc",
            title="Conclusion",
            content="Conclusion  content variant  normal text",
            claims=[],
        ),
    )

    hash1 = OscillationDetector.compute_content_fingerprint(report_1)
    hash2 = OscillationDetector.compute_content_fingerprint(report_2)

    assert hash1 == hash2
