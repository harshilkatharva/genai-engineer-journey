"""Unit tests for CritiqueEngine: defect translation, severity classification, and revision thresholds."""

from __future__ import annotations

from self_improving_report_writer.models.critique import IssueCategory, IssueSeverity
from self_improving_report_writer.models.report import Report
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DefectCategory,
    DefectSeverity,
    DeterministicVerificationReport,
    VerificationIssue,
)
from self_improving_report_writer.reflection.critique_engine import CritiqueEngine


def test_translate_deterministic_defects(valid_report: Report) -> None:
    """Verify deterministic verification issues are translated with actionable recommendations."""
    engine = CritiqueEngine()

    det_report = DeterministicVerificationReport(
        passed=False,
        schema_valid=True,
        citations_valid=False,
        passages_matched=False,
        issues=[
            VerificationIssue(
                issue_id="DEFECT-001",
                category=DefectCategory.CITATION_INTEGRITY,
                severity=DefectSeverity.CRITICAL,
                message="Quotation 'fake text' missing from source chunk chunk_1.",
                section_id="sec_exec",
                claim_id="claim_exec_1",
            ),
            VerificationIssue(
                issue_id="DEFECT-002",
                category=DefectCategory.ORGANIZATION_CLARITY,
                severity=DefectSeverity.MINOR,
                message="Citation 'cite_2' is not referenced in prose.",
                section_id="sec_1",
            ),
        ],
    )

    critique = engine.generate_critique(
        round_number=1,
        report=valid_report,
        deterministic_report=det_report,
        acceptance_threshold=0.85,
    )

    assert critique.round_number == 1
    assert critique.critical_count == 1
    assert critique.minor_count == 1
    assert critique.should_revise is True

    crit_issue = critique.issues[0]
    assert crit_issue.severity == IssueSeverity.CRITICAL
    assert crit_issue.category == IssueCategory.CITATION_INTEGRITY
    assert "exact verbatim substring" in crit_issue.suggested_correction


def test_translate_semantic_grounding_defects(valid_report: Report) -> None:
    """Verify semantic grounding defects (contradiction, neutral, ungrounded prose) are correctly mapped."""
    engine = CritiqueEngine()

    grounding_report = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.50,
        issues=[
            VerificationIssue(
                issue_id="G2-001",
                category=DefectCategory.FACTUAL_GROUNDING,
                severity=DefectSeverity.CRITICAL,
                message="Claim 'claim_1' contradicts cited source evidence: Refuted.",
                section_id="sec_exec",
                claim_id="claim_1",
            ),
            VerificationIssue(
                issue_id="G2-002",
                category=DefectCategory.FACTUAL_GROUNDING,
                severity=DefectSeverity.MAJOR,
                message="Claim 'claim_2' is only weakly supported by cited evidence.",
                section_id="sec_1",
                claim_id="claim_2",
            ),
            VerificationIssue(
                issue_id="G2-003",
                category=DefectCategory.FACTUAL_GROUNDING,
                severity=DefectSeverity.MAJOR,
                message="Section 'sec_1' contains ungrounded factual assertion in prose: 'Battery cost $5B'.",
                section_id="sec_1",
            ),
        ],
    )

    critique = engine.generate_critique(
        round_number=2,
        report=valid_report,
        grounding_report=grounding_report,
        acceptance_threshold=0.85,
    )

    assert critique.critical_count == 1
    assert critique.major_count == 2
    assert critique.should_revise is True

    # Check contradiction correction
    contra_issue = next(i for i in critique.issues if i.severity == IssueSeverity.CRITICAL)
    assert "refutes cited source evidence" in contra_issue.suggested_correction

    # Check ungrounded prose correction
    prose_issue = next(i for i in critique.issues if "Battery cost" in i.explanation)
    assert (
        "Convert the substantive factual statement in prose into a formal structured Claim"
        in prose_issue.suggested_correction
    )


def test_translate_rubric_judge_deficits(valid_report: Report) -> None:
    """Verify failing rubric dimensions are mapped to issues with appropriate severity."""
    engine = CritiqueEngine()

    dim_scores = {
        "factual_accuracy": DimensionScore(
            dimension_name="factual_accuracy",
            score=0.75,
            passed=False,
            reasoning="Contains two hallucinated metrics not present in sources.",
        ),
        "completeness": DimensionScore(
            dimension_name="completeness",
            score=0.65,
            passed=False,
            reasoning="Omitted discussion of battery degradation caveats.",
        ),
        "clarity_organization": DimensionScore(
            dimension_name="clarity_organization",
            score=0.85,
            passed=True,
            reasoning="Well written.",
        ),
    }

    judge_verdict = ReportJudgeVerdict(
        composite_score=0.78,
        passed=False,
        dimension_scores=dim_scores,
        overall_feedback="Requires revision on facts and completeness.",
    )

    critique = engine.generate_critique(
        round_number=1,
        report=valid_report,
        judge_verdict=judge_verdict,
        acceptance_threshold=0.85,
    )

    # factual_accuracy < 0.85 -> CRITICAL
    assert critique.critical_count == 1
    # completeness < 0.70 -> MAJOR
    assert critique.major_count == 1
    assert critique.should_revise is True


def test_severity_based_revision_thresholds_critical(valid_report: Report) -> None:
    """Critical issues always trigger revision even if composite score is very high."""
    engine = CritiqueEngine()

    det_report = DeterministicVerificationReport(
        passed=False,
        schema_valid=True,
        citations_valid=False,
        passages_matched=False,
        issues=[
            VerificationIssue(
                issue_id="DEF-1",
                category=DefectCategory.CITATION_INTEGRITY,
                severity=DefectSeverity.CRITICAL,
                message="Quotation missing from source chunk.",
            )
        ],
    )

    judge_verdict = ReportJudgeVerdict(
        composite_score=0.95,  # Very high score
        passed=True,
        dimension_scores={},
        overall_feedback="Excellent prose.",
    )

    critique = engine.generate_critique(
        round_number=1,
        report=valid_report,
        deterministic_report=det_report,
        judge_verdict=judge_verdict,
        acceptance_threshold=0.85,
    )

    assert critique.should_revise is True


def test_severity_based_revision_thresholds_major(valid_report: Report) -> None:
    """Major issues trigger revision if score < threshold or grounding < 0.85."""
    engine = CritiqueEngine()

    grounding_report_low = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.70,  # Below 0.85
        issues=[
            VerificationIssue(
                issue_id="G2-1",
                category=DefectCategory.FACTUAL_GROUNDING,
                severity=DefectSeverity.MAJOR,
                message="Claim weakly supported.",
            )
        ],
    )

    judge_verdict_high = ReportJudgeVerdict(
        composite_score=0.90,  # Above threshold
        passed=True,
        dimension_scores={},
        overall_feedback="Good.",
    )

    critique = engine.generate_critique(
        round_number=1,
        report=valid_report,
        grounding_report=grounding_report_low,
        judge_verdict=judge_verdict_high,
        acceptance_threshold=0.85,
    )

    # Still revises because grounding_score < 0.85
    assert critique.should_revise is True


def test_severity_based_revision_thresholds_minor_only_terminates(valid_report: Report) -> None:
    """Minor issues only with composite score >= threshold terminates immediately."""
    engine = CritiqueEngine()

    det_report = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
        issues=[
            VerificationIssue(
                issue_id="DEF-1",
                category=DefectCategory.ORGANIZATION_CLARITY,
                severity=DefectSeverity.MINOR,
                message="Citation not marked in prose.",
            )
        ],
    )

    judge_verdict = ReportJudgeVerdict(
        composite_score=0.89,  # >= 0.85 threshold
        passed=True,
        dimension_scores={},
        overall_feedback="Ready.",
    )

    critique = engine.generate_critique(
        round_number=1,
        report=valid_report,
        deterministic_report=det_report,
        judge_verdict=judge_verdict,
        acceptance_threshold=0.85,
    )

    assert critique.critical_count == 0
    assert critique.major_count == 0
    assert critique.minor_count == 1
    # Terminates immediately!
    assert critique.should_revise is False


def test_deduplication_and_sorting(valid_report: Report) -> None:
    """Verify duplicate defects are deduplicated and sorted by severity."""
    engine = CritiqueEngine()

    det_report = DeterministicVerificationReport(
        passed=False,
        schema_valid=False,
        citations_valid=False,
        passages_matched=False,
        issues=[
            VerificationIssue(
                issue_id="D1",
                category=DefectCategory.ORGANIZATION_CLARITY,
                severity=DefectSeverity.MINOR,
                message="Citation not in prose.",
            ),
            VerificationIssue(
                issue_id="D2",
                category=DefectCategory.CITATION_INTEGRITY,
                severity=DefectSeverity.CRITICAL,
                message="Fabricated quotation.",
            ),
            VerificationIssue(
                issue_id="D3",
                category=DefectCategory.CITATION_INTEGRITY,
                severity=DefectSeverity.CRITICAL,
                message="Fabricated quotation.",  # Duplicate explanation & category
            ),
            VerificationIssue(
                issue_id="D4",
                category=DefectCategory.FACTUAL_GROUNDING,
                severity=DefectSeverity.MAJOR,
                message="Missing citation.",
            ),
        ],
    )

    critique = engine.generate_critique(
        round_number=1,
        report=valid_report,
        deterministic_report=det_report,
    )

    # 4 issues reduced to 3 after deduplication
    assert len(critique.issues) == 3
    # Check severity sort order: CRITICAL, then MAJOR, then MINOR
    assert critique.issues[0].severity == IssueSeverity.CRITICAL
    assert critique.issues[1].severity == IssueSeverity.MAJOR
    assert critique.issues[2].severity == IssueSeverity.MINOR
