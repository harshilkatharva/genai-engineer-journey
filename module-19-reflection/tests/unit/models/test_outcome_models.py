"""Unit tests for outcome and review workflow domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_improving_report_writer.models.critique import (
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.outcome import (
    AcceptanceResult,
    HumanReviewDecision,
    HumanReviewOutcome,
    HumanReviewPacket,
    ReportCandidate,
    ReportStatus,
)
from self_improving_report_writer.models.report import Report, ReportSection
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DeterministicVerificationReport,
)


def _make_dummy_report() -> Report:
    exec_sec = ReportSection(
        section_id="sec_exec", title="Executive Summary", content="Exec content", claims=[]
    )
    concl_sec = ReportSection(
        section_id="sec_concl", title="Conclusion", content="Concl content", claims=[]
    )
    return Report(
        topic="Test",
        title="Report Title",
        executive_summary=exec_sec,
        sections=[],
        conclusion=concl_sec,
        version=1,
    )


def _make_dummy_candidate(
    round_number: int = 1,
    det_passed: bool = True,
    det_crit: int = 0,
    det_maj: int = 0,
    grd_passed: bool = True,
    grd_crit: int = 0,
    grd_maj: int = 0,
    comp_score: float = 0.90,
    fa_score: float = 0.92,
    eg_score: float = 0.88,
) -> ReportCandidate:
    report = _make_dummy_report()
    det_rep = DeterministicVerificationReport(
        passed=det_passed,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
        critical_count=det_crit,
        major_count=det_maj,
    )
    grd_rep = SemanticGroundingReport(
        passed=grd_passed,
        all_claims_grounded=grd_passed,
        grounding_score=0.90,
        critical_count=grd_crit,
        major_count=grd_maj,
    )
    dim_scores = {
        "factual_accuracy": DimensionScore(
            dimension_name="factual_accuracy", score=fa_score, passed=True, reasoning="ok"
        ),
        "evidence_grounding": DimensionScore(
            dimension_name="evidence_grounding", score=eg_score, passed=True, reasoning="ok"
        ),
    }
    verdict = ReportJudgeVerdict(
        composite_score=comp_score,
        passed=True,
        dimension_scores=dim_scores,
        overall_feedback="Good",
    )
    return ReportCandidate(
        round_number=round_number,
        report=report,
        deterministic_report=det_rep,
        grounding_report=grd_rep,
        judge_verdict=verdict,
        tokens_used=1500,
        cost_usd=0.015,
    )


def test_report_status_enum() -> None:
    """Test ReportStatus enum members."""
    assert ReportStatus.ACCEPTED == "ACCEPTED"
    assert ReportStatus.UNRESOLVED == "UNRESOLVED"
    assert ReportStatus.HUMAN_APPROVED == "HUMAN_APPROVED"
    assert ReportStatus.HUMAN_REJECTED == "HUMAN_REJECTED"


def test_report_candidate_properties() -> None:
    """Test ReportCandidate computed properties."""
    cand = _make_dummy_candidate(
        det_crit=1,
        det_maj=2,
        grd_crit=0,
        grd_maj=1,
        comp_score=0.87,
        fa_score=0.91,
        eg_score=0.83,
    )
    assert cand.total_critical_count == 1
    assert cand.total_major_count == 3
    assert cand.critical_defect_count == 1
    assert cand.major_defect_count == 3
    assert cand.grounding_subscore == 1.74  # 0.91 + 0.83
    assert cand.deterministic_passed is True
    assert cand.all_claims_grounded is True
    assert cand.composite_score == 0.87
    assert cand.factual_accuracy_score == 0.91
    assert cand.evidence_grounding_score == 0.83


def test_report_candidate_missing_dimensions() -> None:
    """Test ReportCandidate properties when dimensions are missing in judge verdict."""
    report = _make_dummy_report()
    det_rep = DeterministicVerificationReport(
        passed=True, schema_valid=True, citations_valid=True, passages_matched=True
    )
    grd_rep = SemanticGroundingReport(passed=True, all_claims_grounded=True, grounding_score=1.0)
    verdict = ReportJudgeVerdict(
        composite_score=0.85, passed=True, dimension_scores={}, overall_feedback="ok"
    )
    cand = ReportCandidate(
        round_number=0,
        report=report,
        deterministic_report=det_rep,
        grounding_report=grd_rep,
        judge_verdict=verdict,
    )
    assert cand.grounding_subscore == 0.0
    assert cand.factual_accuracy_score == 0.0
    assert cand.evidence_grounding_score == 0.0


def test_human_review_packet_validation() -> None:
    """Test HumanReviewPacket validation and boundaries."""
    report = _make_dummy_report()
    issue = CritiqueIssue(
        issue_id="ISSUE-01",
        category=IssueCategory.FACTUAL_GROUNDING,
        severity=IssueSeverity.MAJOR,
        explanation="Ungrounded assertion",
        suggested_correction="Add evidence",
    )
    packet = HumanReviewPacket(
        run_id="run_123",
        topic="Quantum Computing",
        best_candidate=report,
        best_candidate_version=1,
        total_rounds_executed=3,
        terminal_reason="BUDGET_EXCEEDED: MAX_ROUNDS_REACHED",
        deterministic_passed=True,
        critical_defect_count=0,
        major_defect_count=1,
        rubric_scores={"factual_accuracy": 0.82},
        unresolved_critiques=[issue],
        requires_human_action=True,
    )
    assert packet.run_id == "run_123"
    assert packet.best_candidate_version == 1
    assert packet.requires_human_action is True
    assert len(packet.unresolved_critiques) == 1

    # best_candidate_version must be >= 1
    with pytest.raises(ValidationError):
        HumanReviewPacket(
            run_id="run_123",
            topic="QC",
            best_candidate=report,
            best_candidate_version=0,  # invalid (< 1)
            total_rounds_executed=1,
            terminal_reason="reason",
            deterministic_passed=True,
            critical_defect_count=0,
            major_defect_count=0,
        )


def test_human_review_decision_valid() -> None:
    """Test HumanReviewDecision valid actions."""
    for action in ["APPROVE_OVERRIDE", "REQUEST_REVISION", "REJECT"]:
        decision = HumanReviewDecision(
            run_id="run_456",
            action=action,  # type: ignore[arg-type]
            reviewer_notes=f"Selected {action}",
        )
        assert decision.action == action

    with pytest.raises(ValidationError):
        HumanReviewDecision(
            run_id="run_456",
            action="INVALID_ACTION",  # type: ignore[arg-type]
            reviewer_notes="Should fail",
        )


def test_acceptance_result_and_human_review_outcome() -> None:
    """Test AcceptanceResult and HumanReviewOutcome model instantiations."""
    cand = _make_dummy_candidate()
    acc_res = AcceptanceResult(
        status=ReportStatus.ACCEPTED,
        is_accepted=True,
        best_candidate=cand,
        rationale="All criteria met",
        unresolved_reasons=[],
    )
    assert acc_res.status == ReportStatus.ACCEPTED
    assert acc_res.is_accepted is True
    assert acc_res.review_packet is None

    decision = HumanReviewDecision(
        run_id="run_789",
        action="APPROVE_OVERRIDE",
        reviewer_notes="Approved despite minor defects",
    )
    outcome = HumanReviewOutcome(
        run_id="run_789",
        status=ReportStatus.HUMAN_APPROVED,
        final_report=cand.report,
        decision=decision,
        summary="Report approved by human operator.",
    )
    assert outcome.status == ReportStatus.HUMAN_APPROVED
    assert outcome.final_report.topic == "Test"
