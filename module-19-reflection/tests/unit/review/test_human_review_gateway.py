"""Unit tests for HumanReviewGateway operator action dispatch and packet formatting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from self_improving_report_writer.models.critique import (
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.outcome import (
    HumanReviewDecision,
    HumanReviewPacket,
    ReportStatus,
)
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DeterministicVerificationReport,
)
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate
from self_improving_report_writer.review.human_review_gateway import HumanReviewGateway


@pytest.fixture
def sample_packet(valid_report: Report) -> HumanReviewPacket:
    """Fixture returning a mock HumanReviewPacket."""
    return HumanReviewPacket(
        run_id="run_quantum_123",
        topic="Quantum Annealing vs Classical Solvers",
        best_candidate=valid_report,
        best_candidate_version=1,
        total_rounds_executed=3,
        terminal_reason="MAX_ROUNDS_REACHED",
        deterministic_passed=True,
        critical_defect_count=0,
        major_defect_count=1,
        rubric_scores={"factual_accuracy": 0.88, "evidence_grounding": 0.82},
        unresolved_critiques=[
            CritiqueIssue(
                issue_id="ISSUE-001",
                category=IssueCategory.FACTUAL_GROUNDING,
                severity=IssueSeverity.MAJOR,
                explanation="Evidence grounding score 0.82 is below 0.85 threshold.",
                suggested_correction="Add citations for 100-node performance claims.",
            )
        ],
        requires_human_action=True,
    )


def test_format_review_packet(sample_packet: HumanReviewPacket) -> None:
    """format_review_packet should render structured markdown for operator inspection."""
    gateway = HumanReviewGateway()
    formatted = gateway.format_review_packet(sample_packet)

    assert "# Human Review Packet: Quantum Annealing vs Classical Solvers" in formatted
    assert "`run_quantum_123`" in formatted
    assert "v1" in formatted
    assert "MAX_ROUNDS_REACHED" in formatted
    assert "**factual_accuracy:** 0.88" in formatted
    assert "**evidence_grounding:** 0.82" in formatted
    assert "[MAJOR] (factual_grounding)" in formatted
    assert "Add citations for 100-node performance claims." in formatted


async def test_process_decision_approve_override(sample_packet: HumanReviewPacket) -> None:
    """Operator approval override should set status to HUMAN_APPROVED without mutating verifications."""
    gateway = HumanReviewGateway()
    decision = HumanReviewDecision(
        run_id=sample_packet.run_id,
        action="APPROVE_OVERRIDE",
        reviewer_notes="Approved for internal preliminary briefing.",
    )

    outcome = await gateway.process_decision(decision=decision, packet=sample_packet)

    assert outcome.status == ReportStatus.HUMAN_APPROVED
    assert outcome.final_report.title == sample_packet.best_candidate.title
    assert "Approved for internal preliminary briefing" in outcome.summary
    assert outcome.revised_candidate is None


async def test_process_decision_reject(sample_packet: HumanReviewPacket) -> None:
    """Operator rejection should set status to HUMAN_REJECTED."""
    gateway = HumanReviewGateway()
    decision = HumanReviewDecision(
        run_id=sample_packet.run_id,
        action="REJECT",
        reviewer_notes="Defects are too severe for client presentation.",
    )

    outcome = await gateway.process_decision(decision=decision, packet=sample_packet)

    assert outcome.status == ReportStatus.HUMAN_REJECTED
    assert outcome.final_report.title == sample_packet.best_candidate.title
    assert "Defects are too severe" in outcome.summary
    assert outcome.revised_candidate is None


async def test_process_decision_request_revision_success(
    valid_report: Report,
    sample_sources: list[SourceReference],
    sample_packet: HumanReviewPacket,
) -> None:
    """REQUEST_REVISION should revise report, increment version, set human provenance, and re-verify."""
    revised_rep = valid_report.model_copy(deep=True)
    revised_rep.version = 1

    revision_engine = MagicMock()
    revision_engine.revise_report = AsyncMock(return_value=(revised_rep, 1000, 0.003))

    det_verifier = MagicMock()
    det_verifier.verify.return_value = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
    )

    grounding_verifier = MagicMock()
    grounding_verifier.verify_grounding = AsyncMock(
        return_value=SemanticGroundingReport(
            passed=True,
            all_claims_grounded=True,
            grounding_score=0.96,
            claim_evaluations=[],
            issues=[],
            critical_count=0,
            major_count=0,
            minor_count=0,
        )
    )

    judge = MagicMock()
    judge.evaluate = AsyncMock(
        return_value=ReportJudgeVerdict(
            composite_score=0.92,
            passed=True,
            dimension_scores={
                "factual_accuracy": DimensionScore(
                    dimension_name="factual_accuracy",
                    score=0.92,
                    passed=True,
                    reasoning="All factual claims verified",
                ),
                "evidence_grounding": DimensionScore(
                    dimension_name="evidence_grounding",
                    score=0.92,
                    passed=True,
                    reasoning="Direct evidence support",
                ),
            },
            overall_feedback="Excellent revisions.",
        )
    )

    acceptance_gate = AcceptanceGate()
    gateway = HumanReviewGateway(
        acceptance_gate=acceptance_gate,
        deterministic_verifier=det_verifier,
    )

    decision = HumanReviewDecision(
        run_id=sample_packet.run_id,
        action="REQUEST_REVISION",
        reviewer_notes="Expand discussion of classical speedups on small graphs.",
    )

    outcome = await gateway.process_decision(
        decision=decision,
        packet=sample_packet,
        sources=sample_sources,
        human_revision_round=1,
        revision_engine=revision_engine,
        grounding_verifier=grounding_verifier,
        judge=judge,
    )

    assert outcome.status == ReportStatus.ACCEPTED
    assert outcome.final_report.created_by == "human_revision"
    assert outcome.final_report.version == sample_packet.best_candidate_version + 1
    assert outcome.revised_candidate is not None
    assert outcome.revised_candidate.judge_verdict.composite_score == 0.92
    revision_engine.revise_report.assert_awaited_once()


async def test_process_decision_request_revision_exceeds_max_rounds(
    sample_packet: HumanReviewPacket,
    sample_sources: list[SourceReference],
) -> None:
    """REQUEST_REVISION should raise ValueError when human revision rounds exceed maximum (2)."""
    gateway = HumanReviewGateway(max_human_revision_rounds=2)
    decision = HumanReviewDecision(
        run_id=sample_packet.run_id,
        action="REQUEST_REVISION",
        reviewer_notes="Another revision please.",
    )

    revision_engine = MagicMock()

    with pytest.raises(ValueError, match="Maximum human revision rounds exceeded"):
        await gateway.process_decision(
            decision=decision,
            packet=sample_packet,
            sources=sample_sources,
            human_revision_round=3,  # 3 > 2
            revision_engine=revision_engine,
        )


async def test_process_decision_request_revision_missing_deps_raises(
    sample_packet: HumanReviewPacket,
) -> None:
    """REQUEST_REVISION without revision_engine or sources must raise ValueError."""
    gateway = HumanReviewGateway()
    decision = HumanReviewDecision(
        run_id=sample_packet.run_id,
        action="REQUEST_REVISION",
        reviewer_notes="Revise",
    )

    with pytest.raises(ValueError, match="revision_engine and sources are required"):
        await gateway.process_decision(
            decision=decision,
            packet=sample_packet,
            sources=None,
        )
