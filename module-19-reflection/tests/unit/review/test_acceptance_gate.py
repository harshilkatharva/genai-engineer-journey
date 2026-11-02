"""Unit tests for AcceptanceGate automated evaluation according to Section 6.1."""

from __future__ import annotations

from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.models.outcome import ReportStatus
from self_improving_report_writer.models.report import Report
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DeterministicVerificationReport,
)
from self_improving_report_writer.reflection.candidate_selector import ReportCandidate
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate


def _make_candidate(
    report: Report,
    passed_g1: bool = True,
    critical_count: int = 0,
    major_grounding_count: int = 0,
    composite_score: float = 0.90,
    factual_score: float = 0.90,
    evidence_score: float = 0.90,
    is_self_judged: bool = False,
) -> ReportCandidate:
    """Build mock candidate for acceptance testing."""
    det_rep = DeterministicVerificationReport(
        passed=passed_g1,
        schema_valid=passed_g1,
        citations_valid=passed_g1,
        passages_matched=passed_g1,
        issues=[],
        critical_count=critical_count,
        major_count=0,
        minor_count=0,
    )
    g2_rep = SemanticGroundingReport(
        passed=(critical_count == 0 and major_grounding_count == 0),
        all_claims_grounded=(critical_count == 0 and major_grounding_count == 0),
        grounding_score=0.95 if major_grounding_count == 0 else 0.70,
        claim_evaluations=[],
        issues=[],
        critical_count=0,
        major_count=major_grounding_count,
        minor_count=0,
    )
    dim_scores = {
        "factual_accuracy": DimensionScore(
            dimension_name="factual_accuracy",
            score=factual_score,
            passed=factual_score >= 0.85,
            reasoning="Accurate factual claims",
        ),
        "evidence_grounding": DimensionScore(
            dimension_name="evidence_grounding",
            score=evidence_score,
            passed=evidence_score >= 0.85,
            reasoning="Strong evidence grounding",
        ),
    }
    judge_verdict = ReportJudgeVerdict(
        composite_score=composite_score,
        passed=composite_score >= (0.90 if is_self_judged else 0.85),
        dimension_scores=dim_scores,
        overall_feedback="Evaluation feedback",
        is_self_judged=is_self_judged,
        judge_model="gemini-2.5-pro",
    )
    return ReportCandidate(
        round_number=1,
        report=report,
        deterministic_report=det_rep,
        grounding_report=g2_rep,
        judge_verdict=judge_verdict,
        tokens_used=1500,
        cost_usd=0.005,
    )


def test_acceptance_gate_accepts_clean_candidate(valid_report: Report) -> None:
    """AcceptanceGate should accept a candidate meeting all 5 conditions."""
    gate = AcceptanceGate()
    candidate = _make_candidate(valid_report)

    result = gate.evaluate_candidate(candidate, rounds_executed=1, run_id="run_123")

    assert result.status == ReportStatus.ACCEPTED
    assert result.is_accepted is True
    assert result.review_packet is None
    assert len(result.unresolved_reasons) == 0
    assert "meets all deterministic and semantic acceptance standards" in result.rationale


def test_acceptance_gate_rejects_gate1_failure(valid_report: Report) -> None:
    """AcceptanceGate must never accept a report when Gate 1 fails."""
    gate = AcceptanceGate()
    candidate = _make_candidate(valid_report, passed_g1=False)

    result = gate.evaluate_candidate(candidate, rounds_executed=1, run_id="run_123")

    assert result.status == ReportStatus.UNRESOLVED
    assert result.is_accepted is False
    assert result.review_packet is not None
    assert any("Gate 1 Deterministic Verification failed" in r for r in result.unresolved_reasons)


def test_acceptance_gate_rejects_critical_defects(valid_report: Report) -> None:
    """AcceptanceGate must reject a report with critical defects."""
    gate = AcceptanceGate()
    candidate = _make_candidate(valid_report, critical_count=1)

    result = gate.evaluate_candidate(candidate, rounds_executed=1, run_id="run_123")

    assert result.status == ReportStatus.UNRESOLVED
    assert result.is_accepted is False
    assert any("critical defects remain" in r for r in result.unresolved_reasons)


def test_acceptance_gate_rejects_major_grounding_defects(valid_report: Report) -> None:
    """AcceptanceGate must reject a report with major factual grounding defects."""
    gate = AcceptanceGate()
    candidate = _make_candidate(valid_report, major_grounding_count=2)

    result = gate.evaluate_candidate(candidate, rounds_executed=1, run_id="run_123")

    assert result.status == ReportStatus.UNRESOLVED
    assert result.is_accepted is False
    assert any("major factual grounding defects" in r for r in result.unresolved_reasons)


def test_acceptance_gate_rejects_subthreshold_composite_score(valid_report: Report) -> None:
    """AcceptanceGate must reject candidate if composite score < 0.85."""
    gate = AcceptanceGate()
    candidate = _make_candidate(valid_report, composite_score=0.82)

    result = gate.evaluate_candidate(candidate, rounds_executed=1, run_id="run_123")

    assert result.status == ReportStatus.UNRESOLVED
    assert result.is_accepted is False
    assert any("Composite rubric score 0.8200 is below" in r for r in result.unresolved_reasons)


def test_acceptance_gate_self_judged_threshold_bump(valid_report: Report) -> None:
    """AcceptanceGate must bump threshold from 0.85 to 0.90 if self-judged."""
    gate = AcceptanceGate(config=ReportWriterConfig(acceptance_threshold=0.85))
    # Score 0.88 would pass normally (0.88 >= 0.85), but fails self-judged (0.88 < 0.90)
    candidate = _make_candidate(valid_report, composite_score=0.88, is_self_judged=True)

    result = gate.evaluate_candidate(candidate, rounds_executed=1, run_id="run_123")

    assert result.status == ReportStatus.UNRESOLVED
    assert result.is_accepted is False
    assert any("acceptance threshold 0.90" in r for r in result.unresolved_reasons)


def test_acceptance_gate_rejects_failing_factual_accuracy(valid_report: Report) -> None:
    """AcceptanceGate must reject if factual accuracy < 0.85 even with high composite score."""
    gate = AcceptanceGate()
    candidate = _make_candidate(valid_report, composite_score=0.89, factual_score=0.75)

    result = gate.evaluate_candidate(candidate, rounds_executed=1, run_id="run_123")

    assert result.status == ReportStatus.UNRESOLVED
    assert result.is_accepted is False
    assert any(
        "Factual Accuracy dimension score 0.75 is below" in r for r in result.unresolved_reasons
    )


def test_acceptance_gate_rejects_failing_evidence_grounding(valid_report: Report) -> None:
    """AcceptanceGate must reject if evidence grounding < 0.85 even with high composite score."""
    gate = AcceptanceGate()
    candidate = _make_candidate(valid_report, composite_score=0.89, evidence_score=0.70)

    result = gate.evaluate_candidate(candidate, rounds_executed=1, run_id="run_123")

    assert result.status == ReportStatus.UNRESOLVED
    assert result.is_accepted is False
    assert any(
        "Evidence Grounding dimension score 0.70 is below" in r for r in result.unresolved_reasons
    )


def test_human_review_packet_populated_correctly(valid_report: Report) -> None:
    """HumanReviewPacket should be populated with all metadata when report is unresolved."""
    gate = AcceptanceGate()
    candidate = _make_candidate(valid_report, critical_count=1)

    result = gate.evaluate_candidate(
        candidate, rounds_executed=2, terminal_reason="MAX_ROUNDS_REACHED", run_id="test_run_999"
    )

    assert result.review_packet is not None
    packet = result.review_packet
    assert packet.run_id == "test_run_999"
    assert packet.topic == valid_report.topic
    assert packet.best_candidate_version == valid_report.version
    assert packet.total_rounds_executed == 2
    assert packet.terminal_reason == "MAX_ROUNDS_REACHED"
    assert packet.critical_defect_count == 1
    assert packet.requires_human_action is True
    assert "factual_accuracy" in packet.rubric_scores
