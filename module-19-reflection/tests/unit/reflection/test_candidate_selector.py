"""Unit tests for CandidateSelector and deterministic hierarchical comparison."""

from __future__ import annotations

import pytest

from self_improving_report_writer.models.report import Report
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DeterministicVerificationReport,
)
from self_improving_report_writer.reflection.candidate_selector import (
    CandidateSelector,
    ReportCandidate,
)


def _make_candidate(
    round_number: int,
    report: Report,
    passed_g1: bool = True,
    critical_count: int = 0,
    major_count: int = 0,
    all_grounded: bool = True,
    grounding_score: float = 1.0,
    composite_score: float = 0.90,
    factual_score: float = 0.90,
    evidence_score: float = 0.90,
    tokens_used: int = 1000,
    cost_usd: float = 0.01,
) -> ReportCandidate:
    """Helper factory for creating mock ReportCandidate instances."""
    det_rep = DeterministicVerificationReport(
        passed=passed_g1,
        schema_valid=passed_g1,
        citations_valid=passed_g1,
        passages_matched=passed_g1,
        issues=[],
        critical_count=critical_count,
        major_count=major_count,
        minor_count=0,
    )
    g2_rep = SemanticGroundingReport(
        passed=(critical_count == 0 and major_count == 0 and all_grounded),
        all_claims_grounded=all_grounded,
        grounding_score=grounding_score,
        claim_evaluations=[],
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
    )
    dim_scores = {
        "factual_accuracy": DimensionScore(
            dimension_name="factual_accuracy",
            score=factual_score,
            passed=factual_score >= 0.85,
            reasoning="Factual accuracy evaluation",
        ),
        "evidence_grounding": DimensionScore(
            dimension_name="evidence_grounding",
            score=evidence_score,
            passed=evidence_score >= 0.85,
            reasoning="Evidence grounding evaluation",
        ),
    }
    judge_verdict = ReportJudgeVerdict(
        composite_score=composite_score,
        passed=composite_score >= 0.85,
        dimension_scores=dim_scores,
        overall_feedback="Feedback",
        judge_model="gemini-3.1-flash-lite",
    )
    return ReportCandidate(
        round_number=round_number,
        report=report,
        deterministic_report=det_rep,
        grounding_report=g2_rep,
        judge_verdict=judge_verdict,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


def test_compare_tier1_deterministic_pass_beats_fail(valid_report: Report) -> None:
    """Tier 1: Gate 1 pass must strictly beat Gate 1 fail regardless of rubric scores."""
    c_pass = _make_candidate(
        round_number=1, report=valid_report, passed_g1=True, composite_score=0.70
    )
    c_fail = _make_candidate(
        round_number=0, report=valid_report, passed_g1=False, composite_score=0.99
    )

    assert CandidateSelector.compare(c_pass, c_fail) > 0
    assert CandidateSelector.compare(c_fail, c_pass) < 0


def test_compare_tier2_critical_defect_count(valid_report: Report) -> None:
    """Tier 2: Fewer critical defects strictly beat more critical defects."""
    c_zero_crit = _make_candidate(
        round_number=1, report=valid_report, critical_count=0, composite_score=0.75
    )
    c_one_crit = _make_candidate(
        round_number=0, report=valid_report, critical_count=1, composite_score=0.95
    )

    assert CandidateSelector.compare(c_zero_crit, c_one_crit) > 0
    assert CandidateSelector.compare(c_one_crit, c_zero_crit) < 0


def test_compare_tier3_major_defect_count(valid_report: Report) -> None:
    """Tier 3: Fewer major defects strictly beat more major defects."""
    c_zero_maj = _make_candidate(
        round_number=1, report=valid_report, major_count=0, composite_score=0.80
    )
    c_two_maj = _make_candidate(
        round_number=0, report=valid_report, major_count=2, composite_score=0.90
    )

    assert CandidateSelector.compare(c_zero_maj, c_two_maj) > 0
    assert CandidateSelector.compare(c_two_maj, c_zero_maj) < 0


def test_compare_tier4_grounding_status_and_score(valid_report: Report) -> None:
    """Tier 4: All claims grounded beats not grounded; higher grounding score beats lower."""
    c_grounded = _make_candidate(
        round_number=1, report=valid_report, all_grounded=True, grounding_score=0.90
    )
    c_ungrounded = _make_candidate(
        round_number=0, report=valid_report, all_grounded=False, grounding_score=0.90
    )

    assert CandidateSelector.compare(c_grounded, c_ungrounded) > 0

    c_g_high = _make_candidate(
        round_number=1, report=valid_report, all_grounded=True, grounding_score=0.95
    )
    c_g_low = _make_candidate(
        round_number=0, report=valid_report, all_grounded=True, grounding_score=0.85
    )

    assert CandidateSelector.compare(c_g_high, c_g_low) > 0


def test_compare_tier5_composite_rubric_score(valid_report: Report) -> None:
    """Tier 5: Higher composite score strictly beats lower composite score."""
    c_high = _make_candidate(round_number=1, report=valid_report, composite_score=0.92)
    c_low = _make_candidate(round_number=0, report=valid_report, composite_score=0.85)

    assert CandidateSelector.compare(c_high, c_low) > 0
    assert CandidateSelector.compare(c_low, c_high) < 0


def test_compare_tier6_grounding_subscores(valid_report: Report) -> None:
    """Tier 6: Higher grounding sub-scores (factual + evidence) beats lower."""
    c_high_sub = _make_candidate(
        round_number=1,
        report=valid_report,
        composite_score=0.88,
        factual_score=0.95,
        evidence_score=0.95,
    )
    c_low_sub = _make_candidate(
        round_number=0,
        report=valid_report,
        composite_score=0.88,
        factual_score=0.85,
        evidence_score=0.85,
    )

    assert CandidateSelector.compare(c_high_sub, c_low_sub) > 0
    assert CandidateSelector.compare(c_low_sub, c_high_sub) < 0


def test_compare_tier7_parsimony_tiebreaker(valid_report: Report) -> None:
    """Tier 7: Earlier round candidate preferred to conserve cost when all else is tied."""
    c_round0 = _make_candidate(round_number=0, report=valid_report, composite_score=0.88)
    c_round1 = _make_candidate(round_number=1, report=valid_report, composite_score=0.88)

    assert CandidateSelector.compare(c_round0, c_round1) > 0
    assert CandidateSelector.compare(c_round1, c_round0) < 0


def test_select_best_picks_optimal_amid_regressions(valid_report: Report) -> None:
    """Select best must pick candidate from earlier round when later round regresses."""
    c0 = _make_candidate(
        round_number=0, report=valid_report, critical_count=1, composite_score=0.75
    )
    c1 = _make_candidate(
        round_number=1, report=valid_report, critical_count=0, composite_score=0.92
    )
    # Round 2 regressed by failing Gate 1
    c2 = _make_candidate(round_number=2, report=valid_report, passed_g1=False, composite_score=0.95)

    candidates = [c0, c1, c2]
    best = CandidateSelector.select_best(candidates)
    assert best.round_number == 1
    assert best.judge_verdict.composite_score == 0.92


def test_rank_candidates(valid_report: Report) -> None:
    """Rank candidates should sort from best to worst."""
    c0 = _make_candidate(round_number=0, report=valid_report, critical_count=1)
    c1 = _make_candidate(
        round_number=1, report=valid_report, critical_count=0, composite_score=0.90
    )
    c2 = _make_candidate(
        round_number=2, report=valid_report, critical_count=0, composite_score=0.80
    )

    ranked = CandidateSelector.rank_candidates([c0, c2, c1])
    assert [c.round_number for c in ranked] == [1, 2, 0]


def test_is_regression(valid_report: Report) -> None:
    """Is regression should return True when new candidate is strictly worse."""
    c_good = _make_candidate(round_number=1, report=valid_report, composite_score=0.90)
    c_worse = _make_candidate(round_number=2, report=valid_report, composite_score=0.80)

    assert CandidateSelector.is_regression(previous_candidate=c_good, current_candidate=c_worse)
    assert not CandidateSelector.is_regression(previous_candidate=c_worse, current_candidate=c_good)


def test_select_best_empty_raises() -> None:
    """Select best on empty candidate list must raise ValueError."""
    with pytest.raises(ValueError, match="Cannot select best"):
        CandidateSelector.select_best([])
