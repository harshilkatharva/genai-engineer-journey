"""Regression tests enforcing strict transitivity in CandidateSelector ranking (FINDING-07)."""

from __future__ import annotations

import itertools

from self_improving_report_writer.models.report import Report
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import DeterministicVerificationReport
from self_improving_report_writer.reflection.candidate_selector import (
    CandidateSelector,
    ReportCandidate,
)


def _make_candidate(
    valid_report: Report,
    *,
    round_number: int,
    grounding_score: float = 0.95,
    composite_score: float = 0.90,
    subscore_each: float = 0.90,
) -> ReportCandidate:
    return ReportCandidate(
        round_number=round_number,
        report=valid_report,
        deterministic_report=DeterministicVerificationReport(
            passed=True,
            schema_valid=True,
            citations_valid=True,
            passages_matched=True,
        ),
        grounding_report=SemanticGroundingReport(
            passed=True,
            all_claims_grounded=True,
            grounding_score=grounding_score,
        ),
        judge_verdict=ReportJudgeVerdict(
            composite_score=composite_score,
            passed=True,
            dimension_scores={
                "factual_accuracy": DimensionScore(
                    dimension_name="factual_accuracy",
                    score=subscore_each,
                    passed=True,
                    reasoning="Accurate",
                ),
                "evidence_grounding": DimensionScore(
                    dimension_name="evidence_grounding",
                    score=subscore_each,
                    passed=True,
                    reasoning="Grounded",
                ),
            },
            overall_feedback="Solid",
            judge_model="gemini-3.1-flash-lite",
        ),
    )


def _assert_strict_weak_order(candidates: list[ReportCandidate]) -> None:
    """Assert irreflexivity, antisymmetry, transitivity, and permutation-invariant sorting."""
    for a in candidates:
        assert CandidateSelector.compare(a, a) == 0
        for b in candidates:
            ab = CandidateSelector.compare(a, b)
            ba = CandidateSelector.compare(b, a)
            assert ab == -ba
            for c in candidates:
                bc = CandidateSelector.compare(b, c)
                ac = CandidateSelector.compare(a, c)
                if ab > 0 and bc > 0:
                    assert ac > 0
                if ab < 0 and bc < 0:
                    assert ac < 0
                if ab == 0 and bc == 0:
                    assert ac == 0

    baseline_order = [cand.round_number for cand in CandidateSelector.rank_candidates(candidates)]
    for perm in itertools.permutations(candidates):
        perm_list = list(perm)
        ranked = CandidateSelector.rank_candidates(perm_list)
        assert [cand.round_number for cand in ranked] == baseline_order
        assert CandidateSelector.select_best(perm_list).round_number == baseline_order[0]


def test_candidate_comparator_is_strictly_transitive_across_small_score_steps(
    valid_report: Report,
) -> None:
    """Small <0.01 score increments that previously caused A>B>C>A cycles must rank transitively."""
    # Previously: |0.800-0.808|=0.008 < 0.01 (tied composite -> subscore decides),
    # |0.808-0.816|=0.008 < 0.01 (tied composite -> subscore decides),
    # |0.800-0.816|=0.016 >= 0.01 (composite decides in opposite direction -> cycle).
    cand_a = _make_candidate(
        valid_report, round_number=0, composite_score=0.800, subscore_each=0.95
    )
    cand_b = _make_candidate(
        valid_report, round_number=1, composite_score=0.808, subscore_each=0.90
    )
    cand_c = _make_candidate(
        valid_report, round_number=2, composite_score=0.816, subscore_each=0.85
    )

    _assert_strict_weak_order([cand_a, cand_b, cand_c])


def test_candidate_comparator_is_strictly_transitive_across_grounding_and_subscore_steps(
    valid_report: Report,
) -> None:
    """Small <0.01 grounding_score or grounding_subscore steps must also rank transitively."""
    grounding_cycle_candidates = [
        _make_candidate(valid_report, round_number=0, grounding_score=0.900, composite_score=0.95),
        _make_candidate(valid_report, round_number=1, grounding_score=0.908, composite_score=0.90),
        _make_candidate(valid_report, round_number=2, grounding_score=0.916, composite_score=0.85),
    ]
    _assert_strict_weak_order(grounding_cycle_candidates)

    subscore_cycle_candidates = [
        _make_candidate(valid_report, round_number=2, composite_score=0.90, subscore_each=0.900),
        _make_candidate(valid_report, round_number=1, composite_score=0.90, subscore_each=0.904),
        _make_candidate(valid_report, round_number=0, composite_score=0.90, subscore_each=0.908),
    ]
    _assert_strict_weak_order(subscore_cycle_candidates)


def test_is_regression_uses_transitive_quality_key_without_parsimony_bias(
    valid_report: Report,
) -> None:
    """is_regression must remain transitive across Tiers 1-6 while ignoring round_number."""
    prev_round = _make_candidate(valid_report, round_number=0, composite_score=0.90)
    equal_later_round = _make_candidate(valid_report, round_number=1, composite_score=0.90)
    worse_round = _make_candidate(valid_report, round_number=2, composite_score=0.85)

    assert not CandidateSelector.is_regression(prev_round, equal_later_round)
    assert CandidateSelector.is_regression(equal_later_round, worse_round)
    assert CandidateSelector.is_regression(prev_round, worse_round)
