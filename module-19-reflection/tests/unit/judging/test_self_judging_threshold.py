"""Regression tests for single-application and [0, 1] clamping of self-judging threshold (Observation 9)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from genai_core import LLMError, LLMManagerResponse
from pydantic import ValidationError

from self_improving_report_writer.core.budget import BudgetSummary
from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.judging.prompts import LLMJudgeEvaluation
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.outcome import ReportCandidate, ReportStatus
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import (
    SELF_JUDGING_THRESHOLD_PENALTY,
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
    compute_effective_threshold,
    get_default_rubric_config,
    resolve_effective_threshold,
)
from self_improving_report_writer.models.verification import DeterministicVerificationReport
from self_improving_report_writer.reflection.critique_engine import CritiqueEngine
from self_improving_report_writer.reflection.orchestrator import (
    ReflectionLoopOrchestrator,
    ReflectionRunResult,
)
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate


def _make_uniform_raw_eval(
    score: float, *, overrides: dict[str, float] | None = None
) -> LLMJudgeEvaluation:
    rubric = get_default_rubric_config()
    score_map = dict(overrides or {})
    return LLMJudgeEvaluation(
        dimension_scores=[
            DimensionScore(
                dimension_name=c.name,
                score=(dim_score := score_map.get(c.name, score)),
                passed=dim_score >= c.min_pass_score,
                reasoning=f"Uniform score {dim_score:.2f}",
            )
            for c in rubric.criteria
        ],
        overall_feedback="Consistent quality assessment.",
    )


def test_compute_effective_threshold_clamps_to_one_at_upper_boundaries() -> None:
    """Self-judging penalty (+0.05) must never produce a threshold greater than 1.0."""
    assert SELF_JUDGING_THRESHOLD_PENALTY == 0.05

    assert compute_effective_threshold(0.85, is_self_judged=False) == 0.85
    assert compute_effective_threshold(0.85, is_self_judged=True) == 0.90
    assert compute_effective_threshold(0.95, is_self_judged=True) == 1.0
    assert compute_effective_threshold(0.96, is_self_judged=True) == 1.0
    assert compute_effective_threshold(0.98, is_self_judged=True) == 1.0
    assert compute_effective_threshold(1.0, is_self_judged=True) == 1.0
    assert compute_effective_threshold(-0.1, is_self_judged=False) == 0.0
    assert compute_effective_threshold(1.2, is_self_judged=False) == 1.0


def test_resolve_effective_threshold_is_idempotent_across_repeated_calls() -> None:
    """Repeated threshold resolution using a verdict's effective_threshold must never compound +0.05."""
    verdict = ReportJudgeVerdict(
        composite_score=0.90,
        passed=True,
        dimension_scores={},
        overall_feedback="Evaluated",
        is_self_judged=True,
        effective_threshold=0.90,
    )

    threshold = 0.85
    for _ in range(5):
        threshold = resolve_effective_threshold(threshold, verdict)
        assert threshold == 0.90

    # Mock verdict without explicit effective_threshold applies +0.05 at most once from base
    legacy_verdict = ReportJudgeVerdict(
        composite_score=0.99,
        passed=True,
        dimension_scores={},
        overall_feedback="Legacy mock",
        is_self_judged=True,
    )
    assert resolve_effective_threshold(0.85, legacy_verdict) == 0.90
    assert resolve_effective_threshold(0.98, legacy_verdict) == 1.0


def test_report_judge_verdict_rejects_effective_threshold_above_one() -> None:
    """ReportJudgeVerdict schema must reject effective_threshold > 1.0."""
    with pytest.raises(ValidationError):
        ReportJudgeVerdict(
            composite_score=0.95,
            passed=True,
            dimension_scores={},
            overall_feedback="Invalid threshold",
            is_self_judged=True,
            effective_threshold=1.05,
        )


async def test_rubric_judge_applies_self_judging_bump_once_and_caps_at_one(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """RubricJudge must compute effective_threshold once per evaluation and clamp <= 1.0."""
    sources_dict = {s.chunk_id: s for s in sample_sources}

    # Case 1: High base_threshold (0.98) + self-judging -> capped at 1.0, perfect 1.0 score passes
    mock_llm_cap = AsyncMock()
    mock_llm_cap.complete_structure.return_value = LLMManagerResponse(
        structured_response=_make_uniform_raw_eval(1.0),
        model="gemini-3.5-flash-lite",
        usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    judge_cap = RubricJudge(
        llm_service=mock_llm_cap,
        config=ReportWriterConfig(acceptance_threshold=0.98),
        judge_model="gemini-3.5-flash-lite",
        generator_model="gemini-3.5-flash-lite",
    )
    verdict_cap = await judge_cap.evaluate(valid_report, sources_dict)
    assert verdict_cap.is_self_judged is True
    assert verdict_cap.effective_threshold == 1.0
    assert verdict_cap.passed is True

    # Case 2: Primary failure falls back to generator with base=0.97 -> still capped at 1.0
    mock_llm_fallback = AsyncMock()
    mock_llm_fallback.complete_structure.side_effect = [
        LLMManagerResponse(error=LLMError(provider="google", code="503", message="Unavailable")),
        LLMManagerResponse(
            structured_response=_make_uniform_raw_eval(1.0),
            model="gemini-3.5-flash-lite",
        ),
    ]
    judge_fallback = RubricJudge(
        llm_service=mock_llm_fallback,
        config=ReportWriterConfig(acceptance_threshold=0.97),
        judge_model="gemini-3.1-flash-lite",
        generator_model="gemini-3.5-flash-lite",
    )
    verdict_fallback = await judge_fallback.evaluate(valid_report, sources_dict)
    assert verdict_fallback.is_self_judged is True
    assert verdict_fallback.effective_threshold == 1.0
    assert verdict_fallback.passed is True

    # Case 3: Custom threshold 0.80 + runtime self-judging -> 0.85 (honors user config)
    # Composite ~0.866 (>= 0.85, would fail if wrongly bumped from default 0.85 -> 0.90)
    mock_llm_runtime = AsyncMock()
    mock_llm_runtime.complete_structure.return_value = LLMManagerResponse(
        structured_response=_make_uniform_raw_eval(0.86, overrides={"citation_correctness": 0.90}),
        model="gemini-3.5-flash-lite",  # Provider routed to generator model!
    )
    judge_runtime = RubricJudge(
        llm_service=mock_llm_runtime,
        config=ReportWriterConfig(acceptance_threshold=0.80),
        judge_model="gemini-3.1-flash-lite",
        generator_model="gemini-3.5-flash-lite",
    )
    verdict_runtime = await judge_runtime.evaluate(valid_report, sources_dict)
    assert verdict_runtime.is_self_judged is True
    assert verdict_runtime.effective_threshold == 0.85
    assert verdict_runtime.passed is True


async def test_pipeline_components_do_not_double_apply_penalty_or_ignore_custom_threshold(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Judge -> CritiqueEngine -> AcceptanceGate -> ReflectionRunResult must apply +0.05 at most once."""
    sources_dict = {s.chunk_id: s for s in sample_sources}

    # Base threshold = 0.85, self-judged -> effective_threshold = 0.90.
    # Score = 0.90 would pass at 0.90, but would FAIL if any component added +0.05 again (0.95).
    cfg = ReportWriterConfig(acceptance_threshold=0.85)
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=_make_uniform_raw_eval(0.90),
        model="gemini-3.5-flash-lite",
    )
    judge = RubricJudge(
        llm_service=mock_llm,
        config=cfg,
        judge_model="gemini-3.5-flash-lite",
        generator_model="gemini-3.5-flash-lite",
    )
    verdict = await judge.evaluate(valid_report, sources_dict)
    assert verdict.effective_threshold == 0.90
    assert verdict.passed is True

    det_ok = DeterministicVerificationReport(
        passed=True, schema_valid=True, citations_valid=True, passages_matched=True
    )
    g2_ok = SemanticGroundingReport(passed=True, all_claims_grounded=True, grounding_score=1.0)

    critique_engine = CritiqueEngine()
    # Even if caller passes base 0.85 OR already-adjusted 0.90 alongside verdict, threshold remains 0.90
    critique_from_base = critique_engine.generate_critique(
        round_number=1,
        report=valid_report,
        deterministic_report=det_ok,
        grounding_report=g2_ok,
        judge_verdict=verdict,
        acceptance_threshold=0.85,
    )
    critique_from_adjusted = critique_engine.generate_critique(
        round_number=1,
        report=valid_report,
        deterministic_report=det_ok,
        grounding_report=g2_ok,
        judge_verdict=verdict,
        acceptance_threshold=verdict.effective_threshold,
    )
    assert critique_from_base.should_revise is False
    assert critique_from_adjusted.should_revise is False

    candidate = ReportCandidate(
        round_number=1,
        report=valid_report,
        deterministic_report=det_ok,
        grounding_report=g2_ok,
        judge_verdict=verdict,
    )
    gate_result = AcceptanceGate(config=cfg).evaluate_candidate(candidate)
    assert gate_result.status == ReportStatus.ACCEPTED
    assert gate_result.is_accepted is True

    run_result = ReflectionRunResult(
        topic=valid_report.topic,
        best_candidate=candidate,
        all_candidates=[candidate],
        budget_summary=BudgetSummary(
            used_tokens=100,
            max_tokens=5000,
            used_cost_usd=0.001,
            max_cost_usd=0.20,
            elapsed_seconds=0.5,
            max_seconds=90.0,
            rounds_executed=1,
            max_rounds=3,
            is_exhausted=False,
        ),
        rounds_executed=1,
        termination_reason="ALL_ACCEPTANCE_CRITERIA_MET",
        acceptance_threshold=cfg.acceptance_threshold,
    )
    assert run_result.is_accepted is True


async def test_orchestrator_respects_custom_threshold_and_boundary_cap_in_end_to_end_run(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Orchestrator with high acceptance_threshold=0.98 and self-judging must cap at 1.0 and accept 1.0."""
    cfg = ReportWriterConfig(acceptance_threshold=0.98, max_reflection_rounds=1)
    mock_llm = AsyncMock()
    mock_llm.complete_structure.return_value = LLMManagerResponse(
        structured_response=_make_uniform_raw_eval(1.0),
        model=cfg.generator_model,
    )

    generator = AsyncMock()
    generator.generate_report.return_value = (valid_report, 100, 0.001)

    det_verifier = AsyncMock()
    det_verifier.verify = lambda *_args, **_kwargs: DeterministicVerificationReport(
        passed=True, schema_valid=True, citations_valid=True, passages_matched=True
    )

    grounding_verifier = AsyncMock()
    grounding_verifier.verify_grounding.return_value = SemanticGroundingReport(
        passed=True, all_claims_grounded=True, grounding_score=1.0
    )

    judge = RubricJudge(
        llm_service=mock_llm,
        config=cfg,
        judge_model=cfg.generator_model,
        generator_model=cfg.generator_model,
    )

    orchestrator = ReflectionLoopOrchestrator(
        llm_service=mock_llm,
        config=cfg,
        generator=generator,
        deterministic_verifier=det_verifier,
        grounding_verifier=grounding_verifier,
        judge=judge,
        critique_engine=CritiqueEngine(),
    )

    result = await orchestrator.run(topic=valid_report.topic, sources=sample_sources)

    assert result.best_candidate.judge_verdict.is_self_judged is True
    assert result.best_candidate.judge_verdict.effective_threshold == 1.0
    assert result.termination_reason == "ALL_ACCEPTANCE_CRITERIA_MET"
    assert result.is_accepted is True
    assert AcceptanceGate(config=cfg).evaluate_run_result(result).status == ReportStatus.ACCEPTED
