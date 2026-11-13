"""Unit tests for integrated AcceptanceGate and ReflectionMemoryManager hooks in ReflectionLoopOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from self_improving_report_writer.core.budget import BudgetSummary
from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.memory.reflection_memory_manager import ReflectionMemoryManager
from self_improving_report_writer.models.outcome import ReportStatus
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DeterministicVerificationReport,
)
from self_improving_report_writer.reflection.candidate_selector import ReportCandidate
from self_improving_report_writer.reflection.orchestrator import (
    ReflectionLoopOrchestrator,
    ReflectionRunResult,
)
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate


def _make_verdict(
    composite_score: float = 0.90,
    factual_score: float = 0.90,
    evidence_score: float = 0.90,
    is_self_judged: bool = False,
) -> ReportJudgeVerdict:
    """Create a ReportJudgeVerdict with all required grading dimensions."""
    dim_scores = {
        "factual_accuracy": DimensionScore(
            dimension_name="factual_accuracy",
            score=factual_score,
            passed=factual_score >= 0.85,
            reasoning="Factual check",
        ),
        "evidence_grounding": DimensionScore(
            dimension_name="evidence_grounding",
            score=evidence_score,
            passed=evidence_score >= 0.85,
            reasoning="Evidence check",
        ),
    }
    return ReportJudgeVerdict(
        composite_score=composite_score,
        passed=composite_score >= (0.90 if is_self_judged else 0.85),
        dimension_scores=dim_scores,
        overall_feedback="Mock judge feedback",
        is_self_judged=is_self_judged,
        judge_model="gemini-3.1-flash-lite",
        tokens_used=120,
        cost_usd=0.0005,
    )


def _make_g1(passed: bool = True, critical_count: int = 0) -> DeterministicVerificationReport:
    return DeterministicVerificationReport(
        passed=passed,
        schema_valid=passed,
        citations_valid=passed,
        passages_matched=passed,
        issues=[],
        critical_count=critical_count,
        major_count=0,
        minor_count=0,
    )


def _make_g2(
    passed: bool = True,
    major_count: int = 0,
    grounding_score: float = 0.95,
) -> SemanticGroundingReport:
    return SemanticGroundingReport(
        passed=passed,
        all_claims_grounded=passed and major_count == 0,
        grounding_score=grounding_score,
        claim_evaluations=[],
        issues=[],
        critical_count=0,
        major_count=major_count,
        minor_count=0,
        tokens_used=100,
        cost_usd=0.0004,
    )


def _build_orchestrator(
    config: ReportWriterConfig,
    valid_report: Report,
    sample_sources: list[SourceReference],
    g1_seq: list[DeterministicVerificationReport],
    g2_seq: list[SemanticGroundingReport],
    verdict_seq: list[ReportJudgeVerdict],
    acceptance_gate: AcceptanceGate | None = None,
    memory_manager: ReflectionMemoryManager | None = None,
) -> tuple[ReflectionLoopOrchestrator, AsyncMock]:
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve_sources = AsyncMock(return_value=sample_sources)

    mock_generator = MagicMock()
    mock_generator.generate_report = AsyncMock(return_value=(valid_report, 300, 0.002))

    mock_det = MagicMock()
    mock_det.verify = MagicMock(side_effect=g1_seq)

    mock_g2 = MagicMock()
    mock_g2.verify_grounding = AsyncMock(side_effect=g2_seq)

    mock_judge = MagicMock()
    mock_judge.evaluate = AsyncMock(side_effect=verdict_seq)

    mock_rev = MagicMock()
    mock_rev.revise_report = AsyncMock(return_value=(valid_report, 250, 0.0015))

    orchestrator = ReflectionLoopOrchestrator(
        llm_service=mock_llm,
        config=config,
        retriever=mock_retriever,
        generator=mock_generator,
        deterministic_verifier=mock_det,
        grounding_verifier=mock_g2,
        judge=mock_judge,
        revision_engine=mock_rev,
        acceptance_gate=acceptance_gate,
        memory_manager=memory_manager,
    )
    return orchestrator, mock_generator.generate_report


@pytest.mark.asyncio
async def test_orchestrator_populates_accepted_gate_result(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Orchestrator run() populates AcceptanceResult and exposes is_accepted=True when passing."""
    config = ReportWriterConfig(max_reflection_rounds=2, acceptance_threshold=0.85)
    orchestrator, _ = _build_orchestrator(
        config=config,
        valid_report=valid_report,
        sample_sources=sample_sources,
        g1_seq=[_make_g1(passed=True)],
        g2_seq=[_make_g2(passed=True, major_count=0, grounding_score=0.95)],
        verdict_seq=[_make_verdict(composite_score=0.92, factual_score=0.93, evidence_score=0.91)],
    )

    result = await orchestrator.run(
        topic=valid_report.topic,
        sources=sample_sources,
        run_id="run-accept-1",
    )

    assert result.acceptance_result is not None
    assert result.acceptance_result.status == ReportStatus.ACCEPTED
    assert result.acceptance_result.is_accepted is True
    assert result.is_accepted is True
    assert result.review_packet is None


@pytest.mark.asyncio
async def test_orchestrator_populates_review_packet_on_unresolved_run(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Orchestrator run() exposes AcceptanceGate HumanReviewPacket and critiques when unresolved."""
    config = ReportWriterConfig(max_reflection_rounds=1, acceptance_threshold=0.85)
    orchestrator, _ = _build_orchestrator(
        config=config,
        valid_report=valid_report,
        sample_sources=sample_sources,
        g1_seq=[_make_g1(passed=True), _make_g1(passed=True)],
        g2_seq=[
            _make_g2(passed=False, major_count=1, grounding_score=0.70),
            _make_g2(passed=False, major_count=1, grounding_score=0.72),
        ],
        verdict_seq=[
            _make_verdict(composite_score=0.75, factual_score=0.75, evidence_score=0.75),
            _make_verdict(composite_score=0.76, factual_score=0.76, evidence_score=0.76),
        ],
    )

    result = await orchestrator.run(
        topic=valid_report.topic,
        sources=sample_sources,
        run_id="run-unresolved-1",
    )

    assert result.acceptance_result is not None
    assert result.acceptance_result.status == ReportStatus.UNRESOLVED
    assert result.is_accepted is False
    assert result.review_packet is not None
    assert result.review_packet is result.acceptance_result.review_packet
    assert result.review_packet.run_id == "run-unresolved-1"
    assert result.review_packet.terminal_reason == result.termination_reason
    assert len(result.review_packet.unresolved_critiques) > 0


def test_reflection_run_result_is_accepted_delegates_to_acceptance_gate(
    valid_report: Report,
) -> None:
    """Direct ReflectionRunResult.is_accepted matches AcceptanceGate rules when acceptance_result is None."""
    budget_summary = BudgetSummary(
        used_tokens=100,
        max_tokens=50000,
        used_cost_usd=0.001,
        max_cost_usd=0.20,
        elapsed_seconds=0.5,
        max_seconds=90.0,
        rounds_executed=0,
        max_rounds=3,
        is_exhausted=False,
    )

    # Fails evidence_grounding minimum (0.80 < 0.85) even though composite is 0.90
    low_evidence_candidate = ReportCandidate(
        round_number=0,
        report=valid_report,
        deterministic_report=_make_g1(passed=True),
        grounding_report=_make_g2(passed=True, major_count=0),
        judge_verdict=_make_verdict(composite_score=0.90, factual_score=0.95, evidence_score=0.80),
        tokens_used=100,
        cost_usd=0.001,
    )
    low_evidence_result = ReflectionRunResult(
        topic=valid_report.topic,
        best_candidate=low_evidence_candidate,
        all_candidates=[low_evidence_candidate],
        budget_summary=budget_summary,
        rounds_executed=0,
        termination_reason="MAX_ROUNDS_REACHED",
        acceptance_threshold=0.85,
    )
    assert low_evidence_result.is_accepted is False

    # Self-judged candidate with 0.87 (< 0.85 + 0.05 = 0.90) is also rejected consistently
    self_judged_candidate = ReportCandidate(
        round_number=0,
        report=valid_report,
        deterministic_report=_make_g1(passed=True),
        grounding_report=_make_g2(passed=True, major_count=0),
        judge_verdict=_make_verdict(
            composite_score=0.87,
            factual_score=0.90,
            evidence_score=0.90,
            is_self_judged=True,
        ),
        tokens_used=100,
        cost_usd=0.001,
    )
    self_judged_result = ReflectionRunResult(
        topic=valid_report.topic,
        best_candidate=self_judged_candidate,
        all_candidates=[self_judged_candidate],
        budget_summary=budget_summary,
        rounds_executed=0,
        termination_reason="MAX_ROUNDS_REACHED",
        acceptance_threshold=0.85,
    )
    assert self_judged_result.is_accepted is False


@pytest.mark.asyncio
async def test_orchestrator_memory_manager_retrieves_and_distills_on_acceptance(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """When configured and guidelines=None, orchestrator retrieves domain lessons and distills on accept."""
    config = ReportWriterConfig(max_reflection_rounds=1, acceptance_threshold=0.85)
    mock_memory = MagicMock(spec=ReflectionMemoryManager)
    mock_memory.retrieve_lessons = AsyncMock(return_value=["Verify every statistic inline."])
    mock_memory.distill_lessons = AsyncMock(return_value=[MagicMock(), MagicMock()])

    orchestrator, generate_spy = _build_orchestrator(
        config=config,
        valid_report=valid_report,
        sample_sources=sample_sources,
        g1_seq=[_make_g1(passed=True)],
        g2_seq=[_make_g2(passed=True, major_count=0)],
        verdict_seq=[_make_verdict(composite_score=0.91)],
        memory_manager=mock_memory,
    )

    result = await orchestrator.run(
        topic=valid_report.topic,
        sources=sample_sources,
        domain="healthcare",
    )

    mock_memory.retrieve_lessons.assert_awaited_once_with(domain="healthcare")
    assert generate_spy.call_args.kwargs["guidelines"] == ["Verify every statistic inline."]
    mock_memory.distill_lessons.assert_awaited_once()
    distill_kwargs = mock_memory.distill_lessons.call_args.kwargs
    assert distill_kwargs["domain"] == "healthcare"
    assert distill_kwargs["run_result"].is_accepted is True
    assert result.learned_lessons_count == 2


@pytest.mark.asyncio
async def test_orchestrator_skips_memory_retrieval_when_guidelines_given_and_skips_distill_on_unresolved(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Explicit guidelines suppress auto-retrieval and unresolved runs skip lesson distillation."""
    config = ReportWriterConfig(max_reflection_rounds=1, acceptance_threshold=0.85)
    mock_memory = MagicMock(spec=ReflectionMemoryManager)
    mock_memory.retrieve_lessons = AsyncMock(return_value=["Should not fetch"])
    mock_memory.distill_lessons = AsyncMock(return_value=[MagicMock()])

    orchestrator, generate_spy = _build_orchestrator(
        config=config,
        valid_report=valid_report,
        sample_sources=sample_sources,
        g1_seq=[_make_g1(passed=True), _make_g1(passed=True)],
        g2_seq=[
            _make_g2(passed=False, major_count=1, grounding_score=0.65),
            _make_g2(passed=False, major_count=1, grounding_score=0.68),
        ],
        verdict_seq=[
            _make_verdict(composite_score=0.70, factual_score=0.70, evidence_score=0.70),
            _make_verdict(composite_score=0.72, factual_score=0.72, evidence_score=0.72),
        ],
        memory_manager=mock_memory,
    )

    result = await orchestrator.run(
        topic=valid_report.topic,
        sources=sample_sources,
        guidelines=["Caller-supplied rule"],
        domain="finance",
    )

    mock_memory.retrieve_lessons.assert_not_awaited()
    assert generate_spy.call_args.kwargs["guidelines"] == ["Caller-supplied rule"]
    mock_memory.distill_lessons.assert_not_awaited()
    assert result.is_accepted is False
    assert result.learned_lessons_count == 0


@pytest.mark.asyncio
async def test_run_and_evaluate_accepts_per_call_gate_and_memory_overrides(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """run_and_evaluate supports temporary AcceptanceGate and memory overrides without mutating self."""
    config = ReportWriterConfig(max_reflection_rounds=1, acceptance_threshold=0.85)
    orchestrator, _ = _build_orchestrator(
        config=config,
        valid_report=valid_report,
        sample_sources=sample_sources,
        g1_seq=[_make_g1(passed=True)],
        g2_seq=[_make_g2(passed=True, major_count=0)],
        verdict_seq=[_make_verdict(composite_score=0.90, factual_score=0.92, evidence_score=0.92)],
    )
    strict_gate = AcceptanceGate(config=ReportWriterConfig(acceptance_threshold=0.95))
    override_memory = MagicMock(spec=ReflectionMemoryManager)
    override_memory.retrieve_lessons = AsyncMock(return_value=[])
    override_memory.distill_lessons = AsyncMock(return_value=[MagicMock()])

    result = await orchestrator.run_and_evaluate(
        topic=valid_report.topic,
        sources=sample_sources,
        domain="policy",
        acceptance_gate=strict_gate,
        memory_manager=override_memory,
    )

    assert orchestrator.acceptance_gate.config.acceptance_threshold == 0.85
    assert orchestrator.memory_manager is None
    assert result.acceptance_threshold == 0.95
    # Candidate composite 0.90 fails the strict 0.95 override gate
    assert result.is_accepted is False
    assert result.review_packet is not None
    override_memory.retrieve_lessons.assert_awaited_once_with(domain="policy")
    override_memory.distill_lessons.assert_not_awaited()
