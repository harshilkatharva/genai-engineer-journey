"""Unit tests for ReflectionLoopOrchestrator and ReflectionRunResult."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
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
from self_improving_report_writer.reflection.orchestrator import (
    ReflectionLoopOrchestrator,
)


def _make_mock_verdict(
    composite_score: float = 0.90,
    passed: bool = True,
    factual_score: float = 0.90,
    evidence_score: float = 0.90,
    is_self_judged: bool = False,
) -> ReportJudgeVerdict:
    """Build mock ReportJudgeVerdict."""
    return ReportJudgeVerdict(
        composite_score=composite_score,
        passed=passed,
        dimension_scores={
            "factual_accuracy": DimensionScore(
                dimension_name="factual_accuracy",
                score=factual_score,
                passed=factual_score >= 0.85,
                reasoning="Factual accuracy reasoning",
            ),
            "evidence_grounding": DimensionScore(
                dimension_name="evidence_grounding",
                score=evidence_score,
                passed=evidence_score >= 0.85,
                reasoning="Evidence grounding reasoning",
            ),
        },
        overall_feedback="Good report.",
        is_self_judged=is_self_judged,
        judge_model="gemini-2.5-pro",
        tokens_used=500,
        cost_usd=0.002,
    )


def _make_mock_g1(
    passed: bool = True, critical: int = 0, major: int = 0
) -> DeterministicVerificationReport:
    """Build mock DeterministicVerificationReport."""
    return DeterministicVerificationReport(
        passed=passed,
        schema_valid=passed,
        citations_valid=passed,
        passages_matched=passed,
        issues=[],
        critical_count=critical,
        major_count=major,
        minor_count=0,
    )


def _make_mock_g2(
    passed: bool = True,
    grounding_score: float = 0.95,
    critical: int = 0,
    major: int = 0,
) -> SemanticGroundingReport:
    """Build mock SemanticGroundingReport."""
    return SemanticGroundingReport(
        passed=passed,
        all_claims_grounded=(critical == 0 and major == 0),
        grounding_score=grounding_score,
        claim_evaluations=[],
        issues=[],
        critical_count=critical,
        major_count=major,
        minor_count=0,
        tokens_used=400,
        cost_usd=0.001,
    )


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    """Mock LLMService."""
    return AsyncMock()


async def test_orchestrator_accepts_baseline_immediately(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """If baseline report satisfies all criteria, orchestrator halts at round 0 without revisions."""
    generator = MagicMock()
    generator.generate_report = AsyncMock(return_value=(valid_report, 1000, 0.004))

    det_verifier = MagicMock()
    det_verifier.verify.return_value = _make_mock_g1(passed=True)

    grounding_verifier = MagicMock()
    grounding_verifier.verify_grounding = AsyncMock(return_value=_make_mock_g2(passed=True))

    judge = MagicMock()
    judge.evaluate = AsyncMock(return_value=_make_mock_verdict(composite_score=0.92, passed=True))

    critique_engine = MagicMock()
    critique_engine.generate_critique.return_value = Critique(
        round_number=0,
        issues=[],
        summary="All standards met.",
        should_revise=False,
    )

    revision_engine = MagicMock()
    revision_engine.revise_report = AsyncMock()

    orchestrator = ReflectionLoopOrchestrator(
        llm_service=mock_llm_service,
        generator=generator,
        deterministic_verifier=det_verifier,
        grounding_verifier=grounding_verifier,
        judge=judge,
        critique_engine=critique_engine,
        revision_engine=revision_engine,
    )

    result = await orchestrator.run(topic="Quantum Optimization", sources=sample_sources)

    assert result.rounds_executed == 0
    assert result.termination_reason == "ALL_ACCEPTANCE_CRITERIA_MET"
    assert result.best_candidate.round_number == 0
    assert result.is_accepted
    assert len(result.all_candidates) == 1
    revision_engine.revise_report.assert_not_awaited()


async def test_orchestrator_revises_and_accepts(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Orchestrator should revise baseline when critique indicates revision needed, then accept."""
    generator = MagicMock()
    generator.generate_report = AsyncMock(return_value=(valid_report, 1000, 0.004))

    det_verifier = MagicMock()
    det_verifier.verify.side_effect = [
        _make_mock_g1(passed=True, major=1),  # round 0: 1 major defect
        _make_mock_g1(passed=True, major=0),  # round 1: 0 defects
    ]

    grounding_verifier = MagicMock()
    grounding_verifier.verify_grounding = AsyncMock(
        side_effect=[
            _make_mock_g2(passed=False, grounding_score=0.75, major=1),
            _make_mock_g2(passed=True, grounding_score=0.95, major=0),
        ]
    )

    judge = MagicMock()
    judge.evaluate = AsyncMock(
        side_effect=[
            _make_mock_verdict(composite_score=0.78, passed=False),
            _make_mock_verdict(composite_score=0.91, passed=True),
        ]
    )

    critique_engine = MagicMock()
    critique_engine.generate_critique.side_effect = [
        Critique(
            round_number=0,
            issues=[
                CritiqueIssue(
                    issue_id="C-1",
                    category=IssueCategory.FACTUAL_GROUNDING,
                    severity=IssueSeverity.MAJOR,
                    explanation="Defect",
                    suggested_correction="Fix",
                )
            ],
            summary="Revise needed",
            should_revise=True,
        ),
        Critique(
            round_number=1,
            issues=[],
            summary="All issues resolved",
            should_revise=False,
        ),
    ]

    revised_rep = valid_report.model_copy(deep=True)
    revised_rep.version = 2
    revision_engine = MagicMock()
    revision_engine.revise_report = AsyncMock(return_value=(revised_rep, 1200, 0.005))

    orchestrator = ReflectionLoopOrchestrator(
        llm_service=mock_llm_service,
        generator=generator,
        deterministic_verifier=det_verifier,
        grounding_verifier=grounding_verifier,
        judge=judge,
        critique_engine=critique_engine,
        revision_engine=revision_engine,
    )

    result = await orchestrator.run(topic="Quantum Optimization", sources=sample_sources)

    assert result.rounds_executed == 1
    assert result.termination_reason == "ALL_ACCEPTANCE_CRITERIA_MET"
    assert result.best_candidate.round_number == 1
    assert result.best_candidate.judge_verdict.composite_score == 0.91
    assert result.is_accepted
    assert len(result.all_candidates) == 2


async def test_orchestrator_halts_on_budget_exhaustion(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Orchestrator should halt when token budget is exhausted before next revision."""
    cfg = ReportWriterConfig(max_total_tokens=2000, max_reflection_rounds=3)

    generator = MagicMock()
    generator.generate_report = AsyncMock(return_value=(valid_report, 1500, 0.004))

    det_verifier = MagicMock()
    det_verifier.verify.return_value = _make_mock_g1(passed=True, critical=1)

    grounding_verifier = MagicMock()
    # Consumes 600 tokens -> total = 2100 >= 2000
    grounding_verifier.verify_grounding = AsyncMock(
        return_value=SemanticGroundingReport(
            passed=False,
            all_claims_grounded=False,
            grounding_score=0.60,
            claim_evaluations=[],
            issues=[],
            critical_count=1,
            major_count=0,
            minor_count=0,
            tokens_used=600,
            cost_usd=0.001,
        )
    )

    judge = MagicMock()
    judge.evaluate = AsyncMock(return_value=_make_mock_verdict(composite_score=0.70, passed=False))

    critique_engine = MagicMock()
    critique_engine.generate_critique.return_value = Critique(
        round_number=0,
        issues=[
            CritiqueIssue(
                issue_id="C-1",
                category=IssueCategory.CITATION_INTEGRITY,
                severity=IssueSeverity.CRITICAL,
                explanation="Quotation mismatch",
                suggested_correction="Fix quotation",
            )
        ],
        summary="Revision needed",
        should_revise=True,
    )

    revision_engine = MagicMock()
    revision_engine.revise_report = AsyncMock()

    orchestrator = ReflectionLoopOrchestrator(
        llm_service=mock_llm_service,
        config=cfg,
        generator=generator,
        deterministic_verifier=det_verifier,
        grounding_verifier=grounding_verifier,
        judge=judge,
        critique_engine=critique_engine,
        revision_engine=revision_engine,
    )

    result = await orchestrator.run(topic="Quantum Optimization", sources=sample_sources)

    assert "BUDGET_EXCEEDED" in result.termination_reason
    assert result.rounds_executed == 0
    revision_engine.revise_report.assert_not_awaited()


async def test_orchestrator_halts_on_consecutive_regressions(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Orchestrator should halt when candidate quality degrades and exceeds max_consecutive_regressions."""
    cfg = ReportWriterConfig(max_consecutive_regressions=1, max_reflection_rounds=3)

    generator = MagicMock()
    generator.generate_report = AsyncMock(return_value=(valid_report, 500, 0.001))

    det_verifier = MagicMock()
    det_verifier.verify.side_effect = [
        _make_mock_g1(passed=True, critical=0),  # Round 0 passed Gate 1
        _make_mock_g1(passed=False, critical=2),  # Round 1 regressed: failed Gate 1
    ]

    grounding_verifier = MagicMock()
    grounding_verifier.verify_grounding = AsyncMock(return_value=_make_mock_g2(passed=True))

    judge = MagicMock()
    judge.evaluate = AsyncMock(
        side_effect=[
            _make_mock_verdict(composite_score=0.80),
            _make_mock_verdict(composite_score=0.60),
        ]
    )

    critique_engine = MagicMock()
    critique_engine.generate_critique.return_value = Critique(
        round_number=0,
        issues=[
            CritiqueIssue(
                issue_id="C-1",
                category=IssueCategory.COMPLETENESS,
                severity=IssueSeverity.MAJOR,
                explanation="Missing details",
                suggested_correction="Add details",
            )
        ],
        summary="Revision needed",
        should_revise=True,
    )

    revised_rep = valid_report.model_copy(deep=True)
    revised_rep.version = 2
    revision_engine = MagicMock()
    revision_engine.revise_report = AsyncMock(return_value=(revised_rep, 500, 0.001))

    orchestrator = ReflectionLoopOrchestrator(
        llm_service=mock_llm_service,
        config=cfg,
        generator=generator,
        deterministic_verifier=det_verifier,
        grounding_verifier=grounding_verifier,
        judge=judge,
        critique_engine=critique_engine,
        revision_engine=revision_engine,
    )

    result = await orchestrator.run(topic="Quantum Optimization", sources=sample_sources)

    assert "CONSECUTIVE_REGRESSIONS_EXCEEDED" in result.termination_reason
    # Best-so-far should preserve Round 0 rather than the degraded Round 1
    assert result.best_candidate.round_number == 0
    assert result.best_candidate.deterministic_report.passed is True


async def test_orchestrator_halts_on_oscillation(
    mock_llm_service: AsyncMock,
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Orchestrator should halt when critique stagnation or content cycle is detected."""
    generator = MagicMock()
    generator.generate_report = AsyncMock(return_value=(valid_report, 500, 0.001))

    det_verifier = MagicMock()
    det_verifier.verify.return_value = _make_mock_g1(passed=True)

    grounding_verifier = MagicMock()
    grounding_verifier.verify_grounding = AsyncMock(return_value=_make_mock_g2(passed=True))

    judge = MagicMock()
    judge.evaluate = AsyncMock(return_value=_make_mock_verdict(composite_score=0.80))

    # Identical critique issue set across both rounds -> critique stagnation
    same_critique = Critique(
        round_number=0,
        issues=[
            CritiqueIssue(
                issue_id="C-1",
                category=IssueCategory.COMPLETENESS,
                severity=IssueSeverity.MAJOR,
                explanation="Missing caveats",
                suggested_correction="Add caveats",
            )
        ],
        summary="Revise needed",
        should_revise=True,
    )
    critique_engine = MagicMock()
    critique_engine.generate_critique.return_value = same_critique

    # Revision returns modified title/content to avoid content cycle, but stagnation triggers
    revised_rep = valid_report.model_copy(deep=True)
    revised_rep.title = "Slightly Different Title"
    revised_rep.version = 2
    revision_engine = MagicMock()
    revision_engine.revise_report = AsyncMock(return_value=(revised_rep, 500, 0.001))

    orchestrator = ReflectionLoopOrchestrator(
        llm_service=mock_llm_service,
        generator=generator,
        deterministic_verifier=det_verifier,
        grounding_verifier=grounding_verifier,
        judge=judge,
        critique_engine=critique_engine,
        revision_engine=revision_engine,
    )

    result = await orchestrator.run(topic="Quantum Optimization", sources=sample_sources)

    assert "OSCILLATION_DETECTED" in result.termination_reason
    assert "critique_stagnation" in result.termination_reason
