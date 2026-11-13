"""Regression tests for orchestrator termination reason classification (FINDING-06)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
from self_improving_report_writer.models.verification import DeterministicVerificationReport
from self_improving_report_writer.reflection.orchestrator import ReflectionLoopOrchestrator


def _make_verdict(composite_score: float = 0.78) -> ReportJudgeVerdict:
    return ReportJudgeVerdict(
        composite_score=composite_score,
        passed=composite_score >= 0.85,
        dimension_scores={
            "factual_accuracy": DimensionScore(
                dimension_name="factual_accuracy",
                score=0.90,
                passed=True,
                reasoning="Factual accuracy ok",
            ),
            "evidence_grounding": DimensionScore(
                dimension_name="evidence_grounding",
                score=0.90,
                passed=True,
                reasoning="Grounding ok",
            ),
        },
        overall_feedback="Needs minor expansion.",
        is_self_judged=False,
        judge_model="gemini-3.1-flash-lite",
        tokens_used=200,
        cost_usd=0.001,
    )


def _make_distinct_critiques(num_rounds: int) -> list[Critique]:
    return [
        Critique(
            round_number=idx,
            issues=[
                CritiqueIssue(
                    issue_id=f"ISSUE-{idx}",
                    category=IssueCategory.COMPLETENESS,
                    severity=IssueSeverity.MAJOR,
                    affected_claim_id=f"claim_{idx}",
                    explanation=f"Round {idx} completeness gap",
                    suggested_correction=f"Expand section for round {idx}",
                )
            ],
            summary=f"Round {idx} requires revision",
            should_revise=True,
        )
        for idx in range(num_rounds + 1)
    ]


def _make_distinct_reports(base: Report, count: int) -> list[Report]:
    reports = []
    for idx in range(count):
        rep = base.model_copy(deep=True)
        rep.version = idx + 2
        rep.title = f"{base.title} Revision {idx + 1}"
        reports.append(rep)
    return reports


def _build_mock_orchestrator(
    config: ReportWriterConfig,
    valid_report: Report,
    *,
    num_rounds: int = 3,
    revision_tokens: int = 200,
    revision_cost: float = 0.001,
) -> ReflectionLoopOrchestrator:
    generator = MagicMock()
    generator.generate_report = AsyncMock(return_value=(valid_report, 300, 0.001))

    det_verifier = MagicMock()
    det_verifier.verify.return_value = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
    )

    grounding_verifier = MagicMock()
    grounding_verifier.verify_grounding = AsyncMock(
        return_value=SemanticGroundingReport(
            passed=True,
            all_claims_grounded=True,
            grounding_score=0.95,
            tokens_used=100,
            cost_usd=0.0005,
        )
    )

    judge = MagicMock()
    judge.evaluate = AsyncMock(
        side_effect=[
            _make_verdict(composite_score=round(0.70 + 0.03 * idx, 2))
            for idx in range(num_rounds + 1)
        ]
    )

    critique_engine = MagicMock()
    critique_engine.generate_critique.side_effect = _make_distinct_critiques(num_rounds)

    revision_engine = MagicMock()
    revision_engine.revise_report = AsyncMock(
        side_effect=[
            (rep, revision_tokens, revision_cost)
            for rep in _make_distinct_reports(valid_report, num_rounds)
        ]
    )

    return ReflectionLoopOrchestrator(
        llm_service=AsyncMock(),
        config=config,
        generator=generator,
        deterministic_verifier=det_verifier,
        grounding_verifier=grounding_verifier,
        judge=judge,
        critique_engine=critique_engine,
        revision_engine=revision_engine,
    )


async def test_normal_max_rounds_completion_is_not_mislabeled_as_budget_exceeded(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Completing max_reflection_rounds normally must report MAX_ROUNDS_REACHED, not BUDGET_EXCEEDED."""
    cfg = ReportWriterConfig(
        max_reflection_rounds=3,
        max_total_tokens=50000,
        max_total_cost_usd=0.50,
        max_wall_clock_seconds=90.0,
    )
    orchestrator = _build_mock_orchestrator(cfg, valid_report, num_rounds=3)

    result = await orchestrator.run(topic=valid_report.topic, sources=sample_sources)

    assert result.rounds_executed == 3
    assert result.termination_reason == "MAX_ROUNDS_REACHED"
    assert not result.termination_reason.startswith("BUDGET_EXCEEDED")


async def test_token_exhaustion_on_final_round_is_labeled_budget_exceeded(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """True resource exhaustion (tokens) must still report BUDGET_EXCEEDED even on the final round."""
    # Baseline: 300+100+200 = 600 tokens. Each round: 500+100+200 = 800 tokens.
    # After round 2 (max_rounds=2): 600 + 800 + 800 = 2200 >= 2200 -> TOKENS_EXHAUSTED.
    cfg = ReportWriterConfig(
        max_reflection_rounds=2,
        max_total_tokens=2200,
        max_total_cost_usd=0.50,
        max_wall_clock_seconds=90.0,
    )
    orchestrator = _build_mock_orchestrator(cfg, valid_report, num_rounds=2, revision_tokens=500)

    result = await orchestrator.run(topic=valid_report.topic, sources=sample_sources)

    assert result.rounds_executed == 2
    assert result.termination_reason.startswith("BUDGET_EXCEEDED: TOKENS_EXHAUSTED")


async def test_cost_exhaustion_on_round_is_labeled_budget_exceeded(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """True cost exhaustion must still report BUDGET_EXCEEDED: COST_EXHAUSTED."""
    cfg = ReportWriterConfig(
        max_reflection_rounds=3,
        max_total_tokens=50000,
        max_total_cost_usd=0.01,
        max_wall_clock_seconds=90.0,
    )
    orchestrator = _build_mock_orchestrator(cfg, valid_report, num_rounds=3, revision_cost=0.01)

    result = await orchestrator.run(topic=valid_report.topic, sources=sample_sources)

    assert result.rounds_executed == 1
    assert result.termination_reason.startswith("BUDGET_EXCEEDED: COST_EXHAUSTED")
