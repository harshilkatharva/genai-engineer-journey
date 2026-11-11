"""Regression tests verifying major defects prevent premature loop halting (FINDING-05)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from self_improving_report_writer.core.config import ReportWriterConfig
from self_improving_report_writer.models.outcome import ReportStatus
from self_improving_report_writer.models.report import Report, SourceReference
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
from self_improving_report_writer.reflection.orchestrator import ReflectionLoopOrchestrator
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate


def _make_verdict(composite_score: float = 0.90) -> ReportJudgeVerdict:
    return ReportJudgeVerdict(
        composite_score=composite_score,
        passed=True,
        dimension_scores={
            "factual_accuracy": DimensionScore(
                dimension_name="factual_accuracy",
                score=0.92,
                passed=True,
                reasoning="Accurate.",
            ),
            "evidence_grounding": DimensionScore(
                dimension_name="evidence_grounding",
                score=0.90,
                passed=True,
                reasoning="Generally grounded.",
            ),
        },
        overall_feedback="Strong draft apart from cited gap.",
        is_self_judged=False,
        judge_model="gemini-3.1-flash-lite",
        tokens_used=400,
        cost_usd=0.001,
    )


def test_major_defect_forces_revision_even_if_composite_score_passes(valid_report: Report) -> None:
    """A single major defect must set should_revise=True even when composite and grounding scores pass."""
    engine = CritiqueEngine()

    grounding_report = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.90,
        issues=[
            VerificationIssue(
                issue_id="G2-MAJ-1",
                category=DefectCategory.FACTUAL_GROUNDING,
                severity=DefectSeverity.MAJOR,
                message="Claim asserts specific revenue figure not directly stated in chunk.",
                section_id="sec_exec",
                claim_id="claim_exec_1",
            )
        ],
        critical_count=0,
        major_count=1,
        minor_count=0,
    )

    critique = engine.generate_critique(
        round_number=1,
        report=valid_report,
        grounding_report=grounding_report,
        judge_verdict=_make_verdict(composite_score=0.92),
        acceptance_threshold=0.85,
    )

    assert critique.critical_count == 0
    assert critique.major_count == 1
    assert critique.should_revise is True


def test_major_deterministic_defect_forces_revision_even_when_scores_pass(
    valid_report: Report,
) -> None:
    """Major Gate 1 defects must also force revision regardless of high rubric/grounding scores."""
    engine = CritiqueEngine()

    det_report = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
        issues=[
            VerificationIssue(
                issue_id="G1-MAJ-1",
                category=DefectCategory.CITATION_INTEGRITY,
                severity=DefectSeverity.MAJOR,
                message="Factual claim has no citations.",
                section_id="sec_exec",
                claim_id="claim_exec_1",
            )
        ],
        critical_count=0,
        major_count=1,
        minor_count=0,
    )

    critique = engine.generate_critique(
        round_number=0,
        report=valid_report,
        deterministic_report=det_report,
        judge_verdict=_make_verdict(composite_score=0.95),
        acceptance_threshold=0.85,
    )

    assert critique.critical_count == 0
    assert critique.major_count == 1
    assert critique.should_revise is True


def test_subthreshold_grounding_score_without_issues_forces_revision(valid_report: Report) -> None:
    """Sub-threshold grounding (<0.85) must trigger revision even without enumerated major issues."""
    engine = CritiqueEngine()

    grounding_report = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.80,
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
    )

    critique = engine.generate_critique(
        round_number=0,
        report=valid_report,
        grounding_report=grounding_report,
        judge_verdict=_make_verdict(composite_score=0.92),
        acceptance_threshold=0.85,
    )

    assert critique.should_revise is True


async def test_orchestrator_does_not_halt_prematurely_on_major_grounding_defect(
    valid_report: Report,
    sample_sources: list[SourceReference],
) -> None:
    """Orchestrator must revise a baseline having major grounding defects instead of contradicting AcceptanceGate."""
    config = ReportWriterConfig(acceptance_threshold=0.85, max_reflection_rounds=2)

    revised_report = valid_report.model_copy(
        update={
            "version": 2,
            "title": "Revised Grounded Report",
            "created_by": "revision_llm",
        }
    )

    generator = MagicMock()
    generator.generate_report = AsyncMock(return_value=(valid_report, 800, 0.002))

    det_ok = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
    )
    det_verifier = MagicMock()
    det_verifier.verify.return_value = det_ok

    g2_baseline_major = SemanticGroundingReport(
        passed=False,
        all_claims_grounded=False,
        grounding_score=0.90,
        issues=[
            VerificationIssue(
                issue_id="G2-MAJ-1",
                category=DefectCategory.FACTUAL_GROUNDING,
                severity=DefectSeverity.MAJOR,
                message="Claim in section 1 is only weakly supported.",
                section_id="sec_exec",
                claim_id="claim_exec_1",
            )
        ],
        critical_count=0,
        major_count=1,
        minor_count=0,
        tokens_used=300,
        cost_usd=0.001,
    )
    g2_revised_clean = SemanticGroundingReport(
        passed=True,
        all_claims_grounded=True,
        grounding_score=1.0,
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
        tokens_used=300,
        cost_usd=0.001,
    )
    grounding_verifier = MagicMock()
    grounding_verifier.verify_grounding = AsyncMock(
        side_effect=[g2_baseline_major, g2_revised_clean]
    )

    judge = MagicMock()
    judge.evaluate = AsyncMock(
        side_effect=[_make_verdict(composite_score=0.89), _make_verdict(composite_score=0.93)]
    )

    revision_engine = MagicMock()
    revision_engine.revise_report = AsyncMock(return_value=(revised_report, 900, 0.002))

    orchestrator = ReflectionLoopOrchestrator(
        llm_service=AsyncMock(),
        config=config,
        generator=generator,
        deterministic_verifier=det_verifier,
        grounding_verifier=grounding_verifier,
        judge=judge,
        critique_engine=CritiqueEngine(),
        revision_engine=revision_engine,
    )

    result = await orchestrator.run(topic=valid_report.topic, sources=sample_sources)

    assert result.rounds_executed == 1
    assert result.termination_reason == "ALL_ACCEPTANCE_CRITERIA_MET"
    assert result.best_candidate.round_number == 1
    assert result.is_accepted is True
    revision_engine.revise_report.assert_awaited_once()

    gate = AcceptanceGate(config=config)
    baseline_eval = gate.evaluate_candidate(result.all_candidates[0])
    best_eval = gate.evaluate_candidate(result.best_candidate)
    assert baseline_eval.status == ReportStatus.UNRESOLVED
    assert best_eval.status == ReportStatus.ACCEPTED
