"""Regression tests ensuring unresolved defects are never distilled into episodic memory (FINDING-09)."""

from __future__ import annotations

from genai_core import EpisodicMemory, InMemoryMemoryStore

from self_improving_report_writer.core.budget import BudgetSummary
from self_improving_report_writer.memory.reflection_memory_manager import ReflectionMemoryManager
from self_improving_report_writer.models.outcome import ReportCandidate
from self_improving_report_writer.models.report import Report
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
from self_improving_report_writer.reflection.orchestrator import ReflectionRunResult


def _build_candidate(
    round_number: int,
    report: Report,
    *,
    composite_score: float,
    det_issues: list[VerificationIssue],
) -> ReportCandidate:
    return ReportCandidate(
        round_number=round_number,
        report=report,
        deterministic_report=DeterministicVerificationReport(
            passed=(len(det_issues) == 0),
            schema_valid=True,
            citations_valid=(len(det_issues) == 0),
            passages_matched=(len(det_issues) == 0),
            issues=det_issues,
            critical_count=sum(1 for i in det_issues if i.severity == DefectSeverity.CRITICAL),
            major_count=sum(1 for i in det_issues if i.severity == DefectSeverity.MAJOR),
            minor_count=sum(1 for i in det_issues if i.severity == DefectSeverity.MINOR),
        ),
        grounding_report=SemanticGroundingReport(
            passed=True,
            all_claims_grounded=True,
            grounding_score=composite_score,
        ),
        judge_verdict=ReportJudgeVerdict(
            composite_score=composite_score,
            passed=composite_score >= 0.85,
            dimension_scores={
                "factual_accuracy": DimensionScore(
                    dimension_name="factual_accuracy",
                    score=composite_score,
                    passed=True,
                    reasoning="Rationale",
                ),
                "evidence_grounding": DimensionScore(
                    dimension_name="evidence_grounding",
                    score=composite_score,
                    passed=True,
                    reasoning="Rationale",
                ),
            },
            overall_feedback="Feedback",
            judge_model="gemini-3.1-flash-lite",
        ),
    )


def _build_run_result(
    report: Report,
    *,
    baseline_issues: list[VerificationIssue],
    best_issues: list[VerificationIssue],
    baseline_score: float = 0.70,
    best_score: float = 0.88,
) -> ReflectionRunResult:
    c0 = _build_candidate(0, report, composite_score=baseline_score, det_issues=baseline_issues)
    c1 = _build_candidate(1, report, composite_score=best_score, det_issues=best_issues)
    return ReflectionRunResult(
        topic=report.topic,
        best_candidate=c1,
        all_candidates=[c0, c1],
        budget_summary=BudgetSummary(
            used_tokens=1500,
            max_tokens=50000,
            used_cost_usd=0.002,
            max_cost_usd=0.20,
            elapsed_seconds=1.0,
            max_seconds=90.0,
            rounds_executed=1,
            max_rounds=3,
            is_exhausted=False,
        ),
        rounds_executed=1,
        termination_reason="MAX_ROUNDS_REACHED",
    )


async def test_unresolved_defects_are_not_distilled_into_memory(valid_report: Report) -> None:
    """When resolved_issues is omitted, unresolved baseline defects persisting in best must not be distilled."""
    mgr = ReflectionMemoryManager(
        memory=EpisodicMemory(store=InMemoryMemoryStore()),
        min_improvement=0.10,
    )

    unresolved_issue = VerificationIssue(
        issue_id="CIT-001",
        category=DefectCategory.CITATION_INTEGRITY,
        severity=DefectSeverity.CRITICAL,
        message="Citation 'xyz' does not exist in source pool.",
        section_id="sec_1",
        claim_id="claim_1",
        citation_id="cite_xyz",
    )
    run_result = _build_run_result(
        valid_report,
        baseline_issues=[unresolved_issue],
        best_issues=[unresolved_issue],
    )

    episodes = await mgr.distill_lessons(run_result=run_result, domain="ai_systems")

    assert episodes == []


async def test_only_resolved_baseline_defects_are_distilled_across_renumbering(
    valid_report: Report,
) -> None:
    """When DeterministicVerifier renumbers remaining defects as DEFECT-001, only fixed defects distill."""
    mgr = ReflectionMemoryManager(
        memory=EpisodicMemory(store=InMemoryMemoryStore()),
        min_improvement=0.10,
    )

    fixed_in_round1 = VerificationIssue(
        issue_id="DEFECT-001",
        category=DefectCategory.FACTUAL_GROUNDING,
        severity=DefectSeverity.CRITICAL,
        message="Quotation in citation 'cite_1' does not exist in referenced chunk 'chk_1'.",
        section_id="sec_exec",
        claim_id="claim_exec_1",
        citation_id="cite_1",
    )
    still_broken_baseline = VerificationIssue(
        issue_id="DEFECT-002",
        category=DefectCategory.CITATION_INTEGRITY,
        severity=DefectSeverity.CRITICAL,
        message="Citation 'cite_2' references unknown source/chunk pair ('fake', 'chk_bad').",
        section_id="sec_1",
        claim_id="claim_1",
        citation_id="cite_2",
    )
    # Gate 1 resets issue_counter=1 on the revised report, so remaining issue becomes DEFECT-001
    still_broken_renumbered = VerificationIssue(
        issue_id="DEFECT-001",
        category=DefectCategory.CITATION_INTEGRITY,
        severity=DefectSeverity.CRITICAL,
        message="Citation 'cite_2' references unknown source/chunk pair ('fake', 'chk_bad').",
        section_id="sec_1",
        claim_id="claim_1",
        citation_id="cite_2",
    )

    run_result = _build_run_result(
        valid_report,
        baseline_issues=[fixed_in_round1, still_broken_baseline],
        best_issues=[still_broken_renumbered],
    )

    episodes = await mgr.distill_lessons(run_result=run_result, domain="quantum_computing")

    assert len(episodes) == 1
    assert "cite_1" in episodes[0].result_summary
    assert "chk_bad" not in episodes[0].result_summary
