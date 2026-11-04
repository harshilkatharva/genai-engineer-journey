"""Unit tests for ReflectionMemoryManager episodic distillation and retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock

from genai_core import EpisodicMemory, InMemoryMemoryStore

from self_improving_report_writer.core.budget import BudgetSummary
from self_improving_report_writer.memory.reflection_memory_manager import (
    ReflectionMemoryManager,
)
from self_improving_report_writer.models.critique import (
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.outcome import ReportCandidate
from self_improving_report_writer.models.report import Report
from self_improving_report_writer.models.rubric import (
    DimensionScore,
    ReportJudgeVerdict,
    SemanticGroundingReport,
)
from self_improving_report_writer.models.verification import (
    DeterministicVerificationReport,
)
from self_improving_report_writer.reflection.orchestrator import ReflectionRunResult


def _make_candidate(round_number: int, report: Report, composite_score: float) -> ReportCandidate:
    """Helper for building candidate with specified score."""
    det_rep = DeterministicVerificationReport(
        passed=True,
        schema_valid=True,
        citations_valid=True,
        passages_matched=True,
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
    )
    g2_rep = SemanticGroundingReport(
        passed=True,
        all_claims_grounded=True,
        grounding_score=composite_score,
        claim_evaluations=[],
        issues=[],
        critical_count=0,
        major_count=0,
        minor_count=0,
    )
    judge_verdict = ReportJudgeVerdict(
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
    )
    return ReportCandidate(
        round_number=round_number,
        report=report,
        deterministic_report=det_rep,
        grounding_report=g2_rep,
        judge_verdict=judge_verdict,
        tokens_used=1000,
        cost_usd=0.002,
    )


def _make_run_result(
    report: Report, baseline_score: float, best_score: float
) -> ReflectionRunResult:
    """Helper for building run result with baseline and best candidate scores."""
    c0 = _make_candidate(round_number=0, report=report, composite_score=baseline_score)
    c1 = _make_candidate(round_number=1, report=report, composite_score=best_score)
    budget_sum = BudgetSummary(
        used_tokens=2000,
        max_tokens=50000,
        used_cost_usd=0.004,
        max_cost_usd=0.20,
        elapsed_seconds=1.5,
        max_seconds=90.0,
        rounds_executed=1,
        max_rounds=3,
        is_exhausted=False,
    )
    return ReflectionRunResult(
        topic=report.topic,
        best_candidate=c1,
        all_candidates=[c0, c1],
        budget_summary=budget_sum,
        rounds_executed=1,
        termination_reason="ALL_ACCEPTANCE_CRITERIA_MET",
    )


async def test_distill_lessons_success_on_verified_improvement(valid_report: Report) -> None:
    """Lessons should be distilled when improvement delta >= 0.10 (e.g. 0.75 -> 0.90 = +0.15)."""
    mem_store = InMemoryMemoryStore()
    episodic_mem = EpisodicMemory(store=mem_store)
    mgr = ReflectionMemoryManager(memory=episodic_mem, min_improvement=0.10, min_confidence=0.85)

    run_result = _make_run_result(valid_report, baseline_score=0.75, best_score=0.90)

    resolved_issue = CritiqueIssue(
        issue_id="ISSUE-01",
        category=IssueCategory.CITATION_INTEGRITY,
        severity=IssueSeverity.CRITICAL,
        explanation="Quotation passage was missing in chunk.",
        suggested_correction="Ensure quoted passage is exact verbatim substring of source chunk.",
        confidence=0.95,
    )

    episodes = await mgr.distill_lessons(
        run_result=run_result,
        domain="quantum_computing",
        resolved_issues=[resolved_issue],
    )

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.action_type == "reflection_writing_lesson"
    assert "In quantum_computing reports:" in ep.result_summary
    assert "exact verbatim substring" in ep.result_summary
    assert ep.parameters["topic_domain"] == "quantum_computing"
    assert ep.metadata["confidence"] == 0.95
    assert ep.metadata["improvement_delta"] == 0.15


async def test_distill_lessons_skips_when_improvement_below_threshold(valid_report: Report) -> None:
    """Lessons should NOT be distilled when score delta < 0.10 (e.g. 0.85 -> 0.88 = +0.03)."""
    mem_store = InMemoryMemoryStore()
    episodic_mem = EpisodicMemory(store=mem_store)
    mgr = ReflectionMemoryManager(memory=episodic_mem, min_improvement=0.10)

    # Delta = 0.03 < 0.10
    run_result = _make_run_result(valid_report, baseline_score=0.85, best_score=0.88)

    resolved_issue = CritiqueIssue(
        issue_id="ISSUE-01",
        category=IssueCategory.FACTUAL_GROUNDING,
        severity=IssueSeverity.MAJOR,
        explanation="Weakly supported claim.",
        suggested_correction="Anchor claims with direct source citations.",
        confidence=0.95,
    )

    episodes = await mgr.distill_lessons(
        run_result=run_result,
        domain="quantum_computing",
        resolved_issues=[resolved_issue],
    )

    assert len(episodes) == 0


async def test_distill_lessons_confidence_filtering(valid_report: Report) -> None:
    """Issues with confidence < 0.85 should be filtered out."""
    mem_store = InMemoryMemoryStore()
    episodic_mem = EpisodicMemory(store=mem_store)
    mgr = ReflectionMemoryManager(memory=episodic_mem, min_improvement=0.10, min_confidence=0.85)

    run_result = _make_run_result(valid_report, baseline_score=0.70, best_score=0.92)

    low_conf_issue = CritiqueIssue(
        issue_id="ISSUE-01",
        category=IssueCategory.CLARITY_ORGANIZATION,
        severity=IssueSeverity.MINOR,
        explanation="Style preference.",
        suggested_correction="Use more active voice.",
        confidence=0.70,  # 0.70 < 0.85
    )
    high_conf_issue = CritiqueIssue(
        issue_id="ISSUE-02",
        category=IssueCategory.FACTUAL_GROUNDING,
        severity=IssueSeverity.MAJOR,
        explanation="Ungrounded factual assertion in prose.",
        suggested_correction="Convert prose factual assertion into structured claim.",
        confidence=0.92,  # 0.92 >= 0.85
    )

    episodes = await mgr.distill_lessons(
        run_result=run_result,
        domain="quantum_computing",
        resolved_issues=[low_conf_issue, high_conf_issue],
    )

    assert len(episodes) == 1
    assert "Convert prose factual assertion" in episodes[0].result_summary


async def test_distill_lessons_idempotent_deduplication(valid_report: Report) -> None:
    """Duplicate lessons with identical canonical parameters must be skipped via check()."""
    mem_store = InMemoryMemoryStore()
    episodic_mem = EpisodicMemory(store=mem_store)
    mgr = ReflectionMemoryManager(memory=episodic_mem, min_improvement=0.10)

    run_result = _make_run_result(valid_report, baseline_score=0.72, best_score=0.91)

    issue = CritiqueIssue(
        issue_id="ISSUE-01",
        category=IssueCategory.CITATION_INTEGRITY,
        severity=IssueSeverity.CRITICAL,
        explanation="Quotation mismatch",
        suggested_correction="Verify verbatim quote in source",
        confidence=0.95,
    )

    # First run records episode
    first_episodes = await mgr.distill_lessons(
        run_result=run_result, domain="quantum", resolved_issues=[issue]
    )
    assert len(first_episodes) == 1

    # Second run with identical issue must skip due to check() status == 'do_not_repeat'
    second_episodes = await mgr.distill_lessons(
        run_result=run_result, domain="quantum", resolved_issues=[issue]
    )
    assert len(second_episodes) == 0


async def test_retrieve_lessons_filtering(valid_report: Report) -> None:
    """retrieve_lessons should filter guidelines by domain."""
    mem_store = InMemoryMemoryStore()
    episodic_mem = EpisodicMemory(store=mem_store)
    mgr = ReflectionMemoryManager(memory=episodic_mem)

    # Pre-record episodes in two different domains
    await episodic_mem.record(
        user_id="default_user",
        action_type=ReflectionMemoryManager.ACTION_TYPE,
        parameters={"topic_domain": "quantum_computing"},
        target="quantum_computing",
        result_summary="In quantum_computing reports: always cite NISQ qubit counts accurately.",
    )
    await episodic_mem.record(
        user_id="default_user",
        action_type=ReflectionMemoryManager.ACTION_TYPE,
        parameters={"topic_domain": "medicine"},
        target="medicine",
        result_summary="In medicine reports: distinguish in-vitro results from clinical trials.",
    )

    # Query for quantum domain
    quantum_lessons = await mgr.retrieve_lessons(domain="quantum_computing")
    assert len(quantum_lessons) == 1
    assert "always cite NISQ qubit counts accurately" in quantum_lessons[0]

    # Query for medicine domain
    med_lessons = await mgr.retrieve_lessons(domain="medicine")
    assert len(med_lessons) == 1
    assert "distinguish in-vitro results from clinical trials" in med_lessons[0]


async def test_soft_failure_handling(valid_report: Report) -> None:
    """Exceptions from episodic memory should be caught and logged without raising."""
    faulty_mem = AsyncMock(spec=EpisodicMemory)
    faulty_mem.check.side_effect = RuntimeError("Database connection dropped")

    mgr = ReflectionMemoryManager(memory=faulty_mem)
    run_result = _make_run_result(valid_report, baseline_score=0.70, best_score=0.90)

    # Must return empty list instead of raising RuntimeError
    episodes = await mgr.distill_lessons(
        run_result=run_result,
        domain="quantum",
        resolved_issues=[
            CritiqueIssue(
                issue_id="I-1",
                category=IssueCategory.COMPLETENESS,
                severity=IssueSeverity.MAJOR,
                explanation="Caveat missing",
                suggested_correction="Add caveat",
                confidence=0.95,
            )
        ],
    )
    assert episodes == []
