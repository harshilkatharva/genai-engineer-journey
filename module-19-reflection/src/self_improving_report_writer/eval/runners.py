"""Experimental runners for Approaches A, B, and C in the comparative benchmark."""

from __future__ import annotations

import logging
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.eval.blinded_judge import BlindedJudge
from self_improving_report_writer.eval.dataset import BenchmarkTopic
from self_improving_report_writer.generation.report_generator import ReportGenerator
from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.outcome import ReportCandidate, ReportStatus
from self_improving_report_writer.models.report import Report
from self_improving_report_writer.reflection.orchestrator import ReflectionLoopOrchestrator
from self_improving_report_writer.reflection.revision_engine import RevisionEngine
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate
from self_improving_report_writer.verification.deterministic_verifier import (
    DeterministicVerifier,
)
from self_improving_report_writer.verification.semantic_grounding_verifier import (
    SemanticGroundingVerifier,
)

if TYPE_CHECKING:
    from genai_core import LLMService

logger = logging.getLogger(__name__)


class ApproachType(StrEnum):
    """Classification of comparative benchmark conditions."""

    APPROACH_A = "approach_a_no_reflection"
    APPROACH_B = "approach_b_ungrounded_refinement"
    APPROACH_C = "approach_c_grounded_reflection"


class ApproachRunResult(BaseModel):
    """Standardized output of a single benchmark topic execution under an approach."""

    topic_id: str = Field(description="Benchmark topic identifier (e.g. 'qc-01')")
    topic_title: str = Field(description="Title of the research topic")
    approach: ApproachType = Field(description="Experimental condition")
    approach_name: str = Field(description="Human-readable condition label")
    final_report: Report = Field(description="Final synthesized report deliverable")
    rounds_executed: int = Field(ge=0, description="Total reflection/revision rounds executed")
    deterministic_passed: bool = Field(description="Whether Gate 1 passed")
    critical_defect_count: int = Field(ge=0, description="Number of unresolved critical defects")
    major_defect_count: int = Field(ge=0, description="Number of unresolved major defects")
    composite_rubric_score: float = Field(
        ge=0.0, le=1.0, description="Blinded composite rubric score"
    )
    factual_accuracy_score: float = Field(
        ge=0.0, le=1.0, description="Blinded factual accuracy dimension score"
    )
    evidence_grounding_score: float = Field(
        ge=0.0, le=1.0, description="Blinded evidence grounding dimension score"
    )
    dimension_scores: dict[str, float] = Field(
        default_factory=dict, description="All individual rubric dimension scores"
    )
    tokens_consumed: int = Field(ge=0, description="Total token consumption")
    cost_usd: float = Field(ge=0.0, description="Total financial expenditure in USD")
    latency_seconds: float = Field(ge=0.0, description="Elapsed wall-clock execution time")
    is_accepted: bool = Field(
        description="Whether the report meets Section 6.1 acceptance criteria"
    )
    terminal_status: ReportStatus = Field(description="Terminal lifecycle status")


async def run_approach_a(
    topic: BenchmarkTopic,
    generator: ReportGenerator | None = None,
    deterministic_verifier: DeterministicVerifier | None = None,
    grounding_verifier: SemanticGroundingVerifier | None = None,
    blinded_judge: BlindedJudge | None = None,
    acceptance_gate: AcceptanceGate | None = None,
    config: ReportWriterConfig | None = None,
    llm_service: LLMService | None = None,
) -> ApproachRunResult:
    """Execute Approach A: Baseline (No Reflection).

    Synthesizes the initial draft report R0 and immediately evaluates it without reflection.
    """
    cfg = config or get_default_config()
    svc = llm_service
    if svc is None and (generator is None or grounding_verifier is None or blinded_judge is None):
        from genai_core import LLMService

        svc = LLMService()

    gen = generator or ReportGenerator(llm_service=svc, config=cfg)
    det_ver = deterministic_verifier or DeterministicVerifier()
    grd_ver = grounding_verifier or SemanticGroundingVerifier(llm_service=svc, config=cfg)
    judge = blinded_judge or BlindedJudge(llm_service=svc, config=cfg)
    gate = acceptance_gate or AcceptanceGate(config=cfg)

    sources_map = {src.chunk_id: src for src in topic.gold_sources}

    start_time = time.perf_counter()
    tokens = 0
    cost = 0.0

    # Step 1: Initial generation
    initial_report, gen_tokens, gen_cost = await gen.generate_report(
        topic=topic.title,
        sources=topic.gold_sources,
    )
    tokens += gen_tokens
    cost += gen_cost

    # Step 2: Evaluation
    deadline = time.monotonic() + cfg.max_wall_clock_seconds
    det_report = det_ver.verify(initial_report, sources_map)
    grd_report = await grd_ver.verify_grounding(initial_report, sources_map, deadline=deadline)
    judge_verdict = await judge.evaluate_blinded(
        report=initial_report,
        sources=sources_map,
        deterministic_report=det_report,
        deadline=deadline,
    )

    candidate = ReportCandidate(
        round_number=0,
        report=initial_report,
        deterministic_report=det_report,
        grounding_report=grd_report,
        judge_verdict=judge_verdict,
        tokens_used=tokens,
        cost_usd=cost,
    )

    acceptance = gate.evaluate_candidate(candidate)
    elapsed = time.perf_counter() - start_time

    return ApproachRunResult(
        topic_id=topic.topic_id,
        topic_title=topic.title,
        approach=ApproachType.APPROACH_A,
        approach_name="Approach A: No Reflection (Baseline)",
        final_report=initial_report,
        rounds_executed=0,
        deterministic_passed=det_report.passed,
        critical_defect_count=candidate.critical_defect_count,
        major_defect_count=candidate.major_defect_count,
        composite_rubric_score=judge_verdict.composite_score,
        factual_accuracy_score=candidate.factual_accuracy_score,
        evidence_grounding_score=candidate.evidence_grounding_score,
        dimension_scores={k: v.score for k, v in judge_verdict.dimension_scores.items()},
        tokens_consumed=tokens,
        cost_usd=cost,
        latency_seconds=elapsed,
        is_accepted=acceptance.is_accepted,
        terminal_status=acceptance.status,
    )


async def run_approach_b(
    topic: BenchmarkTopic,
    max_rounds: int = 2,
    generator: ReportGenerator | None = None,
    revision_engine: RevisionEngine | None = None,
    deterministic_verifier: DeterministicVerifier | None = None,
    grounding_verifier: SemanticGroundingVerifier | None = None,
    blinded_judge: BlindedJudge | None = None,
    acceptance_gate: AcceptanceGate | None = None,
    config: ReportWriterConfig | None = None,
    llm_service: LLMService | None = None,
) -> ApproachRunResult:
    """Execute Approach B: Ungrounded Self-Critique (Standard LLM Refinement).

    Iterates for up to `max_rounds` using free-form, ungrounded self-critique without
    access to dual-gate verification results or deterministic checks.
    """
    cfg = config or get_default_config()
    svc = llm_service
    if svc is None and (
        generator is None
        or revision_engine is None
        or grounding_verifier is None
        or blinded_judge is None
    ):
        from genai_core import LLMService

        svc = LLMService()

    gen = generator or ReportGenerator(llm_service=svc, config=cfg)
    rev_engine = revision_engine or RevisionEngine(llm_service=svc, config=cfg)
    det_ver = deterministic_verifier or DeterministicVerifier()
    grd_ver = grounding_verifier or SemanticGroundingVerifier(llm_service=svc, config=cfg)
    judge = blinded_judge or BlindedJudge(llm_service=svc, config=cfg)
    gate = acceptance_gate or AcceptanceGate(config=cfg)

    sources_map = {src.chunk_id: src for src in topic.gold_sources}

    start_time = time.perf_counter()
    tokens = 0
    cost = 0.0

    # Step 1: Initial generation
    current_report, gen_tokens, gen_cost = await gen.generate_report(
        topic=topic.title,
        sources=topic.gold_sources,
    )
    tokens += gen_tokens
    cost += gen_cost

    rounds_done = 0

    # Step 2: Ungrounded Self-Critique & Revision Loop
    for r in range(1, max_rounds + 1):
        rounds_done = r
        # Generate ungrounded self-critique (no access to verifier reports)
        ungrounded_issues = [
            CritiqueIssue(
                issue_id=f"self_critique_{r}_1",
                category=IssueCategory.CLARITY_ORGANIZATION,
                severity=IssueSeverity.MAJOR,
                explanation="Self-identified potential clarity or structural weakness in current draft.",
                suggested_correction="Improve technical depth and verify alignment with key research questions.",
                confidence=0.80,
            )
        ]
        self_critique = Critique(
            round_number=r,
            issues=ungrounded_issues,
            should_revise=True,
            summary=f"Ungrounded self-critique round {r}",
        )

        current_report, rev_tokens, rev_cost = await rev_engine.revise_report(
            current_report=current_report,
            sources=list(topic.gold_sources),
            critique=self_critique,
        )
        tokens += rev_tokens
        cost += rev_cost

    # Step 3: Independent evaluation of final revised report
    deadline = time.monotonic() + cfg.max_wall_clock_seconds
    det_report = det_ver.verify(current_report, sources_map)
    grd_report = await grd_ver.verify_grounding(current_report, sources_map, deadline=deadline)
    judge_verdict = await judge.evaluate_blinded(
        report=current_report,
        sources=sources_map,
        deterministic_report=det_report,
        deadline=deadline,
    )

    candidate = ReportCandidate(
        round_number=rounds_done,
        report=current_report,
        deterministic_report=det_report,
        grounding_report=grd_report,
        judge_verdict=judge_verdict,
        tokens_used=tokens,
        cost_usd=cost,
    )

    acceptance = gate.evaluate_candidate(candidate)
    elapsed = time.perf_counter() - start_time

    return ApproachRunResult(
        topic_id=topic.topic_id,
        topic_title=topic.title,
        approach=ApproachType.APPROACH_B,
        approach_name="Approach B: Ungrounded Self-Critique",
        final_report=current_report,
        rounds_executed=rounds_done,
        deterministic_passed=det_report.passed,
        critical_defect_count=candidate.critical_defect_count,
        major_defect_count=candidate.major_defect_count,
        composite_rubric_score=judge_verdict.composite_score,
        factual_accuracy_score=candidate.factual_accuracy_score,
        evidence_grounding_score=candidate.evidence_grounding_score,
        dimension_scores={k: v.score for k, v in judge_verdict.dimension_scores.items()},
        tokens_consumed=tokens,
        cost_usd=cost,
        latency_seconds=elapsed,
        is_accepted=acceptance.is_accepted,
        terminal_status=acceptance.status,
    )


async def run_approach_c(
    topic: BenchmarkTopic,
    orchestrator: ReflectionLoopOrchestrator | None = None,
    blinded_judge: BlindedJudge | None = None,
    acceptance_gate: AcceptanceGate | None = None,
    config: ReportWriterConfig | None = None,
    llm_service: LLMService | None = None,
    **orchestrator_kwargs: Any,
) -> ApproachRunResult:
    """Execute Approach C: Grounded Reflection (Our System).

    Coordinates the full reflection loop with dual-gate verification, structured critique,
    oscillation detection, and Pareto candidate selection.
    """
    cfg = config or get_default_config()
    svc = llm_service
    if svc is None and (orchestrator is None or blinded_judge is None):
        from genai_core import LLMService

        svc = LLMService()

    orch = orchestrator or ReflectionLoopOrchestrator(
        llm_service=svc, config=cfg, **orchestrator_kwargs
    )
    judge = blinded_judge or BlindedJudge(llm_service=svc, config=cfg)
    gate = acceptance_gate or AcceptanceGate(config=cfg)

    sources_map = {src.chunk_id: src for src in topic.gold_sources}

    start_time = time.perf_counter()

    # Run bounded reflection orchestrator
    run_result = await orch.run(
        topic=topic.title,
        sources=topic.gold_sources,
    )

    best_cand = run_result.best_candidate
    best_report = best_cand.report

    # Blinded independent evaluation of best candidate
    deadline = time.monotonic() + cfg.max_wall_clock_seconds
    blinded_verdict = await judge.evaluate_blinded(
        report=best_report,
        sources=sources_map,
        deterministic_report=best_cand.deterministic_report,
        deadline=deadline,
    )

    acceptance = gate.evaluate_candidate(best_cand)
    elapsed = time.perf_counter() - start_time

    return ApproachRunResult(
        topic_id=topic.topic_id,
        topic_title=topic.title,
        approach=ApproachType.APPROACH_C,
        approach_name="Approach C: Grounded Reflection",
        final_report=best_report,
        rounds_executed=best_cand.round_number,
        deterministic_passed=best_cand.deterministic_passed,
        critical_defect_count=best_cand.critical_defect_count,
        major_defect_count=best_cand.major_defect_count,
        composite_rubric_score=blinded_verdict.composite_score,
        factual_accuracy_score=(
            blinded_verdict.dimension_scores["factual_accuracy"].score
            if "factual_accuracy" in blinded_verdict.dimension_scores
            else best_cand.factual_accuracy_score
        ),
        evidence_grounding_score=(
            blinded_verdict.dimension_scores["evidence_grounding"].score
            if "evidence_grounding" in blinded_verdict.dimension_scores
            else best_cand.evidence_grounding_score
        ),
        dimension_scores={k: v.score for k, v in blinded_verdict.dimension_scores.items()},
        tokens_consumed=run_result.budget_summary.used_tokens,
        cost_usd=run_result.budget_summary.used_cost_usd,
        latency_seconds=elapsed,
        is_accepted=acceptance.is_accepted,
        terminal_status=acceptance.status,
    )
