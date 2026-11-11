"""ReflectionLoopOrchestrator: manages iterative reflection, multi-resource budget, and candidate selection."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log
from pydantic import BaseModel, Field

from self_improving_report_writer.core.budget import BudgetSummary, RunBudget
from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.generation.report_generator import ReportGenerator
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.memory.reflection_memory_manager import ReflectionMemoryManager
from self_improving_report_writer.models.critique import Critique
from self_improving_report_writer.models.outcome import AcceptanceResult, HumanReviewPacket
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.reflection.candidate_selector import (
    CandidateSelector,
    ReportCandidate,
)
from self_improving_report_writer.reflection.critique_engine import CritiqueEngine
from self_improving_report_writer.reflection.oscillation_detector import OscillationDetector
from self_improving_report_writer.reflection.revision_engine import RevisionEngine
from self_improving_report_writer.retrieval.source_retriever import SourceRetriever
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate
from self_improving_report_writer.verification.deterministic_verifier import DeterministicVerifier
from self_improving_report_writer.verification.semantic_grounding_verifier import (
    SemanticGroundingVerifier,
)

if TYPE_CHECKING:
    from genai_core import LLMService

logger = logging.getLogger(__name__)


class ReflectionRunResult(BaseModel):
    """Encapsulates the complete trajectory, metrics, and final outcome of a reflection run."""

    topic: str = Field(description="The research topic or prompt")
    best_candidate: ReportCandidate = Field(
        description="The highest-ranking candidate selected via deterministic hierarchical comparison"
    )
    all_candidates: list[ReportCandidate] = Field(
        default_factory=list,
        description="All candidates produced across baseline and revision rounds",
    )
    budget_summary: BudgetSummary = Field(
        description="Final resource consumption metrics across tokens, cost, time, and rounds"
    )
    rounds_executed: int = Field(
        ge=0, description="Number of revision rounds completed (0 = baseline only)"
    )
    termination_reason: str = Field(
        description="Specific reason the reflection loop terminated (e.g., criteria met, budget, oscillation)"
    )
    acceptance_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Base acceptance threshold configured for the reflection run",
    )
    acceptance_result: AcceptanceResult | None = Field(
        default=None,
        description="Integrated AcceptanceGate verdict and optional HumanReviewPacket",
    )
    learned_lessons_count: int = Field(
        default=0,
        ge=0,
        description="Number of authoring lessons distilled into episodic memory for this run",
    )

    @property
    def review_packet(self) -> HumanReviewPacket | None:
        """Return the HumanReviewPacket when AcceptanceGate marked the run UNRESOLVED."""
        if self.acceptance_result is None:
            return None
        return self.acceptance_result.review_packet

    @property
    def is_accepted(self) -> bool:
        """Evaluate whether best_candidate passes automated acceptance criteria (Section 6.1)."""
        if self.acceptance_result is not None:
            return self.acceptance_result.is_accepted
        return (
            AcceptanceGate(
                config=ReportWriterConfig(acceptance_threshold=self.acceptance_threshold)
            )
            .evaluate_candidate(
                candidate=self.best_candidate,
                rounds_executed=self.rounds_executed,
                terminal_reason=self.termination_reason,
            )
            .is_accepted
        )


class ReflectionLoopOrchestrator:
    """Coordinates baseline generation, dual-gate verification, rubric judging, and bounded revision."""

    def __init__(
        self,
        llm_service: LLMService,
        config: ReportWriterConfig | None = None,
        retriever: SourceRetriever | None = None,
        generator: ReportGenerator | None = None,
        deterministic_verifier: DeterministicVerifier | None = None,
        grounding_verifier: SemanticGroundingVerifier | None = None,
        judge: RubricJudge | None = None,
        critique_engine: CritiqueEngine | None = None,
        oscillation_detector: OscillationDetector | None = None,
        revision_engine: RevisionEngine | None = None,
        acceptance_gate: AcceptanceGate | None = None,
        memory_manager: ReflectionMemoryManager | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.config = config or get_default_config()

        self.retriever = retriever or SourceRetriever(config=self.config)
        self.generator = generator or ReportGenerator(
            llm_service=self.llm_service,
            config=self.config,
        )
        self.deterministic_verifier = deterministic_verifier or DeterministicVerifier()
        self.grounding_verifier = grounding_verifier or SemanticGroundingVerifier(
            llm_service=self.llm_service,
            config=self.config,
        )
        self.judge = judge or RubricJudge(
            llm_service=self.llm_service,
            config=self.config,
        )
        self.critique_engine = critique_engine or CritiqueEngine()
        self.oscillation_detector = oscillation_detector or OscillationDetector()
        self.revision_engine = revision_engine or RevisionEngine(
            llm_service=self.llm_service,
            config=self.config,
        )
        self.acceptance_gate = acceptance_gate or AcceptanceGate(config=self.config)
        self.memory_manager = memory_manager

    async def run(
        self,
        topic: str,
        sources: Sequence[SourceReference] | None = None,
        initial_report: Report | None = None,
        guidelines: list[str] | None = None,
        domain: str = "general",
        run_id: str | None = None,
    ) -> ReflectionRunResult:
        """Execute the self-improving reflection loop, acceptance gate, and optional memory integration.

        Args:
            topic: Natural language topic or research prompt.
            sources: Optional pre-retrieved source references. If None, retrieved via SourceRetriever.
            initial_report: Optional pre-generated baseline report (Round 0).
            guidelines: Optional authoring guidelines. If None and memory_manager is configured,
                recent domain lessons are retrieved automatically.
            domain: Topical domain classification used when retrieving or distilling memory lessons.
            run_id: Optional run identifier forwarded to AcceptanceGate review packets.

        Returns:
            ReflectionRunResult containing best candidate, all candidates, budget summary,
            and integrated AcceptanceResult.
        """
        return await self._execute_run(
            topic=topic,
            sources=sources,
            initial_report=initial_report,
            guidelines=guidelines,
            domain=domain,
            run_id=run_id,
            acceptance_gate=self.acceptance_gate,
            memory_manager=self.memory_manager,
        )

    async def run_and_evaluate(
        self,
        topic: str,
        sources: Sequence[SourceReference] | None = None,
        initial_report: Report | None = None,
        guidelines: list[str] | None = None,
        domain: str = "general",
        run_id: str | None = None,
        acceptance_gate: AcceptanceGate | None = None,
        memory_manager: ReflectionMemoryManager | None = None,
    ) -> ReflectionRunResult:
        """Convenience wrapper executing reflection, AcceptanceGate evaluation, and memory updates."""
        return await self._execute_run(
            topic=topic,
            sources=sources,
            initial_report=initial_report,
            guidelines=guidelines,
            domain=domain,
            run_id=run_id,
            acceptance_gate=acceptance_gate or self.acceptance_gate,
            memory_manager=memory_manager if memory_manager is not None else self.memory_manager,
        )

    async def _execute_run(
        self,
        topic: str,
        sources: Sequence[SourceReference] | None,
        initial_report: Report | None,
        guidelines: list[str] | None,
        domain: str,
        run_id: str | None,
        acceptance_gate: AcceptanceGate,
        memory_manager: ReflectionMemoryManager | None,
    ) -> ReflectionRunResult:
        """Internal reflection loop execution with integrated gate and episodic memory hooks."""
        budget = RunBudget(config=self.config)
        self.oscillation_detector.reset()
        candidates: list[ReportCandidate] = []
        critiques_by_round: dict[int, Critique] = {}
        consecutive_regressions = 0

        safe_log(
            logger,
            logging.INFO,
            f"Starting ReflectionLoopOrchestrator for topic '{topic}' "
            f"(max_rounds={self.config.max_reflection_rounds}, max_cost=${self.config.max_total_cost_usd})",
            event=ObservabilityEvent.TOOL_LOOP_START,
        )

        # Retrieve domain authoring guidelines from episodic memory when configured
        effective_guidelines = guidelines
        if effective_guidelines is None and initial_report is None and memory_manager is not None:
            retrieved_lessons = await memory_manager.retrieve_lessons(domain=domain)
            if retrieved_lessons:
                effective_guidelines = retrieved_lessons

        # 1. Resolve Sources
        if sources is None:
            source_list = await self.retriever.retrieve_sources(
                topic=topic,
                top_k=self.config.default_top_k,
                min_score=self.config.min_retrieval_score,
            )
        else:
            source_list = list(sources)

        # 2. Baseline Generation (Round 0)
        budget.assert_within_budget(current_round=0)
        if initial_report is not None:
            current_report = initial_report
            gen_tokens = 0
            gen_cost = 0.0
        else:
            current_report, gen_tokens, gen_cost = await self.generator.generate_report(
                topic=topic,
                sources=source_list,
                guidelines=effective_guidelines,
            )
            budget.record_usage(tokens=gen_tokens, cost_usd=gen_cost)

        source_dict = {s.chunk_id: s for s in source_list}

        # 3. Evaluate Baseline Candidate
        det_report = self.deterministic_verifier.verify(current_report, source_list)
        deadline = time.monotonic() + budget.remaining_seconds
        g2_report = await self.grounding_verifier.verify_grounding(
            current_report, source_list, deadline=deadline
        )
        budget.record_usage(tokens=g2_report.tokens_used, cost_usd=g2_report.cost_usd)

        judge_verdict = await self.judge.evaluate(
            current_report,
            source_dict,
            deterministic_report=det_report,
            deadline=deadline,
        )
        budget.record_usage(tokens=judge_verdict.tokens_used, cost_usd=judge_verdict.cost_usd)

        baseline_candidate = ReportCandidate(
            round_number=0,
            report=current_report,
            deterministic_report=det_report,
            grounding_report=g2_report,
            judge_verdict=judge_verdict,
            tokens_used=gen_tokens + g2_report.tokens_used + judge_verdict.tokens_used,
            cost_usd=round(gen_cost + g2_report.cost_usd + judge_verdict.cost_usd, 6),
        )
        candidates.append(baseline_candidate)

        # Generate baseline critique
        critique = self.critique_engine.generate_critique(
            round_number=0,
            report=current_report,
            deterministic_report=det_report,
            grounding_report=g2_report,
            judge_verdict=judge_verdict,
            acceptance_threshold=self.config.acceptance_threshold,
        )
        critiques_by_round[0] = critique

        # Check immediate acceptance
        if not critique.should_revise:
            termination_reason = "ALL_ACCEPTANCE_CRITERIA_MET"
            return await self._finalize_run(
                topic=topic,
                candidates=candidates,
                budget=budget,
                rounds_executed=0,
                termination_reason=termination_reason,
                critiques_by_round=critiques_by_round,
                run_id=run_id,
                domain=domain,
                acceptance_gate=acceptance_gate,
                memory_manager=memory_manager,
            )

        # Record round 0 fingerprints
        self.oscillation_detector.check_oscillation(0, current_report, critique)

        # 4. Iterative Revision Loop
        termination_reason = "MAX_ROUNDS_REACHED"

        for round_idx in range(1, self.config.max_reflection_rounds + 1):
            # Check budget before revising
            breach = budget.check_budget_exceeded(current_round=round_idx - 1)
            if breach:
                termination_reason = self._format_budget_termination_reason(breach)
                break

            safe_log(
                logger,
                logging.INFO,
                f"Starting reflection round {round_idx}/{self.config.max_reflection_rounds} "
                f"(current_budget: tokens={budget.used_tokens}/{budget.max_tokens}, "
                f"cost=${budget.used_cost_usd:.4f}/${budget.max_cost_usd:.2f})",
                event=ObservabilityEvent.TOOL_LOOP_STEP,
            )

            # Revise report
            revised_report, rev_tokens, rev_cost = await self.revision_engine.revise_report(
                current_report=current_report,
                sources=source_list,
                critique=critique,
            )
            budget.record_usage(tokens=rev_tokens, cost_usd=rev_cost)

            # Evaluate revised report
            rev_det_report = self.deterministic_verifier.verify(revised_report, source_list)
            deadline = time.monotonic() + budget.remaining_seconds
            rev_g2_report = await self.grounding_verifier.verify_grounding(
                revised_report, source_list, deadline=deadline
            )
            budget.record_usage(tokens=rev_g2_report.tokens_used, cost_usd=rev_g2_report.cost_usd)

            deadline = time.monotonic() + budget.remaining_seconds
            rev_judge_verdict = await self.judge.evaluate(
                revised_report,
                source_dict,
                deterministic_report=rev_det_report,
                deadline=deadline,
            )
            budget.record_usage(
                tokens=rev_judge_verdict.tokens_used, cost_usd=rev_judge_verdict.cost_usd
            )

            rev_candidate = ReportCandidate(
                round_number=round_idx,
                report=revised_report,
                deterministic_report=rev_det_report,
                grounding_report=rev_g2_report,
                judge_verdict=rev_judge_verdict,
                tokens_used=rev_tokens + rev_g2_report.tokens_used + rev_judge_verdict.tokens_used,
                cost_usd=round(rev_cost + rev_g2_report.cost_usd + rev_judge_verdict.cost_usd, 6),
            )

            # Check for regression relative to previous candidate
            if CandidateSelector.is_regression(candidates[-1], rev_candidate):
                consecutive_regressions += 1
                safe_log(
                    logger,
                    logging.WARNING,
                    f"Round {round_idx} candidate regressed compared to round {candidates[-1].round_number} "
                    f"(consecutive_regressions={consecutive_regressions}/{self.config.max_consecutive_regressions})",
                    event=ObservabilityEvent.TOOL_EXECUTION_FAILED,
                )
            else:
                consecutive_regressions = 0

            candidates.append(rev_candidate)
            current_report = revised_report
            budget.record_round_completed()

            # Enforce max consecutive regressions
            if consecutive_regressions >= self.config.max_consecutive_regressions:
                termination_reason = f"CONSECUTIVE_REGRESSIONS_EXCEEDED ({consecutive_regressions} consecutive regressions)"
                break

            # Generate critique for revised candidate
            critique = self.critique_engine.generate_critique(
                round_number=round_idx,
                report=revised_report,
                deterministic_report=rev_det_report,
                grounding_report=rev_g2_report,
                judge_verdict=rev_judge_verdict,
                acceptance_threshold=self.config.acceptance_threshold,
            )
            critiques_by_round[round_idx] = critique

            # Check if all acceptance criteria are met
            if not critique.should_revise:
                termination_reason = "ALL_ACCEPTANCE_CRITERIA_MET"
                break

            # Check for oscillation / stagnation
            osc_check = self.oscillation_detector.check_oscillation(
                round_idx, revised_report, critique
            )
            if osc_check.has_oscillation:
                termination_reason = (
                    f"OSCILLATION_DETECTED: {osc_check.oscillation_type} ({osc_check.details})"
                )
                break

            # Check if round or resource budget exhausted at end of round
            breach = budget.check_budget_exceeded(current_round=round_idx)
            if breach:
                termination_reason = self._format_budget_termination_reason(breach)
                break

        return await self._finalize_run(
            topic=topic,
            candidates=candidates,
            budget=budget,
            rounds_executed=budget.rounds_executed,
            termination_reason=termination_reason,
            critiques_by_round=critiques_by_round,
            run_id=run_id,
            domain=domain,
            acceptance_gate=acceptance_gate,
            memory_manager=memory_manager,
        )

    @staticmethod
    def _format_budget_termination_reason(breach: str) -> str:
        """Distinguish normal max-rounds completion from resource budget exhaustion."""
        if breach.startswith("MAX_ROUNDS_REACHED"):
            return "MAX_ROUNDS_REACHED"
        return f"BUDGET_EXCEEDED: {breach}"

    async def _finalize_run(
        self,
        topic: str,
        candidates: list[ReportCandidate],
        budget: RunBudget,
        rounds_executed: int,
        termination_reason: str,
        critiques_by_round: dict[int, Critique],
        run_id: str | None,
        domain: str,
        acceptance_gate: AcceptanceGate,
        memory_manager: ReflectionMemoryManager | None,
    ) -> ReflectionRunResult:
        """Assemble integrated run result and distill lessons when accepted and configured."""
        result = self._build_result(
            topic=topic,
            candidates=candidates,
            budget=budget,
            rounds_executed=rounds_executed,
            termination_reason=termination_reason,
            critiques_by_round=critiques_by_round,
            run_id=run_id,
            acceptance_gate=acceptance_gate,
        )
        if memory_manager is not None and result.is_accepted:
            recorded_lessons = await memory_manager.distill_lessons(
                run_result=result,
                domain=domain,
            )
            if recorded_lessons:
                result = result.model_copy(update={"learned_lessons_count": len(recorded_lessons)})
        return result

    def _build_result(
        self,
        topic: str,
        candidates: list[ReportCandidate],
        budget: RunBudget,
        rounds_executed: int,
        termination_reason: str,
        critiques_by_round: dict[int, Critique] | None = None,
        run_id: str | None = None,
        acceptance_gate: AcceptanceGate | None = None,
    ) -> ReflectionRunResult:
        """Select best candidate, evaluate AcceptanceGate, and assemble ReflectionRunResult."""
        best_candidate = CandidateSelector.select_best(candidates)
        gate = acceptance_gate or self.acceptance_gate
        critique_map = critiques_by_round or {}
        best_critique = critique_map.get(best_candidate.round_number)
        unresolved_critiques = best_critique.issues if best_critique is not None else None

        acceptance_result = gate.evaluate_candidate(
            candidate=best_candidate,
            rounds_executed=rounds_executed,
            terminal_reason=termination_reason,
            run_id=run_id,
            unresolved_critiques=unresolved_critiques,
        )

        safe_log(
            logger,
            logging.INFO,
            f"Reflection loop completed for '{topic}': best_round={best_candidate.round_number}, "
            f"rounds_executed={rounds_executed}, termination='{termination_reason}', "
            f"best_composite={best_candidate.judge_verdict.composite_score:.4f}, "
            f"status={acceptance_result.status}",
            event=ObservabilityEvent.TOOL_LOOP_COMPLETED,
        )

        return ReflectionRunResult(
            topic=topic,
            best_candidate=best_candidate,
            all_candidates=candidates,
            budget_summary=budget.to_summary(),
            rounds_executed=rounds_executed,
            termination_reason=termination_reason,
            acceptance_threshold=gate.config.acceptance_threshold,
            acceptance_result=acceptance_result,
        )
