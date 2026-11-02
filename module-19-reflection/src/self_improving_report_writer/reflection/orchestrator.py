"""ReflectionLoopOrchestrator: manages iterative reflection, multi-resource budget, and candidate selection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log
from pydantic import BaseModel, Field

from self_improving_report_writer.core.budget import BudgetSummary, RunBudget
from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.generation.report_generator import ReportGenerator
from self_improving_report_writer.judging.rubric_judge import RubricJudge
from self_improving_report_writer.models.report import Report, SourceReference
from self_improving_report_writer.reflection.candidate_selector import (
    CandidateSelector,
    ReportCandidate,
)
from self_improving_report_writer.reflection.critique_engine import CritiqueEngine
from self_improving_report_writer.reflection.oscillation_detector import OscillationDetector
from self_improving_report_writer.reflection.revision_engine import RevisionEngine
from self_improving_report_writer.retrieval.source_retriever import SourceRetriever
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

    @property
    def is_accepted(self) -> bool:
        """Evaluate whether best_candidate passes automated acceptance criteria (Section 6.1)."""
        cand = self.best_candidate
        if not cand.deterministic_report.passed:
            return False
        if cand.total_critical_count > 0 or cand.total_major_count > 0:
            return False

        effective_threshold = (
            cand.judge_verdict.effective_threshold
            if hasattr(cand.judge_verdict, "effective_threshold")
            else (0.90 if cand.judge_verdict.is_self_judged else 0.85)
        )
        if cand.judge_verdict.composite_score < effective_threshold:
            return False

        scores = cand.judge_verdict.dimension_scores
        fa = scores.get("factual_accuracy")
        eg = scores.get("evidence_grounding")
        if fa and fa.score < 0.85:
            return False
        return not (eg and eg.score < 0.85)


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
    ) -> None:
        self.llm_service = llm_service
        self.config = config or get_default_config()

        self.retriever = retriever or SourceRetriever()
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

    async def run(
        self,
        topic: str,
        sources: Sequence[SourceReference] | None = None,
        initial_report: Report | None = None,
        guidelines: list[str] | None = None,
    ) -> ReflectionRunResult:
        """Execute the self-improving reflection loop for a research topic.

        Args:
            topic: Natural language topic or research prompt.
            sources: Optional pre-retrieved source references. If None, retrieved via SourceRetriever.
            initial_report: Optional pre-generated baseline report (Round 0).
            guidelines: Optional authoring guidelines from episodic memory.

        Returns:
            ReflectionRunResult containing best candidate, all candidates, and budget summary.
        """
        budget = RunBudget(config=self.config)
        self.oscillation_detector.reset()
        candidates: list[ReportCandidate] = []
        consecutive_regressions = 0

        safe_log(
            logger,
            logging.INFO,
            f"Starting ReflectionLoopOrchestrator for topic '{topic}' "
            f"(max_rounds={self.config.max_reflection_rounds}, max_cost=${self.config.max_total_cost_usd})",
            event=ObservabilityEvent.TOOL_LOOP_START,
        )

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
                guidelines=guidelines,
            )
            budget.record_usage(tokens=gen_tokens, cost_usd=gen_cost)

        source_dict = {s.chunk_id: s for s in source_list}

        # 3. Evaluate Baseline Candidate
        det_report = self.deterministic_verifier.verify(current_report, source_list)
        g2_report = await self.grounding_verifier.verify_grounding(current_report, source_list)
        budget.record_usage(tokens=g2_report.tokens_used, cost_usd=g2_report.cost_usd)

        judge_verdict = await self.judge.evaluate(
            current_report, source_dict, deterministic_report=det_report
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

        # Check immediate acceptance
        if not critique.should_revise:
            termination_reason = "ALL_ACCEPTANCE_CRITERIA_MET"
            return self._build_result(topic, candidates, budget, 0, termination_reason)

        # Record round 0 fingerprints
        self.oscillation_detector.check_oscillation(0, current_report, critique)

        # 4. Iterative Revision Loop
        termination_reason = "MAX_ROUNDS_REACHED"

        for round_idx in range(1, self.config.max_reflection_rounds + 1):
            # Check budget before revising
            breach = budget.check_budget_exceeded(current_round=round_idx - 1)
            if breach:
                termination_reason = f"BUDGET_EXCEEDED: {breach}"
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
            rev_g2_report = await self.grounding_verifier.verify_grounding(
                revised_report, source_list
            )
            budget.record_usage(tokens=rev_g2_report.tokens_used, cost_usd=rev_g2_report.cost_usd)

            rev_judge_verdict = await self.judge.evaluate(
                revised_report, source_dict, deterministic_report=rev_det_report
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

            # Check if budget exhausted at end of round
            if budget.is_exhausted(current_round=round_idx):
                termination_reason = (
                    f"BUDGET_EXCEEDED: {budget.check_budget_exceeded(current_round=round_idx)}"
                )
                break

        return self._build_result(
            topic, candidates, budget, budget.rounds_executed, termination_reason
        )

    def _build_result(
        self,
        topic: str,
        candidates: list[ReportCandidate],
        budget: RunBudget,
        rounds_executed: int,
        termination_reason: str,
    ) -> ReflectionRunResult:
        """Select best candidate and assemble ReflectionRunResult."""
        best_candidate = CandidateSelector.select_best(candidates)

        safe_log(
            logger,
            logging.INFO,
            f"Reflection loop completed for '{topic}': best_round={best_candidate.round_number}, "
            f"rounds_executed={rounds_executed}, termination='{termination_reason}', "
            f"best_composite={best_candidate.judge_verdict.composite_score:.4f}",
            event=ObservabilityEvent.TOOL_LOOP_COMPLETED,
        )

        return ReflectionRunResult(
            topic=topic,
            best_candidate=best_candidate,
            all_candidates=candidates,
            budget_summary=budget.to_summary(),
            rounds_executed=rounds_executed,
            termination_reason=termination_reason,
        )
