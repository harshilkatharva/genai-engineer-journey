"""AcceptanceGate: evaluates report candidates against strict automated acceptance criteria (Section 6.1)."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log

from self_improving_report_writer.core.config import ReportWriterConfig, get_default_config
from self_improving_report_writer.models.outcome import (
    AcceptanceResult,
    HumanReviewPacket,
    ReportStatus,
)

if TYPE_CHECKING:
    from self_improving_report_writer.models.critique import CritiqueIssue
    from self_improving_report_writer.reflection.candidate_selector import ReportCandidate
    from self_improving_report_writer.reflection.orchestrator import ReflectionRunResult

logger = logging.getLogger(__name__)


class AcceptanceGate:
    """Enforces honest machine acceptance vs unresolved determination according to Section 6.1."""

    def __init__(self, config: ReportWriterConfig | None = None) -> None:
        self.config = config or get_default_config()

    def evaluate_candidate(
        self,
        candidate: ReportCandidate,
        rounds_executed: int = 0,
        terminal_reason: str = "MANUAL_EVALUATION",
        run_id: str | None = None,
        unresolved_critiques: list[CritiqueIssue] | None = None,
    ) -> AcceptanceResult:
        """Evaluate an individual ReportCandidate directly against acceptance criteria."""
        active_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        unresolved_reasons: list[str] = []

        # 1. Gate 1 Deterministic Verification Check
        if not candidate.deterministic_report.passed:
            unresolved_reasons.append(
                f"Gate 1 Deterministic Verification failed (schema_valid={candidate.deterministic_report.schema_valid}, "
                f"citations_valid={candidate.deterministic_report.citations_valid}, "
                f"passages_matched={candidate.deterministic_report.passages_matched})"
            )

        # 2. Zero Critical Defects Check
        if candidate.total_critical_count > 0:
            unresolved_reasons.append(
                f"Unresolved critical defects remain ({candidate.total_critical_count} critical defects)"
            )

        # 3. Zero Major Factual Grounding Defects Check
        if candidate.grounding_report.major_count > 0:
            unresolved_reasons.append(
                f"Unresolved major factual grounding defects remain "
                f"({candidate.grounding_report.major_count} major grounding defects)"
            )

        # 4. Composite Rubric Score Threshold Check (with self-judging disclosure adjustment)
        effective_threshold = (
            self.config.acceptance_threshold + 0.05
            if candidate.judge_verdict.is_self_judged
            else self.config.acceptance_threshold
        )
        if candidate.judge_verdict.composite_score < effective_threshold:
            unresolved_reasons.append(
                f"Composite rubric score {candidate.judge_verdict.composite_score:.4f} is below "
                f"acceptance threshold {effective_threshold:.2f} "
                f"(self_judged={candidate.judge_verdict.is_self_judged})"
            )

        # 5. Factual Accuracy and Evidence Grounding Minimum Pass Checks
        scores = candidate.judge_verdict.dimension_scores
        fa = scores.get("factual_accuracy")
        eg = scores.get("evidence_grounding")

        if fa and fa.score < 0.85:
            unresolved_reasons.append(
                f"Factual Accuracy dimension score {fa.score:.2f} is below minimum standard (0.85): {fa.reasoning}"
            )
        elif not fa:
            unresolved_reasons.append(
                "Factual Accuracy dimension score is missing from judge verdict"
            )

        if eg and eg.score < 0.85:
            unresolved_reasons.append(
                f"Evidence Grounding dimension score {eg.score:.2f} is below minimum standard (0.85): {eg.reasoning}"
            )
        elif not eg:
            unresolved_reasons.append(
                "Evidence Grounding dimension score is missing from judge verdict"
            )

        # Determine outcome
        if not unresolved_reasons:
            safe_log(
                logger,
                logging.INFO,
                f"Report candidate for topic '{candidate.report.topic}' ACCEPTED "
                f"(composite={candidate.judge_verdict.composite_score:.4f}, round={candidate.round_number})",
                event=ObservabilityEvent.TOOL_EXECUTION_SUCCESS,
            )
            return AcceptanceResult(
                status=ReportStatus.ACCEPTED,
                is_accepted=True,
                best_candidate=candidate,
                rationale=(
                    f"Candidate meets all deterministic and semantic acceptance standards "
                    f"(composite_score={candidate.judge_verdict.composite_score:.4f} >= {effective_threshold:.2f}, "
                    f"critical=0, major_grounding=0)."
                ),
                unresolved_reasons=[],
                review_packet=None,
            )

        # Report is UNRESOLVED: construct HumanReviewPacket
        rubric_score_map = {k: v.score for k, v in candidate.judge_verdict.dimension_scores.items()}
        review_packet = HumanReviewPacket(
            run_id=active_run_id,
            topic=candidate.report.topic,
            best_candidate=candidate.report,
            best_candidate_version=candidate.report.version,
            total_rounds_executed=rounds_executed,
            terminal_reason=terminal_reason,
            deterministic_passed=candidate.deterministic_report.passed,
            critical_defect_count=candidate.total_critical_count,
            major_defect_count=candidate.total_major_count,
            rubric_scores=rubric_score_map,
            unresolved_critiques=unresolved_critiques or [],
            requires_human_action=True,
        )

        safe_log(
            logger,
            logging.WARNING,
            f"Report candidate for topic '{candidate.report.topic}' marked UNRESOLVED "
            f"due to {len(unresolved_reasons)} unmet criteria: {'; '.join(unresolved_reasons)}",
            event=ObservabilityEvent.TOOL_APPROVAL_REQUIRED,
        )

        return AcceptanceResult(
            status=ReportStatus.UNRESOLVED,
            is_accepted=False,
            best_candidate=candidate,
            rationale=(
                f"Candidate did not meet automated acceptance criteria: {'; '.join(unresolved_reasons)}"
            ),
            unresolved_reasons=unresolved_reasons,
            review_packet=review_packet,
        )

    def evaluate_run_result(
        self,
        run_result: ReflectionRunResult,
        run_id: str | None = None,
        unresolved_critiques: list[CritiqueIssue] | None = None,
    ) -> AcceptanceResult:
        """Evaluate a completed ReflectionRunResult."""
        return self.evaluate_candidate(
            candidate=run_result.best_candidate,
            rounds_executed=run_result.rounds_executed,
            terminal_reason=run_result.termination_reason,
            run_id=run_id,
            unresolved_critiques=unresolved_critiques,
        )
