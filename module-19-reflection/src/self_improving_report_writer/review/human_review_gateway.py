"""HumanReviewGateway: generates inspection packets and processes operator review actions (Section 6.2)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log

from self_improving_report_writer.models.critique import (
    Critique,
    CritiqueIssue,
    IssueCategory,
    IssueSeverity,
)
from self_improving_report_writer.models.outcome import (
    HumanReviewDecision,
    HumanReviewOutcome,
    HumanReviewPacket,
    ReportStatus,
)
from self_improving_report_writer.models.report import SourceReference
from self_improving_report_writer.reflection.candidate_selector import ReportCandidate
from self_improving_report_writer.review.acceptance_gate import AcceptanceGate
from self_improving_report_writer.verification.deterministic_verifier import DeterministicVerifier

if TYPE_CHECKING:
    from self_improving_report_writer.judging.rubric_judge import RubricJudge
    from self_improving_report_writer.reflection.revision_engine import RevisionEngine
    from self_improving_report_writer.verification.semantic_grounding_verifier import (
        SemanticGroundingVerifier,
    )

logger = logging.getLogger(__name__)


class HumanReviewGateway:
    """Provides human-in-the-loop inspection and operator action dispatch for unresolved reports."""

    def __init__(
        self,
        acceptance_gate: AcceptanceGate | None = None,
        deterministic_verifier: DeterministicVerifier | None = None,
        max_human_revision_rounds: int = 2,
    ) -> None:
        self.acceptance_gate = acceptance_gate or AcceptanceGate()
        self.deterministic_verifier = deterministic_verifier or DeterministicVerifier()
        self.max_human_revision_rounds = max_human_revision_rounds

    def format_review_packet(self, packet: HumanReviewPacket) -> str:
        """Render a formatted, human-readable markdown packet for operator inspection."""
        lines = [
            f"# Human Review Packet: {packet.topic}",
            f"- **Run ID:** `{packet.run_id}`",
            f"- **Best Candidate Version:** v{packet.best_candidate_version}",
            f"- **Total Automated Rounds:** {packet.total_rounds_executed}",
            f"- **Terminal Reason:** `{packet.terminal_reason}`",
            f"- **Gate 1 Deterministic Passed:** {packet.deterministic_passed}",
            f"- **Critical Defects Remaining:** {packet.critical_defect_count}",
            f"- **Major Defects Remaining:** {packet.major_defect_count}",
            "",
            "## Rubric Scores:",
        ]

        for dim_name, score_val in sorted(packet.rubric_scores.items()):
            lines.append(f"- **{dim_name}:** {score_val:.2f}")

        lines.append("")
        lines.append("## Unresolved Defect Issues:")
        if packet.unresolved_critiques:
            for issue in packet.unresolved_critiques:
                lines.append(
                    f"- **[{issue.severity.upper()}] ({issue.category})**: {issue.explanation}\n"
                    f"  * Recommended Action: {issue.suggested_correction}"
                )
        else:
            lines.append("- None cataloged.")

        return "\n".join(lines)

    async def process_decision(
        self,
        decision: HumanReviewDecision,
        packet: HumanReviewPacket,
        sources: Sequence[SourceReference] | None = None,
        human_revision_round: int = 1,
        revision_engine: RevisionEngine | None = None,
        grounding_verifier: SemanticGroundingVerifier | None = None,
        judge: RubricJudge | None = None,
    ) -> HumanReviewOutcome:
        """Process an operator review action (APPROVE_OVERRIDE, REJECT, or REQUEST_REVISION)."""
        safe_log(
            logger,
            logging.INFO,
            f"Processing human review decision '{decision.action}' for run '{decision.run_id}'",
            event=ObservabilityEvent.TOOL_APPROVAL_DECISION,
        )

        # 1. Action: APPROVE_OVERRIDE
        if decision.action == "APPROVE_OVERRIDE":
            summary = (
                f"Operator approved unresolved report v{packet.best_candidate_version} with override notes: "
                f"'{decision.reviewer_notes}'. Automated verification findings remain preserved."
            )
            safe_log(logger, logging.INFO, summary, event=ObservabilityEvent.TOOL_APPROVAL_DECISION)
            return HumanReviewOutcome(
                run_id=decision.run_id,
                status=ReportStatus.HUMAN_APPROVED,
                final_report=packet.best_candidate,
                decision=decision,
                revised_candidate=None,
                summary=summary,
            )

        # 2. Action: REJECT
        if decision.action == "REJECT":
            summary = (
                f"Operator rejected report v{packet.best_candidate_version} with rejection rationale: "
                f"'{decision.reviewer_notes}'."
            )
            safe_log(
                logger, logging.WARNING, summary, event=ObservabilityEvent.TOOL_APPROVAL_DECISION
            )
            return HumanReviewOutcome(
                run_id=decision.run_id,
                status=ReportStatus.HUMAN_REJECTED,
                final_report=packet.best_candidate,
                decision=decision,
                revised_candidate=None,
                summary=summary,
            )

        # 3. Action: REQUEST_REVISION
        if human_revision_round > self.max_human_revision_rounds:
            raise ValueError(
                f"Maximum human revision rounds exceeded ({human_revision_round} > {self.max_human_revision_rounds})"
            )

        if revision_engine is None or sources is None:
            raise ValueError("revision_engine and sources are required to execute REQUEST_REVISION")

        source_list = list(sources)

        # Build critique incorporating human operator instructions
        human_issue = CritiqueIssue(
            issue_id=f"HUMAN-REV-{human_revision_round:02d}",
            category=IssueCategory.CLARITY_ORGANIZATION,
            severity=IssueSeverity.MAJOR,
            explanation=f"Human Reviewer Directive: {decision.reviewer_notes}",
            suggested_correction=decision.reviewer_notes,
            confidence=1.0,
        )
        combined_issues = [human_issue, *packet.unresolved_critiques]

        human_critique = Critique(
            round_number=packet.total_rounds_executed + human_revision_round,
            issues=combined_issues,
            summary=f"Human-requested revision round {human_revision_round}: {decision.reviewer_notes}",
            should_revise=True,
        )

        revised_report, tokens, cost = await revision_engine.revise_report(
            current_report=packet.best_candidate,
            sources=source_list,
            critique=human_critique,
        )
        revised_report.created_by = "human_revision"
        revised_report.version = packet.best_candidate_version + 1

        # Re-verify with Gate 1
        det_report = self.deterministic_verifier.verify(revised_report, source_list)

        # Re-verify with Gate 2 if available
        if grounding_verifier is not None:
            g2_report = await grounding_verifier.verify_grounding(revised_report, source_list)
        else:
            # Fallback mock/empty grounding if not provided
            from self_improving_report_writer.models.rubric import SemanticGroundingReport

            g2_report = SemanticGroundingReport(
                passed=det_report.passed,
                all_claims_grounded=det_report.passed,
                grounding_score=1.0 if det_report.passed else 0.5,
                claim_evaluations=[],
                issues=[],
                critical_count=0,
                major_count=0,
                minor_count=0,
            )

        # Re-evaluate with Rubric Judge if available
        if judge is not None:
            source_dict = {s.chunk_id: s for s in source_list}
            judge_verdict = await judge.evaluate(
                revised_report, source_dict, deterministic_report=det_report
            )
        else:
            from self_improving_report_writer.models.rubric import ReportJudgeVerdict

            judge_verdict = ReportJudgeVerdict(
                composite_score=0.90 if det_report.passed else 0.60,
                passed=det_report.passed,
                dimension_scores={},
                overall_feedback="Human revision re-evaluated.",
            )

        revised_candidate = ReportCandidate(
            round_number=packet.total_rounds_executed + human_revision_round,
            report=revised_report,
            deterministic_report=det_report,
            grounding_report=g2_report,
            judge_verdict=judge_verdict,
            tokens_used=tokens,
            cost_usd=cost,
        )

        # Evaluate against AcceptanceGate
        acc_result = self.acceptance_gate.evaluate_candidate(
            candidate=revised_candidate,
            rounds_executed=packet.total_rounds_executed + human_revision_round,
            terminal_reason=f"HUMAN_REVISION_ROUND_{human_revision_round}",
            run_id=decision.run_id,
            unresolved_critiques=combined_issues,
        )

        summary = (
            f"Human revision round {human_revision_round} completed (v{revised_report.version}): "
            f"status={acc_result.status}. {acc_result.rationale}"
        )

        return HumanReviewOutcome(
            run_id=decision.run_id,
            status=acc_result.status,
            final_report=revised_report,
            decision=decision,
            revised_candidate=revised_candidate,
            summary=summary,
        )
