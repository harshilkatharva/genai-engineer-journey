"""Human plan review gateway implementation."""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Any

from genai_core import safe_log

from ..core import AssistantConfig, get_assistant_config
from ..models import (
    Plan,
    PlanReviewDecision,
    PlanReviewRequest,
    ReviewAction,
)
from ..planning import PlanValidationError, PlanValidator
from .exceptions import (
    PlanRejectedError,
    ReviewValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_logger = logging.getLogger("plan_and_execute_research_assistant.review")


class HumanPlanReviewGateway:
    """Gateway orchestrating human operator plan inspection, approval, modification, and rejection."""

    def __init__(
        self,
        config: AssistantConfig | None = None,
        validator: PlanValidator | None = None,
    ) -> None:
        """Initialize review gateway with configuration and DAG validator."""
        self.config = config or get_assistant_config()
        self.validator = validator or PlanValidator()

    def format_plan_for_review(self, plan: Plan) -> str:
        """Format a plan into a structured, human-readable review representation."""
        lines: list[str] = [
            f"=== Proposed Plan (Version {plan.version}, Created By: {plan.created_by}) ===",
            f"Goal: {plan.goal}",
        ]
        if plan.change_summary:
            lines.append(f"Change Summary: {plan.change_summary}")
        if plan.assumptions:
            lines.append("Assumptions:")
            for a in plan.assumptions:
                lines.append(f"  - {a}")

        lines.append(f"Total Steps: {len(plan.steps)}")
        lines.append("-" * 60)

        for step in plan.steps:
            deps_str = ", ".join(step.depends_on) if step.depends_on else "None (entry step)"
            tool_hint = step.tool_name_hint or "General Reasoning"
            lines.append(f"[{step.id}] {step.description}")
            lines.append(f"  Dependencies:    {deps_str}")
            lines.append(f"  Tool Hint:       {tool_hint}")
            lines.append(f"  Success Criteria: {step.success_criteria}")
            lines.append("")

        return "\n".join(lines)

    def create_review_request(
        self,
        plan: Plan,
        goal: str,
        user_id: str,
    ) -> PlanReviewRequest:
        """Create a pending review request for a proposed plan."""
        return PlanReviewRequest(
            plan=plan,
            goal=goal,
            user_id=user_id,
        )

    def process_decision(
        self,
        request: PlanReviewRequest,
        decision: PlanReviewDecision,
    ) -> Plan:
        """Process an operator review decision and return the approved or modified plan.

        Args:
            request: The pending review request.
            decision: Operator decision (APPROVE, MODIFY, or REJECT).

        Returns:
            Approved or modified validated Plan.

        Raises:
            PlanRejectedError: If the operator rejected the plan.
            ReviewValidationError: If the modified plan failed DAG validation.
            ValueError: If MODIFY action is specified without a modified_plan.
        """
        if decision.action == ReviewAction.APPROVE:
            safe_log(
                _logger,
                logging.INFO,
                f"Plan version {request.plan.version} approved by reviewer '{decision.reviewer_id}'",
                reviewer_id=decision.reviewer_id,
                plan_version=request.plan.version,
            )
            return request.plan

        if decision.action == ReviewAction.MODIFY:
            if decision.modified_plan is None:
                raise ValueError("ReviewAction.MODIFY requires 'modified_plan' to be provided.")

            # Validate the modified plan through PlanValidator
            try:
                self.validator.validate_plan(decision.modified_plan)
            except PlanValidationError as exc:
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Operator-modified plan failed DAG validation: {exc}",
                    reviewer_id=decision.reviewer_id,
                    error=str(exc),
                )
                raise ReviewValidationError(
                    f"Operator-modified plan failed DAG validation: {exc}",
                    validation_error=exc,
                ) from exc

            # Stamp new immutable version metadata
            new_version = request.plan.version + 1
            change_notes = decision.reason or f"Modified by operator {decision.reviewer_id}"

            stamped_plan = decision.modified_plan.model_copy(
                update={
                    "version": new_version,
                    "parent_version": request.plan.version,
                    "created_by": "human_reviewer",
                    "change_summary": change_notes,
                },
                deep=True,
            )

            safe_log(
                _logger,
                logging.INFO,
                f"Plan modified by reviewer '{decision.reviewer_id}': version {stamped_plan.version} (parent: {stamped_plan.parent_version})",
                reviewer_id=decision.reviewer_id,
                version=stamped_plan.version,
                parent_version=stamped_plan.parent_version,
            )
            return stamped_plan

        if decision.action == ReviewAction.REJECT:
            reason = decision.reason or "No specific reason provided."
            safe_log(
                _logger,
                logging.WARNING,
                f"Plan version {request.plan.version} rejected by reviewer '{decision.reviewer_id}': {reason}",
                reviewer_id=decision.reviewer_id,
                reason=reason,
            )
            raise PlanRejectedError(
                message=f"Plan rejected by reviewer '{decision.reviewer_id}': {reason}",
                reviewer_id=decision.reviewer_id,
                reason=reason,
            )

        raise ValueError(f"Unknown ReviewAction: {decision.action}")

    async def _default_cli_review_prompt(self, request: PlanReviewRequest) -> PlanReviewDecision:
        """Interactive CLI prompt presenting the plan and reading operator decision from stdin."""
        formatted_plan = self.format_plan_for_review(request.plan)
        print("\n" + "=" * 60)
        print(f"Goal: {request.goal}")
        print(f"User ID: {request.user_id}")
        print("=" * 60)
        print(formatted_plan)
        print("=" * 60)

        def _read_stdin() -> PlanReviewDecision:
            try:
                choice = input("Enter decision ([a]pprove / [r]eject): ").strip().lower()
            except (EOFError, OSError):
                return PlanReviewDecision(
                    action=ReviewAction.REJECT,
                    reviewer_id="cli_operator",
                    reason="Non-interactive stdin or EOF encountered during CLI review.",
                )

            if choice in ("a", "approve", "y", "yes", ""):
                return PlanReviewDecision(
                    action=ReviewAction.APPROVE,
                    reviewer_id="cli_operator",
                )
            if choice in ("r", "reject", "n", "no"):
                try:
                    reason = input("Enter rejection reason: ").strip()
                except (EOFError, OSError):
                    reason = "Rejected by operator"
                return PlanReviewDecision(
                    action=ReviewAction.REJECT,
                    reviewer_id="cli_operator",
                    reason=reason or "Rejected by CLI operator",
                )
            return PlanReviewDecision(
                action=ReviewAction.REJECT,
                reviewer_id="cli_operator",
                reason=f"Unrecognized decision option '{choice}' provided via CLI prompt.",
            )

        return await asyncio.to_thread(_read_stdin)

    async def review_plan(
        self,
        plan: Plan,
        goal: str,
        user_id: str,
        decision_provider: (
            Callable[[PlanReviewRequest], Awaitable[PlanReviewDecision] | PlanReviewDecision] | None
        ) = None,
    ) -> Plan:
        """Review a plan, auto-approving if configured or delegating to decision_provider.

        Args:
            plan: The generated Plan to review.
            goal: Original user goal.
            user_id: User identifier.
            decision_provider: Async or sync callable that inspects request and returns decision.

        Returns:
            Approved or modified validated Plan.
        """
        # Config-based review bypass (for automated testing, evaluation benchmarks, and CI)
        if self.config.auto_approve_review:
            safe_log(
                _logger,
                logging.INFO,
                f"Auto-approving plan version {plan.version} (auto_approve_review=True)",
                plan_version=plan.version,
            )
            return plan

        if decision_provider is None:
            if sys.stdin and sys.stdin.isatty():
                decision_provider = self._default_cli_review_prompt
            else:
                raise ValueError(
                    "A decision_provider callable is required when auto_approve_review is False in a non-interactive environment."
                )

        request = self.create_review_request(plan=plan, goal=goal, user_id=user_id)

        # Invoke provider (supporting both async and synchronous callables)
        decision_result: Any = decision_provider(request)
        if inspect.isawaitable(decision_result):
            decision = await decision_result
        else:
            decision = decision_result

        if not isinstance(decision, PlanReviewDecision):
            raise TypeError(
                f"decision_provider must return a PlanReviewDecision, got {type(decision).__name__}"
            )

        return self.process_decision(request=request, decision=decision)
