"""Dynamic replanning engine providing concurrency-safe failure recovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log
from pydantic import BaseModel, Field

from ..core import AssistantConfig, get_assistant_config
from ..models import Plan, PlanStep, RunBudget, StepStatus
from ..planning import PlanRepairer, PlanValidationError, PlanValidator
from .exceptions import ReplanningError, ReplanningLimitExceededError
from .prompts import build_replanning_prompt

if TYPE_CHECKING:
    from genai_core import LLMService

_logger = logging.getLogger("plan_and_execute_research_assistant.replanning")


class ReplanResult(BaseModel):
    """Structured result returned by ReplanningEngine after revising an execution plan."""

    new_plan: Plan = Field(description="The validated revised Plan DAG")
    replan_count: int = Field(description="Monotonically updated replan counter for this run")
    preserved_step_ids: list[str] = Field(
        default_factory=list, description="IDs of completed steps preserved from parent plan"
    )
    new_step_ids: list[str] = Field(
        default_factory=list, description="IDs of newly introduced or replacement steps"
    )


class ReplanningEngine:
    """Executes concurrency-safe dynamic replanning when a step fails verification.

    Enforces the Work Preservation Invariant: all COMPLETED steps remain immutable
    and are retained in the revised plan along with their cached outputs.
    Enforces replanning ceiling to prevent runaway thrashing.
    Validates and repairs the resulting revised DAG.
    """

    def __init__(
        self,
        llm_service: LLMService,
        validator: PlanValidator | None = None,
        repairer: PlanRepairer | None = None,
        config: AssistantConfig | None = None,
    ) -> None:
        """Initialize ReplanningEngine with LLMService, validator, repairer, and config."""
        self.llm_service = llm_service
        self.config = config or get_assistant_config()
        self.validator = validator or PlanValidator()
        self.repairer = repairer or PlanRepairer(
            llm_service=llm_service,
            validator=self.validator,
            config=self.config,
        )

    async def replan(
        self,
        current_plan: Plan,
        failed_step_id: str,
        failure_reason: str,
        step_results: dict[str, str],
        replan_count: int,
        capabilities: dict[str, str] | None = None,
        user_preferences: list[str] | None = None,
        budget: RunBudget | None = None,
    ) -> ReplanResult:
        """Dynamically revise an execution plan following a step failure.

        Args:
            current_plan: The active Plan DAG that experienced a failure.
            failed_step_id: The ID of the step that exhausted retries and failed.
            failure_reason: Critique or error detail explaining the failure.
            step_results: Mapping of completed step IDs to their verified deliverable outputs.
            replan_count: Number of replans already performed during this run.
            capabilities: Optional dictionary of available tool/agent capabilities.
            user_preferences: Optional list of user preference constraints.
            budget: Optional RunBudget for token/cost tracking and timeout enforcement.

        Returns:
            ReplanResult containing the new validated Plan and audit metadata.

        Raises:
            ReplanningLimitExceededError: If replan_count has reached or exceeded max_replans.
            ReplanningError: If replanning or DAG repair fails.
            BudgetExceededError: If the run budget is exhausted.
        """
        # 1. Enforce replanning ceiling
        if replan_count >= self.config.max_replans:
            safe_log(
                _logger,
                logging.ERROR,
                f"Replanning ceiling reached ({replan_count}/{self.config.max_replans}) for goal: '{current_plan.goal}'",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
                replan_count=replan_count,
                max_replans=self.config.max_replans,
            )
            raise ReplanningLimitExceededError(
                current_replans=replan_count,
                max_replans=self.config.max_replans,
                goal=current_plan.goal,
            )

        # 2. Enforce global budget before invoking LLM
        if budget is not None:
            budget.check_budget()

        safe_log(
            _logger,
            logging.INFO,
            f"Initiating dynamic replan {replan_count + 1}/{self.config.max_replans} for plan version {current_plan.version} after failure of step '{failed_step_id}'",
            event=ObservabilityEvent.LLM_REQUEST_START,
            failed_step_id=failed_step_id,
            replan_count=replan_count,
        )

        # 3. Categorize steps: Completed (Immutable), Failed, and Remaining
        step_map: dict[str, PlanStep] = {s.id: s for s in current_plan.steps}
        failed_step = step_map.get(
            failed_step_id,
            PlanStep(
                id=failed_step_id,
                description=f"Step {failed_step_id}",
                success_criteria="Criteria unfulfilled",
                status=StepStatus.FAILED,
            ),
        )

        completed_steps: list[PlanStep] = []
        for s in current_plan.steps:
            if s.id in step_results or s.status == StepStatus.COMPLETED:
                preserved = s.model_copy()
                preserved.status = StepStatus.COMPLETED
                preserved.result_summary = step_results.get(s.id, s.result_summary)
                completed_steps.append(preserved)

        completed_step_ids = {s.id for s in completed_steps}
        remaining_steps = [
            s
            for s in current_plan.steps
            if s.id not in completed_step_ids and s.id != failed_step_id
        ]

        # 4. Construct replanning prompt
        prompt = build_replanning_prompt(
            goal=current_plan.goal,
            completed_steps=completed_steps,
            failed_step=failed_step,
            failure_reason=failure_reason,
            remaining_steps=remaining_steps,
            capabilities=capabilities,
            user_preferences=user_preferences,
        )

        # 5. Invoke LLM to generate candidate revised plan
        try:
            raw_response = await self.llm_service.complete_structure(
                prompt=prompt,
                response_schema=Plan,
            )
        except Exception as exc:
            safe_log(
                _logger,
                logging.ERROR,
                f"LLM request error during replanning: {exc}",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
                error=str(exc),
            )
            raise ReplanningError(f"Replanning LLM invocation failed: {exc}") from exc

        # Extract structured Plan
        candidate_plan: Plan
        if isinstance(raw_response, Plan):
            candidate_plan = raw_response
        elif hasattr(raw_response, "structured_response") and isinstance(
            raw_response.structured_response, Plan
        ):
            candidate_plan = raw_response.structured_response
        elif hasattr(raw_response, "content") and isinstance(raw_response.content, Plan):
            candidate_plan = raw_response.content
        else:
            msg = (
                f"Unexpected response type from LLMService during replanning: {type(raw_response)}"
            )
            raise ReplanningError(msg)

        # Record usage if available
        if budget is not None:
            usage = getattr(raw_response, "usage", None)
            tokens = getattr(usage, "total_tokens", 0) if usage else 0
            cost = getattr(raw_response, "cost_usd", 0.0)
            budget.record_usage(tokens=tokens, cost_usd=cost)

        # 6. Programmatically Enforce Work Preservation Invariant
        # Ensure all completed steps are present with original IDs and status=COMPLETED
        candidate_steps_by_id: dict[str, PlanStep] = {s.id: s for s in candidate_plan.steps}
        final_steps: list[PlanStep] = []

        # Keep all completed steps first or in their original dependency order
        for comp_step in completed_steps:
            if comp_step.id in candidate_steps_by_id:
                # Merge: ensure status and cached summary are strictly preserved
                candidate_s = candidate_steps_by_id[comp_step.id]
                candidate_s.status = StepStatus.COMPLETED
                candidate_s.result_summary = comp_step.result_summary
                final_steps.append(candidate_s)
                del candidate_steps_by_id[comp_step.id]
            else:
                # Re-insert preserved completed step if omitted by LLM
                final_steps.append(comp_step)

        # Append all new / revised candidate steps
        for remaining_candidate in candidate_steps_by_id.values():
            # If the candidate step matches failed_step_id and status is FAILED, reset to PENDING
            if (
                remaining_candidate.id == failed_step_id
                and remaining_candidate.status == StepStatus.FAILED
            ):
                remaining_candidate.status = StepStatus.PENDING
            final_steps.append(remaining_candidate)

        candidate_plan.steps = final_steps

        # 7. Validate and repair DAG structure
        try:
            self.validator.validate(candidate_plan)
        except PlanValidationError as val_exc:
            safe_log(
                _logger,
                logging.WARNING,
                f"Candidate replanned DAG failed validation: {val_exc}. Triggering PlanRepairer...",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
                error=str(val_exc),
            )
            try:
                candidate_plan = await self.repairer.repair_plan(
                    invalid_plan=candidate_plan,
                    error=val_exc,
                    capabilities=capabilities,
                    user_preferences=user_preferences,
                    budget=budget,
                )
            except Exception as repair_exc:
                safe_log(
                    _logger,
                    logging.ERROR,
                    f"Plan repair failed during replanning: {repair_exc}",
                    event=ObservabilityEvent.LLM_REQUEST_FAILED,
                    error=str(repair_exc),
                )
                raise ReplanningError(
                    f"Replanned DAG validation and repair failed: {repair_exc}"
                ) from repair_exc

        # 8. Version metadata stamping
        candidate_plan.version = current_plan.version + 1
        candidate_plan.parent_version = current_plan.version
        candidate_plan.created_by = "replanner"
        candidate_plan.change_summary = (
            f"Replanned after failure of step '{failed_step_id}': {failure_reason}"
        )

        new_step_ids = [s.id for s in candidate_plan.steps if s.id not in completed_step_ids]

        safe_log(
            _logger,
            logging.INFO,
            f"Successfully generated revised plan version {candidate_plan.version} ({len(candidate_plan.steps)} total steps, {len(completed_step_ids)} preserved, {len(new_step_ids)} new)",
            event=ObservabilityEvent.LLM_REQUEST_SUCCESS,
            new_version=candidate_plan.version,
            parent_version=candidate_plan.parent_version,
            preserved_steps=len(completed_step_ids),
            new_steps=len(new_step_ids),
        )

        return ReplanResult(
            new_plan=candidate_plan,
            replan_count=replan_count + 1,
            preserved_step_ids=list(completed_step_ids),
            new_step_ids=new_step_ids,
        )
