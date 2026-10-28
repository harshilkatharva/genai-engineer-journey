"""Plan generator coordinating structured decomposition, memory recall, and repair."""

from __future__ import annotations

import logging
from typing import Any

from genai_core import (
    LLMService,
    ObservabilityEvent,
    get_logger,
    safe_log,
)

from ..core.config import AssistantConfig, get_assistant_config
from ..memory.manager import PlanningMemoryManager
from ..models.budget import RunBudget
from ..models.plan import Plan
from ..protocols.specialist import StepExecutor
from .exceptions import PlanGenerationError, PlanValidationError
from .prompts import build_planning_prompt
from .repairer import PlanRepairer
from .validator import PlanValidator

_logger = get_logger("plan_and_execute_research_assistant.planning.generator")


class PlanGenerator:
    """Generates structured DAG execution plans with automated repair and memory integration."""

    def __init__(
        self,
        llm_service: LLMService,
        validator: PlanValidator | None = None,
        repairer: PlanRepairer | None = None,
        config: AssistantConfig | None = None,
        memory_manager: PlanningMemoryManager | None = None,
        step_executor: StepExecutor | None = None,
        available_capabilities: dict[str, str] | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.validator = validator or PlanValidator()
        self.config = config or get_assistant_config()
        self.memory_manager = memory_manager
        self.step_executor = step_executor
        self._capabilities = available_capabilities or {}

        # Resolve capabilities from executor if available
        capabilities = self.get_capabilities()
        self.repairer = repairer or PlanRepairer(
            llm_service=self.llm_service,
            validator=self.validator,
            config=self.config,
            available_capabilities=capabilities,
        )

    def get_capabilities(self) -> dict[str, str]:
        """Dynamically retrieve available capabilities from the executor or explicit config."""
        if self.step_executor is not None:
            return self.step_executor.get_available_capabilities()
        return self._capabilities

    async def generate_plan(
        self,
        goal: str,
        user_id: str = "default_user",
        user_preferences: list[str] | None = None,
        past_plans: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        budget: RunBudget | None = None,
    ) -> Plan:
        """Decompose a high-level goal into a validated Directed Acyclic Graph (DAG) Plan.

        Args:
            goal: The objective to plan for.
            user_id: User identifier for memory isolation and recall.
            user_preferences: Optional list of user preferences (auto-recalled if None).
            past_plans: Optional list of past plan blueprints (auto-recalled if None).
            context: Additional runtime inputs or constraints.

        Returns:
            A verified, structurally valid Plan instance.

        Raises:
            PlanGenerationError: If plan generation fails or exhausts repair retries.
        """
        if not goal or not goal.strip():
            raise PlanGenerationError("Goal cannot be empty.")

        clean_goal = goal.strip()
        capabilities = self.get_capabilities()

        # 1. Recall memory context if not explicitly passed
        prefs = user_preferences
        if prefs is None and self.memory_manager:
            prefs = await self.memory_manager.get_user_preferences(
                user_id=user_id, query=clean_goal
            )

        past = past_plans
        if past is None and self.memory_manager:
            past = await self.memory_manager.get_similar_past_plans(
                user_id=user_id, goal=clean_goal
            )

        safe_log(
            _logger,
            logging.INFO,
            f"Generating plan for user '{user_id}': '{clean_goal}'",
            event=ObservabilityEvent.LLM_REQUEST_START,
            user_id=user_id,
            goal=clean_goal,
            has_preferences=bool(prefs),
            has_past_plans=bool(past),
        )

        # 2. Build structured prompt
        prompt = build_planning_prompt(
            goal=clean_goal,
            available_capabilities=capabilities,
            user_preferences=prefs,
            past_plans=past,
            context=context,
        )

        # 3. Call LLM for structured output
        try:
            response = await self.llm_service.complete_structure(
                prompt=prompt,
                response_schema=Plan,
                provider=self.config.default_provider,
            )
        except Exception as exc:
            safe_log(
                _logger,
                logging.ERROR,
                f"LLM complete_structure failed: {exc}",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
                error=str(exc),
            )
            raise PlanGenerationError(f"LLM plan generation failed: {exc}") from exc

        if response.error:
            raise PlanGenerationError(f"LLM service returned error: {response.error.message}")

        # Record planning usage if budget provided
        if budget is not None:
            usage = getattr(response, "usage", {}) or {}
            if isinstance(usage, dict):
                tokens = int(usage.get("total_tokens", 0)) or (
                    int(usage.get("output_tokens", 0)) + int(usage.get("input_tokens", 0))
                )
            else:
                tokens = getattr(usage, "total_tokens", 0)
            cost = float(getattr(response, "cost_usd", 0.0) or 0.0)
            budget.record_usage(tokens=tokens, cost_usd=cost)

        # Extract structured Plan
        candidate: Plan | None = None
        if isinstance(response.structured_response, Plan):
            candidate = response.structured_response
        elif isinstance(response.structured_response, dict):
            candidate = Plan.model_validate(response.structured_response)
        elif response.text:
            candidate = Plan.model_validate_json(response.text)

        if candidate is None:
            raise PlanGenerationError("LLM returned empty structured response for plan.")

        # 4. Validate candidate DAG
        try:
            self.validator.validate(candidate)
            safe_log(
                _logger,
                logging.INFO,
                f"Plan generated and validated successfully (version={candidate.version}, steps={len(candidate.steps)})",
                event=ObservabilityEvent.LLM_REQUEST_SUCCESS,
                goal=clean_goal,
                step_count=len(candidate.steps),
            )
            return candidate
        except PlanValidationError as val_exc:
            safe_log(
                _logger,
                logging.WARNING,
                f"Initial plan generation failed DAG validation: {val_exc}. Triggering repair loop.",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
                error=str(val_exc),
            )
            # Delegate to automated repair loop
            return await self.repairer.repair(
                goal=clean_goal,
                invalid_plan=candidate,
                initial_error=val_exc,
                budget=budget,
            )
