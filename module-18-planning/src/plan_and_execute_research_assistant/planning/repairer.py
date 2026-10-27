"""Automated plan repair loop for correcting structural DAG validation errors."""

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
from ..models.plan import Plan
from .exceptions import PlanGenerationError, PlanValidationError
from .prompts import build_repair_prompt
from .validator import PlanValidator

_logger = get_logger("plan_and_execute_research_assistant.planning.repairer")


class PlanRepairer:
    """Repairs invalid plans by feeding structural DAG critiques back to LLMService."""

    def __init__(
        self,
        llm_service: LLMService,
        validator: PlanValidator | None = None,
        config: AssistantConfig | None = None,
        available_capabilities: dict[str, str] | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.validator = validator or PlanValidator()
        self.config = config or get_assistant_config()
        self.available_capabilities = available_capabilities or {}

    async def repair(
        self,
        goal: str,
        invalid_plan: Plan | dict[str, Any] | str,
        initial_error: Exception,
        max_retries: int | None = None,
    ) -> Plan:
        """Attempt to repair an invalid plan by querying the LLM with the validation error.

        Args:
            goal: Original goal of the plan.
            invalid_plan: The plan instance or raw output that failed validation.
            initial_error: The PlanValidationError that was raised.
            max_retries: Maximum repair attempts (defaults to config).

        Returns:
            A structurally valid Plan instance.

        Raises:
            PlanGenerationError: If repair attempts are exhausted without generating a valid DAG.
        """
        retries = max_retries if max_retries is not None else self.config.max_planning_retries
        current_error: Exception = initial_error
        current_invalid: Plan | dict[str, Any] | str = invalid_plan

        safe_log(
            _logger,
            logging.WARNING,
            f"Initiating plan repair loop for goal: '{goal}'. Initial error: {initial_error}",
            event=ObservabilityEvent.LLM_REQUEST_START,
            goal=goal,
            error=str(initial_error),
            max_retries=retries,
        )

        for attempt in range(1, retries + 1):
            repair_prompt = build_repair_prompt(
                goal=goal,
                invalid_plan=current_invalid,
                validation_error=str(current_error),
                available_capabilities=self.available_capabilities,
            )

            try:
                response = await self.llm_service.complete_structure(
                    prompt=repair_prompt,
                    response_schema=Plan,
                    provider=self.config.default_provider,
                )
            except Exception as exc:  # noqa: BLE001
                current_error = exc
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Plan repair attempt {attempt}/{retries} LLM call failed: {exc}",
                    event=ObservabilityEvent.LLM_REQUEST_FAILED,
                    attempt=attempt,
                    error=str(exc),
                )
                continue

            if response.error:
                current_error = PlanGenerationError(response.error.message)
                continue

            # Extract structured plan from response
            candidate: Plan | None = None
            if isinstance(response.structured_response, Plan):
                candidate = response.structured_response
            elif isinstance(response.structured_response, dict):
                candidate = Plan.model_validate(response.structured_response)
            elif response.text:
                candidate = Plan.model_validate_json(response.text)

            if candidate is None:
                current_error = PlanGenerationError(
                    "LLM returned empty structured response during repair."
                )
                continue

            # Validate structural integrity of repaired candidate
            try:
                self.validator.validate(candidate)
                candidate.change_summary = (
                    f"Repaired structural validation error on attempt {attempt}: {initial_error}"
                )
                safe_log(
                    _logger,
                    logging.INFO,
                    f"Plan successfully repaired on attempt {attempt}/{retries}",
                    event=ObservabilityEvent.LLM_REQUEST_SUCCESS,
                    attempt=attempt,
                )
                return candidate
            except PlanValidationError as val_exc:
                current_invalid = candidate
                current_error = val_exc
                safe_log(
                    _logger,
                    logging.WARNING,
                    f"Plan repair attempt {attempt}/{retries} produced invalid DAG: {val_exc}",
                    event=ObservabilityEvent.LLM_REQUEST_FAILED,
                    attempt=attempt,
                    error=str(val_exc),
                )

        raise PlanGenerationError(
            f"Failed to generate a valid DAG plan after {retries} repair retries. "
            f"Last validation error: {current_error}"
        )
