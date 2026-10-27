"""Dual-gate step verifier combining programmatic assertions and LLM judge evaluation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from genai_core import ObservabilityEvent, safe_log

from ..core import AssistantConfig, get_assistant_config
from ..models import PlanStep, StepJudgeVerdict, StepStatus, StepVerificationResult
from .prompts import build_judge_prompt

if TYPE_CHECKING:
    from genai_core import LLMService

_logger = logging.getLogger("plan_and_execute_research_assistant.verification")


class StepVerifier:
    """Evaluates step deliverables using a dual-gate architecture:

    Gate 1 (Programmatic): Status check, non-empty text, minimum length (>= 20 chars).
    Gate 2 (Model-Based Judge): Semantic criteria satisfaction check using LLMService (threshold >= 0.85).
    """

    def __init__(
        self,
        llm_service: LLMService,
        config: AssistantConfig | None = None,
    ) -> None:
        """Initialize verifier with LLMService and configuration."""
        self.llm_service = llm_service
        self.config = config or get_assistant_config()

    def verify_programmatic(
        self,
        step: PlanStep,
        output_text: str,
        execution_status: StepStatus,
    ) -> tuple[bool, str]:
        """Perform fast, deterministic assertions on the step outcome.

        Returns:
            Tuple of (passed: bool, message: str).
        """
        if execution_status != StepStatus.COMPLETED:
            return (
                False,
                f"Step execution failed with status '{execution_status.value}'",
            )

        trimmed = output_text.strip()
        if not trimmed:
            return False, "Agent deliverable output is empty."

        if len(trimmed) < 20:
            return (
                False,
                f"Agent deliverable output is too short ({len(trimmed)} chars; minimum is 20 chars).",
            )

        return True, "Programmatic verification checks passed."

    async def verify_model(
        self,
        step: PlanStep,
        output_text: str,
    ) -> StepJudgeVerdict:
        """Evaluate semantic criteria satisfaction using the LLM judge."""
        prompt = build_judge_prompt(
            step_description=step.description,
            success_criteria=step.success_criteria,
            output_text=output_text,
            tool_hint=step.tool_name_hint,
        )

        try:
            response = await self.llm_service.complete_structure(
                prompt=prompt,
                response_schema=StepJudgeVerdict,
                provider=self.config.default_provider,
            )
            # Unwrap LLMManagerResponse or raw response
            structured = getattr(response, "structured_response", None)
            if structured is None and hasattr(response, "content"):
                structured = response.content

            candidate = structured if structured is not None else response

            if isinstance(candidate, StepJudgeVerdict):
                return candidate
            if isinstance(candidate, dict):
                return StepJudgeVerdict.model_validate(candidate)
            if hasattr(response, "text") and response.text:
                return StepJudgeVerdict.model_validate_json(response.text)

            raise TypeError(f"Unexpected response type from LLMService: {type(response).__name__}")

        except Exception as exc:  # noqa: BLE001
            safe_log(
                _logger,
                logging.ERROR,
                f"Model judge verification failed for step '{step.id}': {exc}",
                event=ObservabilityEvent.LLM_REQUEST_FAILED,
                step_id=step.id,
                error=str(exc),
            )
            return StepJudgeVerdict(
                is_satisfied=False,
                score=0.0,
                reasoning=f"Model judge evaluation failed with error: {exc}",
                missing_criteria=["Model judge evaluation failed to produce a structured verdict."],
            )

    async def verify_step(
        self,
        step: PlanStep,
        output_text: str,
        execution_status: StepStatus,
    ) -> StepVerificationResult:
        """Run both programmatic and model-based verification gates.

        Args:
            step: PlanStep under evaluation.
            output_text: Text produced by executor.
            execution_status: Status of the execution attempt.

        Returns:
            StepVerificationResult with combined pass/fail status, scores, and diagnosis.
        """
        # Gate 1: Programmatic checks
        prog_passed, prog_msg = self.verify_programmatic(
            step=step,
            output_text=output_text,
            execution_status=execution_status,
        )

        if not prog_passed:
            safe_log(
                _logger,
                logging.WARNING,
                f"Step '{step.id}' failed programmatic verification: {prog_msg}",
                step_id=step.id,
                reason=prog_msg,
            )
            return StepVerificationResult(
                step_id=step.id,
                passed=False,
                programmatic_passed=False,
                model_passed=False,
                score=0.0,
                reasoning=prog_msg,
                missing_criteria=[prog_msg],
            )

        # Gate 2: Model-based semantic judge
        verdict = await self.verify_model(step=step, output_text=output_text)

        threshold = self.config.verification_threshold
        model_passed = verdict.is_satisfied and (verdict.score >= threshold)
        overall_passed = prog_passed and model_passed

        log_level = logging.INFO if overall_passed else logging.WARNING
        safe_log(
            _logger,
            log_level,
            f"Step '{step.id}' verification completed: passed={overall_passed} (score={verdict.score:.2f}/{threshold:.2f})",
            step_id=step.id,
            passed=overall_passed,
            score=verdict.score,
            is_satisfied=verdict.is_satisfied,
        )

        return StepVerificationResult(
            step_id=step.id,
            passed=overall_passed,
            programmatic_passed=True,
            model_passed=model_passed,
            score=verdict.score,
            reasoning=verdict.reasoning,
            missing_criteria=verdict.missing_criteria,
        )
