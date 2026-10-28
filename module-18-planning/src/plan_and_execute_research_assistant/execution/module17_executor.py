"""Step executor wrapping Module 17 BoundedAgent."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from genai_core import ObservabilityEvent, safe_log
from pydantic import BaseModel, Field

from ..core import AssistantConfig, get_assistant_config, truncate_to_tokens
from ..models import PlanStep, StepStatus
from ..protocols.specialist import StepExecutor

if TYPE_CHECKING:
    from research_and_support_agent.agent import BoundedAgent

_logger = logging.getLogger("plan_and_execute_research_assistant.execution")


class StepExecutionOutput(BaseModel):
    """Output returned by StepExecutor for an individual step execution attempt."""

    step_id: str = Field(description="Step identifier")
    output_text: str = Field(description="Final output or deliverable text produced by the agent")
    status: StepStatus = Field(description="Execution lifecycle status")
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed during execution")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Cost incurred in USD")
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Wall-clock duration in milliseconds"
    )
    error_message: str | None = Field(default=None, description="Error detail if execution failed")


class Module17StepExecutor(StepExecutor):
    """StepExecutor implementation wrapping Module 17 BoundedAgent."""

    def __init__(
        self,
        agent: BoundedAgent | Any | None = None,
        config: AssistantConfig | None = None,
    ) -> None:
        """Initialize with BoundedAgent instance and configuration."""
        self.agent = agent
        self.config = config or get_assistant_config()

    def get_available_capabilities(self) -> dict[str, str]:
        """Dynamically extract available tool names and descriptions from the agent's ToolRegistry."""
        if self.agent is None:
            return {}

        registry = getattr(self.agent, "tool_registry", None)
        if registry is None:
            return {}

        caps: dict[str, str] = {}
        list_tools_fn = getattr(registry, "list_tools", None)
        if callable(list_tools_fn):
            tools = list_tools_fn()
            caps = {getattr(t, "name", str(t)): getattr(t, "description", "") for t in tools}
        else:
            tools_dict = getattr(registry, "tools", None)
            if isinstance(tools_dict, dict):
                caps = {name: getattr(tool, "description", "") for name, tool in tools_dict.items()}

        if not self.config.enable_episodic_similarity_search:
            caps = {name: desc for name, desc in caps.items() if "episodic" not in name.lower()}

        return caps

    def format_step_prompt(
        self,
        step: PlanStep,
        predecessor_context: dict[str, str],
        user_id: str | None = None,
        retry_feedback: str | None = None,
    ) -> str:
        """Assemble structured prompt for the step execution."""
        lines: list[str] = [
            f"STEP IDENTIFIER: {step.id}",
            f"OBJECTIVE: {step.description}",
            f"SUCCESS CRITERIA: {step.success_criteria}",
        ]

        if user_id:
            lines.append(f"TARGET USER / ENTITY CONTEXT: {user_id}")

        if step.tool_name_hint:
            lines.append(f"RECOMMENDED TOOL HINT: {step.tool_name_hint}")

        if predecessor_context:
            lines.append("\nPREDECESSOR CONTEXT (Completed Dependency Outputs):")
            for dep_id, summary in predecessor_context.items():
                lines.append(f"- [{dep_id}]: {summary}")

        if retry_feedback:
            lines.append(
                f"\nCRITICAL VERIFICATION FEEDBACK FROM PREVIOUS ATTEMPT:\n{retry_feedback}"
            )
            lines.append(
                "Please address the missing criteria and ensure all requirements are satisfied."
            )

        return "\n".join(lines)

    async def execute_step(
        self,
        step: PlanStep,
        predecessor_context: dict[str, str],
        user_id: str,
        retry_feedback: str | None = None,
    ) -> StepExecutionOutput:
        """Execute an individual plan step via the underlying BoundedAgent.

        Args:
            step: PlanStep to execute.
            predecessor_context: Mapping of predecessor step ID to its verified result summary.
            user_id: User identifier for authorization and memory.
            retry_feedback: Diagnostic feedback from previous failed verification if retrying.

        Returns:
            StepExecutionOutput with text, status, tokens, cost, and duration.
        """
        if self.agent is None:
            raise RuntimeError("Module17StepExecutor agent is not initialized.")

        query = self.format_step_prompt(
            step=step,
            predecessor_context=predecessor_context,
            user_id=user_id,
            retry_feedback=retry_feedback,
        )

        start_time = time.monotonic()
        safe_log(
            _logger,
            logging.INFO,
            f"Dispatching step '{step.id}' to BoundedAgent (hint: {step.tool_name_hint or 'none'})",
            event=ObservabilityEvent.TOOL_LOOP_START,
            step_id=step.id,
            user_id=user_id,
        )

        try:
            result = await self.agent.run(query=query, user_id=user_id)
            duration_ms = round((time.monotonic() - start_time) * 1000.0, 2)

            # Extract fields from AgentRunResult
            status_val = getattr(result, "status", None)
            status_str = getattr(status_val, "value", str(status_val)).lower()
            is_success = "completed" in status_str or "success" in status_str

            output_text = getattr(result, "final_text", None) or ""
            tokens_used = getattr(result, "total_tokens", 0) or 0
            cost_usd = getattr(result, "total_cost_usd", 0.0) or 0.0
            error_msg = getattr(result, "error_message", None)

            # Enforce 5K token limit on web-retrieved data / response using tiktoken
            is_web_step = (
                step.tool_name_hint == "search_web"
                or "search_web" in str(step.description).lower()
                or "web" in str(step.description).lower()
            )
            if is_web_step and output_text:
                output_text, _ = truncate_to_tokens(
                    text=output_text,
                    max_tokens=self.config.max_web_retrieved_tokens,
                    suffix=f"\n... [Web-retrieved response truncated to {self.config.max_web_retrieved_tokens} tokens limit]",
                )

            step_status = StepStatus.COMPLETED if is_success else StepStatus.FAILED
            if not is_success and not output_text and error_msg:
                output_text = f"Execution failed: {error_msg}"

            return StepExecutionOutput(
                step_id=step.id,
                output_text=output_text,
                status=step_status,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                error_message=error_msg,
            )

        except Exception as exc:  # noqa: BLE001
            duration_ms = round((time.monotonic() - start_time) * 1000.0, 2)
            safe_log(
                _logger,
                logging.ERROR,
                f"Step '{step.id}' execution threw unhandled exception: {exc}",
                step_id=step.id,
                error=str(exc),
            )
            return StepExecutionOutput(
                step_id=step.id,
                output_text="",
                status=StepStatus.FAILED,
                duration_ms=duration_ms,
                error_message=str(exc),
            )
