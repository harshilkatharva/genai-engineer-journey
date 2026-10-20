from __future__ import annotations

import time

from pydantic import BaseModel

from ..core.settings import get_settings
from ..models.llm_model.llm_response_model import ChatMessage, ToolCall
from ..models.tools_model.tool_spec import ToolChoice, ToolSpec
from ..observability import ObservabilityEvent, get_logger, safe_log
from ..services.llm_services import LLMService
from .executor import ToolExecutor
from .guards import LoopDetector
from .models import (
    ApprovalDecision,
    ApprovalRequest,
    ToolBudget,
    ToolLoopResult,
    ToolLoopStatus,
)

_logger = get_logger("genai_core.tools.loop")


class ToolLoopRunner:
    """Bounded, multi-turn orchestration engine coordinating LLM and tool interactions."""

    def __init__(
        self,
        llm_service: LLMService,
        executor: ToolExecutor,
        max_iterations: int | None = None,
        default_timeout_s: float | None = None,
        max_repeated_calls: int | None = None,
    ) -> None:
        settings = get_settings()
        self.llm_service = llm_service
        self.executor = executor
        self.max_iterations = (
            max_iterations if max_iterations is not None else settings.tool_max_iterations
        )
        self.default_timeout_s = default_timeout_s
        self.max_repeated_calls = (
            max_repeated_calls
            if max_repeated_calls is not None
            else settings.tool_max_repeated_calls
        )

    async def run(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
        provider: str | None = None,
        budget: ToolBudget | None = None,
        timeout_s: float | None = None,
        initial_iterations: int = 0,
        initial_tool_calls: int = 0,
    ) -> ToolLoopResult:
        """Run the bounded tool loop until completion, budget exhaustion, or approval pause."""
        effective_timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        active_budget = budget or ToolBudget(
            max_steps=self.max_iterations,
            deadline_monotonic=(
                time.monotonic() + effective_timeout if effective_timeout is not None else None
            ),
        )
        if (
            budget is not None
            and effective_timeout is not None
            and active_budget.deadline_monotonic is None
        ):
            active_budget.deadline_monotonic = time.monotonic() + effective_timeout

        current_messages = list(messages)
        active_specs = tools if tools is not None else self.executor.registry.get_specs()
        detector = LoopDetector(max_repeated_calls=self.max_repeated_calls)

        safe_log(
            _logger,
            20,  # INFO
            "Tool loop initiated",
            event=ObservabilityEvent.TOOL_LOOP_START,
            initial_message_count=len(current_messages),
            max_steps=active_budget.max_steps,
        )

        iteration = initial_iterations
        total_tool_calls = initial_tool_calls

        while active_budget.can_step():
            iteration += 1
            safe_log(
                _logger,
                20,  # INFO
                f"Starting tool loop step {iteration}",
                event=ObservabilityEvent.TOOL_LOOP_STEP,
                iteration=iteration,
            )

            if active_budget.is_deadline_exceeded():
                return ToolLoopResult(
                    status=ToolLoopStatus.DEADLINE_EXCEEDED,
                    messages=current_messages,
                    iterations=iteration,
                    total_tool_calls=total_tool_calls,
                    error_message="Execution deadline exceeded before model response",
                )

            try:
                response = await self.llm_service.complete_tool(
                    messages=current_messages,
                    tools=active_specs,
                    tool_choice=tool_choice,
                    provider=provider,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"LLM completion error: {type(exc).__name__}: {exc}"
                safe_log(
                    _logger,
                    40,  # ERROR
                    msg,
                    event=ObservabilityEvent.TOOL_LOOP_COMPLETED,
                    status=ToolLoopStatus.FAILED,
                    iteration=iteration,
                )
                return ToolLoopResult(
                    status=ToolLoopStatus.FAILED,
                    messages=current_messages,
                    iterations=iteration,
                    total_tool_calls=total_tool_calls,
                    error_message=msg,
                )

            input_tokens = response.usage.get("input_tokens", 0) if response.usage else 0
            output_tokens = response.usage.get("output_tokens", 0) if response.usage else 0
            active_budget.record_step(tokens=(input_tokens + output_tokens))

            # Case 1: Model finished without calling any tools
            if not response.tool_calls:
                if response.text is not None:
                    current_messages.append(ChatMessage(role="assistant", content=response.text))

                safe_log(
                    _logger,
                    20,  # INFO
                    "Tool loop completed successfully",
                    event=ObservabilityEvent.TOOL_LOOP_COMPLETED,
                    status=ToolLoopStatus.COMPLETED,
                    iterations=iteration,
                    total_tool_calls=total_tool_calls,
                )
                return ToolLoopResult(
                    status=ToolLoopStatus.COMPLETED,
                    final_text=response.text,
                    messages=current_messages,
                    iterations=iteration,
                    total_tool_calls=total_tool_calls,
                )

            # Case 2: Model emitted tool calls
            current_messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.text or "",
                )
            )

            for call in response.tool_calls:
                total_tool_calls += 1

                # Loop detection guard
                if detector.record_and_check(call.name, call.arguments):
                    safe_log(
                        _logger,
                        30,  # WARNING
                        f"Infinite loop detected for tool '{call.name}'",
                        event=ObservabilityEvent.TOOL_LOOP_DETECTED,
                        tool_name=call.name,
                    )
                    loop_msg = ChatMessage(
                        role="tool",
                        content=(
                            f"Error: Loop detected. Repeated identical call to '{call.name}'. "
                            "Please analyze previous tool outputs, revise your approach, or provide your final response."
                        ),
                        tool_call_id=call.id,
                        tool_name=call.name,
                    )
                    current_messages.append(loop_msg)
                    continue

                # Approval policy guard
                tool_instance = self.executor.registry.get(call.name)
                if (
                    tool_instance is not None
                    and tool_instance.risk_level in self.executor.require_approval_for
                ):
                    spec = tool_instance.to_spec()
                    parsed_args, _ = spec.parse_arguments(call.arguments)
                    args_dict = (
                        parsed_args.model_dump()
                        if isinstance(parsed_args, BaseModel)
                        else (parsed_args or {})
                    )
                    approved, approval_req = await self.executor._check_approval(
                        call, tool_instance, args_dict
                    )
                    if not approved and approval_req is not None:
                        # Pause loop for external / human authorization
                        return ToolLoopResult(
                            status=ToolLoopStatus.APPROVAL_REQUIRED,
                            messages=current_messages,
                            iterations=iteration,
                            total_tool_calls=total_tool_calls,
                            pending_approval=approval_req,
                        )

                # Tool Execution
                result = await self.executor.execute(call)
                current_messages.append(result.to_chat_message())

            # Budget checks after step completion
            if active_budget.is_deadline_exceeded():
                return ToolLoopResult(
                    status=ToolLoopStatus.DEADLINE_EXCEEDED,
                    messages=current_messages,
                    iterations=iteration,
                    total_tool_calls=total_tool_calls,
                    error_message="Execution deadline exceeded after tool execution",
                )

            if active_budget.is_token_budget_exhausted():
                return ToolLoopResult(
                    status=ToolLoopStatus.BUDGET_EXCEEDED,
                    messages=current_messages,
                    iterations=iteration,
                    total_tool_calls=total_tool_calls,
                    error_message="Token budget exhausted",
                )

            if active_budget.is_cost_budget_exhausted():
                return ToolLoopResult(
                    status=ToolLoopStatus.BUDGET_EXCEEDED,
                    messages=current_messages,
                    iterations=iteration,
                    total_tool_calls=total_tool_calls,
                    error_message="Cost budget exhausted",
                )

        # Reached when budget.can_step() is False
        terminal_status = (
            ToolLoopStatus.DEADLINE_EXCEEDED
            if active_budget.is_deadline_exceeded()
            else (
                ToolLoopStatus.BUDGET_EXCEEDED
                if (
                    active_budget.is_token_budget_exhausted()
                    or active_budget.is_cost_budget_exhausted()
                )
                else ToolLoopStatus.MAX_ITERATIONS_REACHED
            )
        )

        return ToolLoopResult(
            status=terminal_status,
            messages=current_messages,
            iterations=iteration,
            total_tool_calls=total_tool_calls,
            error_message=f"Terminated due to {terminal_status.value}",
        )

    async def resume(
        self,
        decision: ApprovalDecision,
        request: ApprovalRequest,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
        provider: str | None = None,
        budget: ToolBudget | None = None,
        timeout_s: float | None = None,
        initial_iterations: int = 0,
        initial_tool_calls: int = 0,
    ) -> ToolLoopResult:
        """Resume execution of a paused tool loop after human or caller authorization decision."""
        if self.executor.approval_store is not None:
            await self.executor.approval_store.record_decision(request.request_id, decision)

        current_messages = list(messages)

        safe_log(
            _logger,
            20,  # INFO
            f"Resuming tool loop for request {request.request_id}",
            event=ObservabilityEvent.TOOL_APPROVAL_DECISION,
            request_id=request.request_id,
            approved=decision.approved,
        )

        if decision.approved:
            effective_arguments = (
                decision.modified_arguments
                if decision.modified_arguments is not None
                else request.arguments
            )
            approved_call = ToolCall(
                id=request.call_id,
                name=request.tool_name,
                arguments=effective_arguments,
            )
            result = await self.executor.execute(approved_call, bypass_approval=True)
            current_messages.append(result.to_chat_message())
        else:
            reason = decision.reason or "Tool operation rejected by operator."
            rejection_message = ChatMessage(
                role="tool",
                content=f"Error: {reason}",
                tool_call_id=request.call_id,
                tool_name=request.tool_name,
            )
            current_messages.append(rejection_message)

        # Continue loop with remaining steps
        return await self.run(
            messages=current_messages,
            tools=tools,
            tool_choice=tool_choice,
            provider=provider,
            budget=budget,
            timeout_s=timeout_s,
            initial_iterations=initial_iterations,
            initial_tool_calls=initial_tool_calls,
        )
