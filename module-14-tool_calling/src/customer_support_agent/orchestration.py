from __future__ import annotations

import json

from pydantic import BaseModel

from customer_support_agent.models import ChatMessage, LLMResponseModel
from customer_support_agent.observability.events import EventName
from customer_support_agent.observability.logger import logger
from customer_support_agent.observability.tool_logger import ToolInvocationLogger
from customer_support_agent.providers.llm_provider import LLMProvider
from customer_support_agent.tools import ToolRegistry


class ToolChatResult(BaseModel):
    text: str
    messages: list[ChatMessage]
    iterations: int
    exhausted: bool = False
    error: str | None = None


async def run_tool_chat(
    messages: list[ChatMessage],
    provider: LLMProvider,
    registry: ToolRegistry,
    max_iterations: int = 5,
) -> ToolChatResult:
    """Run the model/tool loop with a hard bound and user-safe error messages."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    conversation = list(messages)
    tool_logger = ToolInvocationLogger()
    for iteration in range(1, max_iterations + 1):
        try:
            response: LLMResponseModel = await provider.complete(
                conversation, registry.definitions()
            )
        except Exception:  # noqa: BLE001 - provider SDK errors are intentionally normalized
            logger.exception(
                "LLM call failed",
                event=EventName.LLM_FAILED,
                component="orchestration",
                iteration=iteration,
            )
            return ToolChatResult(
                text="I'm sorry, but I'm unable to reach support services right now.",
                messages=conversation,
                iterations=iteration,
                error="model_error",
            )
        logger.info(
            "LLM call completed",
            event=EventName.LLM_COMPLETED,
            component="orchestration",
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            iteration=iteration,
        )
        if not response.tool_calls:
            return ToolChatResult(
                text=response.text or "I'm sorry, I couldn't generate a response.",
                messages=conversation,
                iterations=iteration,
            )
        conversation.append(
            ChatMessage(
                role="assistant",
                content=response.text or "",
                tool_call_id=response.tool_calls[0].id,
                tool_name=response.tool_calls[0].name,
            )
        )
        for call in response.tool_calls:
            started = tool_logger.start(call.name, call.arguments, iteration)
            try:
                value = await registry.execute(call.name, call.arguments)
                content = json.dumps(value, default=str)
            except (KeyError, TypeError, ValueError, RuntimeError):
                # Tool failures are returned to the model without exposing internals.
                # Do not expose database/provider details to the model or caller.
                content = json.dumps({"error": "The requested support tool failed."})
                tool_logger.fail(
                    call.name,
                    started,
                    iteration,
                    RuntimeError("tool execution failed"),
                )
            else:
                tool_logger.complete(call.name, started, iteration)
            conversation.append(
                ChatMessage(role="tool", content=content, tool_call_id=call.id, tool_name=call.name)
            )
    return ToolChatResult(
        text="I couldn't complete that request within the allowed number of steps.",
        messages=conversation,
        iterations=max_iterations,
        exhausted=True,
        error="max_iterations",
    )
