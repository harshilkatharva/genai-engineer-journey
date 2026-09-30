from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from ..models import (
    ChatMessage,
    LLMManagerRequest,
    ToolArgumentError,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from .llm_services import LLMService


class ToolExecutionResult(BaseModel):
    call_id: str
    tool_name: str
    result: Any = None
    error: ToolArgumentError | None = None


async def execute_tool_calls(
    calls: Sequence[ToolCall], tools: Sequence[ToolSpec]
) -> list[ToolExecutionResult]:
    tools_by_name = {tool.name: tool for tool in tools}

    async def execute_one(call: ToolCall) -> ToolExecutionResult:
        tool = tools_by_name.get(call.name)
        if tool is None:
            return ToolExecutionResult(
                call_id=call.id,
                tool_name=call.name,
                error=ToolArgumentError(
                    code="unknown_tool", message=f"No tool named '{call.name}' is registered."
                ),
            )

        arguments, error = tool.parse_arguments(call.arguments)
        if error:
            return ToolExecutionResult(call_id=call.id, tool_name=call.name, error=error)
        if tool.handler is None or arguments is None:
            return ToolExecutionResult(
                call_id=call.id,
                tool_name=call.name,
                error=ToolArgumentError(
                    code="missing_handler", message=f"Tool '{call.name}' has no handler."
                ),
            )

        try:
            result = tool.handler(arguments)
            if inspect.isawaitable(result):
                result = await result
            return ToolExecutionResult(call_id=call.id, tool_name=call.name, result=result)
        except Exception as error:  # noqa: BLE001 - tool handlers are arbitrary user code
            return ToolExecutionResult(
                call_id=call.id,
                tool_name=call.name,
                error=ToolArgumentError(
                    code="execution_error", message=f"Tool execution failed: {error}"
                ),
            )

    return list(await asyncio.gather(*(execute_one(call) for call in calls)))


async def extract_structured_output(
    service: LLMService,
    messages: list[ChatMessage],
    output_model: type[BaseModel],
    *,
    provider: str | None = None,
    tool_name: str = "emit_structured_output",
    description: str = "Return the requested structured output.",
) -> tuple[BaseModel | None, ToolArgumentError | None]:
    spec = ToolSpec(
        name=tool_name,
        description=description,
        argument_model=output_model,
    )
    response = await service.complete(
        LLMManagerRequest(
            provider=provider,
            messages=messages,
            tools=[spec],
            tool_choice=ToolChoice(mode="specific", tool_name=tool_name),
        )
    )
    if response.error is not None:
        return None, ToolArgumentError(
            code=f"llm_{response.error.code}",
            message=response.error.message,
            details=[{"provider": response.error.provider}],
        )
    call = next((item for item in response.tool_calls if item.name == tool_name), None)
    if call is None:
        return None, ToolArgumentError(
            code="missing_structured_call",
            message="The provider did not return the forced structured-output call.",
        )
    return spec.parse_arguments(call.arguments)
