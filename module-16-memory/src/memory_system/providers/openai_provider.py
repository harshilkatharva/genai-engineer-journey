from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam, ResponseInputParam
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from ..core import get_settings
from ..models import ChatMessage, LLMResponseModel, ToolCall, ToolChoice, ToolSpec
from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = (
            settings.default_llm_model if "gpt" in settings.default_llm_model else "gpt-4o-mini"
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponseModel:
        request_tools: list[FunctionToolParam] = []
        for tool in tools or []:
            request_tools.append(tool.to_openai_tool())
        start = time.perf_counter()
        request_options = {"tools": request_tools} if request_tools else {}
        if request_tools:
            request_options["tool_choice"] = (tool_choice or ToolChoice()).to_openai()
        response = await self.client.responses.create(
            model=self.model,
            input=cast(
                ResponseInputParam,
                [message.model_dump(exclude_none=True) for message in messages],
            ),
            **request_options,
        )
        calls: list[ToolCall] = []
        for item in response.output:
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = item.arguments
            calls.append(ToolCall(id=item.call_id, name=item.name, arguments=arguments))
        usage = response.usage
        return LLMResponseModel(
            text=response.output_text if not calls else None,
            tool_calls=calls,
            model=response.model,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw_response=response,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self.client.responses.stream(
            model=self.model,
            input=cast(
                ResponseInputParam,
                [message.model_dump(exclude_none=True) for message in messages],
            ),
        ) as stream:
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    yield event.delta
