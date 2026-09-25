from __future__ import annotations

import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from customer_support_agent.core import get_settings
from customer_support_agent.models import ChatMessage, LLMResponseModel, ToolCall, ToolDefinition

from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.default_llm_model

    async def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition] | None = None
    ) -> LLMResponseModel:
        request_tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools or []
        ]
        start = time.perf_counter()
        response = await self.client.responses.create(
            model=self.model,
            input=[message.model_dump(exclude_none=True) for message in messages],
            tools=request_tools,
        )
        calls = [
            ToolCall(id=item.call_id, name=item.name, arguments=item.arguments)
            for item in response.output
            if item.type == "function_call"
        ]
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
        response = await self.client.responses.create(
            model=self.model,
            input=[message.model_dump(exclude_none=True) for message in messages],
            stream=True,
        )
        async for event in response:
            if event.type == "response.output_text.delta":
                yield event.delta
