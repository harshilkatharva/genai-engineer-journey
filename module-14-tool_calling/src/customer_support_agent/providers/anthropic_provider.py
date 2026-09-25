from __future__ import annotations

import time
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from customer_support_agent.core import get_settings
from customer_support_agent.models import ChatMessage, LLMResponseModel, ToolCall, ToolDefinition

from .llm_provider import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        settings = get_settings()
        self.client = client or AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.default_llm_model

    async def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition] | None = None
    ) -> LLMResponseModel:
        system = "\n".join(message.content for message in messages if message.role == "system")
        request_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]
        start = time.perf_counter()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system or None,
            messages=request_messages,
            tools=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools or []
            ],
        )
        calls = [
            ToolCall(id=block.id, name=block.name, arguments=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponseModel(
            text=text or None,
            tool_calls=calls,
            model=response.model,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": messages[-1].content}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
