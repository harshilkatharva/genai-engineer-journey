from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import cast

from anthropic import AsyncAnthropic, Omit
from anthropic.types import MessageParam

from ..core import get_settings
from ..models import ChatMessage, LLMResponseModel, ToolCall, ToolChoice, ToolSpec
from .llm_provider import LLMProvider


class AnthropicProvider(LLMProvider):
    # Configures the Anthropic client and selected model.
    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        settings = get_settings()
        self.client = client or AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = (
            settings.default_llm_model
            if "claude" in settings.default_llm_model
            else "claude-3-5-sonnet-20241022"
        )

    # Sends messages to Anthropic and maps text, tool calls, and usage.
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponseModel:
        system = "\n".join(message.content for message in messages if message.role == "system")
        request_messages = cast(
            list[MessageParam],
            [
                {"role": message.role, "content": message.content}
                for message in messages
                if message.role != "system"
            ],
        )
        start = time.perf_counter()
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system if system else Omit(),
            messages=request_messages,
            tools=([tool.to_anthropic_tool() for tool in tools] if tools else Omit()),
            tool_choice=((tool_choice or ToolChoice()).to_anthropic() if tools else Omit()),
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

    # Streams generated text from Anthropic.
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": messages[-1].content}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
