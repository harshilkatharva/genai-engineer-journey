from __future__ import annotations

import time
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from customer_support_agent.core import get_settings
from customer_support_agent.models import ChatMessage, LLMResponseModel, ToolCall, ToolDefinition

from .llm_provider import LLMProvider


class GoogleProvider(LLMProvider):
    def __init__(self, client: genai.Client | None = None) -> None:
        settings = get_settings()
        self.client = client or genai.Client(api_key=settings.google_api_key)
        self.model = settings.default_llm_model
        self.temperature = settings.default_llm_temperature

    async def complete(
        self, messages: list[ChatMessage], tools: list[ToolDefinition] | None = None
    ) -> LLMResponseModel:
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in tools or []
        ]
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
        )
        contents = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part(text=message.content)],
            )
            for message in messages
        ]
        start = time.perf_counter()
        response = await self.client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        calls: list[ToolCall] = []
        for candidate in response.candidates or []:
            for part in candidate.content.parts if candidate.content else []:
                if part.function_call:
                    calls.append(
                        ToolCall(
                            id=f"google-{len(calls) + 1}",
                            name=part.function_call.name,
                            arguments=dict(part.function_call.args or {}),
                        )
                    )
        usage = response.usage_metadata
        return LLMResponseModel(
            text=response.text if not calls else None,
            tool_calls=calls,
            model=self.model,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            raw_response=response,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        response = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=[message.content for message in messages],
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text
