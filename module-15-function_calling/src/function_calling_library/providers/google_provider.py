from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from ..core import get_settings
from ..models import ChatMessage, LLMResponseModel, ToolCall, ToolChoice, ToolSpec
from .llm_provider import LLMProvider


class GoogleProvider(LLMProvider):
    def __init__(self, client: genai.Client | None = None) -> None:
        settings = get_settings()
        self.client = client or genai.Client(api_key=settings.google_api_key)
        self.model = (
            settings.default_llm_model
            if "gemini" in settings.default_llm_model
            else "gemini-3.5-flash-lite"
        )
        self.temperature = settings.default_llm_temperature or 0.2

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponseModel:
        declarations = [
            types.FunctionDeclaration(**tool.to_google_function_declaration())
            for tool in tools or []
        ]
        system_instruction = "\n\n".join(
            message.content for message in messages if message.role == "system"
        )
        tool_choice_config = tool_choice.to_google() if tool_choice and declarations else None
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction or None,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            tool_config=(
                types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(**tool_choice_config)
                )
                if tool_choice_config
                else None
            ),
        )
        contents: list[types.Content] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                if not message.tool_name:
                    raise ValueError("Google tool-result messages require tool_name")
                try:
                    result = json.loads(message.content)
                except json.JSONDecodeError:
                    result = message.content
                if not isinstance(result, dict):
                    result = {"result": result}
                function_response = types.FunctionResponse(
                    id=message.tool_call_id,
                    name=message.tool_name,
                    response=result,
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(function_response=function_response)],
                    )
                )
                continue
            contents.append(
                types.Content(
                    role="model" if message.role == "assistant" else "user",
                    parts=[types.Part(text=message.content)],
                )
            )
        start = time.perf_counter()
        response = await self.client.aio.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        calls: list[ToolCall] = []
        for candidate in response.candidates or []:
            parts = candidate.content.parts if candidate.content else None
            for part in parts or []:
                function_call = part.function_call
                if function_call and function_call.name:
                    calls.append(
                        ToolCall(
                            id=f"google-{len(calls) + 1}",
                            name=function_call.name,
                            arguments=dict(function_call.args or {}),
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
