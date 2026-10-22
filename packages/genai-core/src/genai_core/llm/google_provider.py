from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from ..core import get_settings
from ..exceptions import LLMValidationError
from ..models import ChatMessage, LLMResponseModel, ToolCall, ToolChoice, ToolSpec
from .llm_provider import LLMProvider


class GoogleProvider(LLMProvider):
    """Google Gemini LLM provider adapter supporting structured outputs, tool calling, and streaming."""

    def __init__(
        self,
        client: genai.Client | None = None,
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        min_call_interval: float = 3.0,
    ) -> None:
        settings = get_settings()
        self.client = client or genai.Client(api_key=api_key or settings.google_api_key)
        self.model = model or (
            settings.default_llm_model
            if "gemini" in settings.default_llm_model
            else "gemini-3.5-flash-lite"
        )
        self.temperature = (
            temperature
            if temperature is not None
            else (
                settings.default_llm_temperature
                if settings.default_llm_temperature is not None
                else 0.2
            )
        )
        self.min_call_interval = min_call_interval
        self._last_call_time: float = 0.0
        self._thought_signatures: dict[str, bytes | str] = {}

    async def complete_structure(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        max_retries: int | None = None,
    ) -> LLMResponseModel:
        if response_schema is None:
            start = time.perf_counter()
            config = types.GenerateContentConfig(
                temperature=self.temperature,
            )
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            latency = (time.perf_counter() - start) * 1000
            usage = getattr(response, "usage_metadata", None)
            return LLMResponseModel(
                text=getattr(response, "text", None),
                structured_response=None,
                model=getattr(response, "model_version", None) or self.model,
                latency_ms=latency,
                input_tokens=(getattr(usage, "prompt_token_count", 0) if usage else 0) or 0,
                output_tokens=(getattr(usage, "candidates_token_count", 0) if usage else 0) or 0,
                raw_response=response,
            )

        start = time.perf_counter()
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        effective_max_retries = (
            max_retries if max_retries is not None else get_settings().max_structured_error_retry
        )
        current_prompt = prompt
        total_input_tokens = 0
        total_output_tokens = 0
        last_error: Exception | None = None
        last_response_text: str | None = None

        for attempt in range(effective_max_retries + 1):
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=current_prompt,
                config=config,
            )
            usage = getattr(response, "usage_metadata", None)
            total_input_tokens += (getattr(usage, "prompt_token_count", 0) if usage else 0) or 0
            total_output_tokens += (
                getattr(usage, "candidates_token_count", 0) if usage else 0
            ) or 0
            response_text = getattr(response, "text", None)
            last_response_text = response_text

            if response_text:
                try:
                    parsed_json = json.loads(response_text)
                    validated_output = response_schema.model_validate(parsed_json)
                    latency = (time.perf_counter() - start) * 1000
                    return LLMResponseModel(
                        text=response_text,
                        structured_response=validated_output,
                        model=getattr(response, "model_version", None) or self.model,
                        latency_ms=latency,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        raw_response=response,
                    )
                except (
                    json.JSONDecodeError,
                    ValidationError,
                    TypeError,
                    ValueError,
                ) as exc:
                    last_error = exc
            else:
                last_error = ValueError("Empty response text received from model")

            if attempt < effective_max_retries:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation with the required schema.\n"
                    f"Previous output:\n{response_text or ''}\n"
                    f"Validation error:\n{last_error}\n"
                    f"Please correct the output and return only valid JSON matching the schema."
                )

        raise LLMValidationError(
            message=f"Failed to validate response against schema '{response_schema.__name__}' after {effective_max_retries} retries. Last error: {last_error}",
            raw_output=last_response_text,
            errors=[str(last_error)],
        )

    async def complete_tool(
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
                except (json.JSONDecodeError, TypeError):
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
            if message.role == "assistant":
                if message.tool_calls:
                    parts: list[types.Part] = []
                    if message.content:
                        parts.append(types.Part(text=message.content))
                    for tc in message.tool_calls:
                        sig = getattr(tc, "thought_signature", None)
                        if sig is None:
                            sig = self._thought_signatures.get(
                                tc.id
                            ) or self._thought_signatures.get(f"{tc.name}_{tc.id}")
                        sig_bytes: bytes | None = None
                        if isinstance(sig, bytes):
                            sig_bytes = sig
                        elif isinstance(sig, str):
                            try:
                                sig_bytes = base64.b64decode(sig, validate=True)
                            except (ValueError, binascii.Error):
                                sig_bytes = sig.encode("utf-8")
                        fn_call = types.FunctionCall(
                            name=tc.name,
                            args=tc.arguments if isinstance(tc.arguments, dict) else {},
                            id=tc.id if tc.id else None,
                        )
                        parts.append(types.Part(function_call=fn_call, thought_signature=sig_bytes))
                    contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(text=message.content)],
                        )
                    )
                continue
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=message.content)],
                )
            )

        # Enforce minimum call interval (pacing) between successive requests
        if self.min_call_interval > 0.0 and self._last_call_time > 0.0:
            elapsed = time.monotonic() - self._last_call_time
            if elapsed < self.min_call_interval:
                await asyncio.sleep(self.min_call_interval - elapsed)

        start = time.perf_counter()
        response = None
        for attempt in range(4):
            try:
                self._last_call_time = time.monotonic()
                response = await self.client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                break
            except Exception as exc:
                err_str = str(exc)
                if (
                    "429" in err_str
                    or "quota" in err_str.lower()
                    or "resource_exhausted" in err_str.lower()
                ) and attempt < 3:
                    match = re.search(r"retry in ([0-9\.]+)s", err_str, re.IGNORECASE)
                    wait_sec = float(match.group(1)) + 1.5 if match else (5.0 * (attempt + 1))
                    await asyncio.sleep(wait_sec)
                    continue
                raise

        if response is None:
            raise RuntimeError("Google API generation failed without response")

        calls: list[ToolCall] = []
        for candidate in getattr(response, "candidates", []) or []:
            candidate_parts = candidate.content.parts if candidate and candidate.content else None
            for part in candidate_parts or []:
                function_call = getattr(part, "function_call", None)
                if function_call and getattr(function_call, "name", None):
                    sig = getattr(part, "thought_signature", None)
                    call_id = getattr(function_call, "id", None) or f"call_{uuid.uuid4().hex[:12]}"
                    if sig is not None:
                        self._thought_signatures[call_id] = sig
                        self._thought_signatures[f"{function_call.name}_{call_id}"] = sig
                    calls.append(
                        ToolCall(
                            id=call_id,
                            name=function_call.name,
                            arguments=dict(getattr(function_call, "args", {}) or {}),
                            thought_signature=sig,
                        )
                    )
        usage = getattr(response, "usage_metadata", None)
        return LLMResponseModel(
            text=getattr(response, "text", None) if not calls else None,
            tool_calls=calls,
            model=getattr(response, "model_version", None) or self.model,
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            raw_response=response,
        )

    async def stream(self, messages: list[ChatMessage] | str) -> AsyncIterator[str]:
        system_instruction: str | None = None
        if isinstance(messages, str):
            contents: list[types.Content] = [
                types.Content(role="user", parts=[types.Part(text=messages)])
            ]
        else:
            system_parts = [m.content for m in messages if m.role == "system"]
            if system_parts:
                system_instruction = "\n".join(system_parts)
            contents = [
                types.Content(
                    role="model" if m.role == "assistant" else "user",
                    parts=[types.Part(text=m.content)],
                )
                for m in messages
                if m.role != "system"
            ]
            if not contents:
                contents = [types.Content(role="user", parts=[types.Part(text="")])]

        config = (
            types.GenerateContentConfig(system_instruction=system_instruction)
            if system_instruction
            else None
        )
        kwargs: dict[str, Any] = {}
        if config is not None:
            kwargs["config"] = config
        stream_or_coro = self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            **kwargs,
        )
        if inspect.isawaitable(stream_or_coro):
            response = await stream_or_coro
        else:
            response = stream_or_coro
        async for chunk in response:
            chunk_text = getattr(chunk, "text", None)
            if isinstance(chunk_text, str) and chunk_text:
                yield chunk_text
