from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from anthropic import AsyncAnthropic, Omit
from anthropic.types import MessageParam
from pydantic import BaseModel, ValidationError

from ..core import get_settings
from ..exceptions import LLMValidationError
from ..models import ChatMessage, LLMResponseModel, ToolCall, ToolChoice, ToolSpec
from .llm_provider import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider adapter supporting structured outputs, tool calling, and streaming."""

    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self.model = model or (
            settings.default_llm_model
            if "claude" in settings.default_llm_model
            else "claude-3-5-sonnet-20241022"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete_structure(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        max_retries: int | None = None,
    ) -> LLMResponseModel:
        if response_schema is None:
            start = time.perf_counter()
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            response = await self.client.messages.create(**request_kwargs)
            latency = (time.perf_counter() - start) * 1000

            text = "".join(
                getattr(block, "text", "")
                for block in getattr(response, "content", []) or []
                if getattr(block, "type", None) == "text"
            )

            usage = getattr(response, "usage", None)
            return LLMResponseModel(
                text=text or None,
                structured_response=None,
                model=getattr(response, "model", self.model),
                latency_ms=latency,
                input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                raw_response=response,
            )

        start = time.perf_counter()
        effective_max_retries = (
            max_retries if max_retries is not None else get_settings().max_structured_error_retry
        )
        current_prompt = (
            f"{prompt}\n\n"
            f"Please output a valid JSON object strictly matching this schema: "
            f"{json.dumps(response_schema.model_json_schema())}"
        )
        total_input_tokens = 0
        total_output_tokens = 0
        last_error: Exception | None = None
        last_response_text: str | None = None

        for attempt in range(effective_max_retries + 1):
            request_kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": current_prompt}],
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            response = await self.client.messages.create(**request_kwargs)
            usage = getattr(response, "usage", None)
            total_input_tokens += getattr(usage, "input_tokens", 0) if usage else 0
            total_output_tokens += getattr(usage, "output_tokens", 0) if usage else 0

            text = "".join(
                getattr(block, "text", "")
                for block in getattr(response, "content", []) or []
                if getattr(block, "type", None) == "text"
            )
            last_response_text = text

            if text:
                try:
                    parsed_json = json.loads(text)
                    validated_output = response_schema.model_validate(parsed_json)
                    latency = (time.perf_counter() - start) * 1000
                    return LLMResponseModel(
                        text=text,
                        structured_response=validated_output,
                        model=getattr(response, "model", self.model),
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
                last_error = ValueError("Empty text received from model")

            if attempt < effective_max_retries:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation with the required schema.\n"
                    f"Previous output:\n{text or ''}\n"
                    f"Validation error:\n{last_error}\n"
                    f"Please correct the error and return only a valid JSON object matching the required schema: "
                    f"{json.dumps(response_schema.model_json_schema())}"
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
        request_options: dict[str, Any] = {
            "tools": [tool.to_anthropic_tool() for tool in tools or []]
        }
        if tools:
            request_options["tool_choice"] = (tool_choice or ToolChoice()).to_anthropic()
        if self.temperature is not None:
            request_options["temperature"] = self.temperature

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system if system else Omit(),
            messages=request_messages,
            **cast(Any, request_options),
        )
        calls = [
            ToolCall(
                id=getattr(block, "id", ""),
                name=getattr(block, "name", ""),
                arguments=getattr(block, "input", {}),
            )
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", None) == "tool_use"
        ]
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(response, "content", []) or []
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return LLMResponseModel(
            text=text or None,
            tool_calls=calls,
            model=getattr(response, "model", self.model),
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw_response=response,
        )

    async def stream(self, messages: list[ChatMessage] | str) -> AsyncIterator[str]:
        if isinstance(messages, str):
            content = messages
        else:
            content = messages[-1].content if messages else ""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
