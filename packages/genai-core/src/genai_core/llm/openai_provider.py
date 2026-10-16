from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.responses import FunctionToolParam, ResponseInputParam
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from pydantic import BaseModel, ValidationError

from ..core import get_settings
from ..exceptions import LLMValidationError
from ..models import ChatMessage, LLMResponseModel, ToolCall, ToolChoice, ToolSpec
from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider adapter supporting structured outputs, tool calling, and streaming."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or (
            settings.default_llm_model if "gpt" in settings.default_llm_model else "gpt-4o-mini"
        )
        self.temperature = temperature

    @staticmethod
    def _prepare_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
        result = dict(schema)
        if result.get("type") == "object":
            result["additionalProperties"] = False
            if "properties" in result and isinstance(result["properties"], dict):
                props = {}
                for k, v in result["properties"].items():
                    props[k] = (
                        OpenAIProvider._prepare_strict_json_schema(v) if isinstance(v, dict) else v
                    )
                result["properties"] = props
                if "required" not in result:
                    result["required"] = list(props.keys())
        if "$defs" in result and isinstance(result["$defs"], dict):
            defs = {}
            for k, v in result["$defs"].items():
                defs[k] = (
                    OpenAIProvider._prepare_strict_json_schema(v) if isinstance(v, dict) else v
                )
            result["$defs"] = defs
        return result

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
                "input": prompt,
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            response = await self.client.responses.create(**request_kwargs)
            latency = (time.perf_counter() - start) * 1000
            usage = getattr(response, "usage", None)

            return LLMResponseModel(
                text=getattr(response, "output_text", None),
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
        current_prompt = prompt
        total_input_tokens = 0
        total_output_tokens = 0
        last_error: Exception | None = None
        last_response_text: str | None = None

        for attempt in range(effective_max_retries + 1):
            prepared_schema = self._prepare_strict_json_schema(response_schema.model_json_schema())
            request_kwargs = {
                "model": self.model,
                "input": current_prompt,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": response_schema.__name__,
                        "schema": prepared_schema,
                        "strict": True,
                    }
                },
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            response = await self.client.responses.create(**request_kwargs)
            usage = getattr(response, "usage", None)
            total_input_tokens += getattr(usage, "input_tokens", 0) if usage else 0
            total_output_tokens += getattr(usage, "output_tokens", 0) if usage else 0

            output_text = getattr(response, "output_text", None)
            last_response_text = output_text

            if output_text:
                try:
                    parsed_json = json.loads(output_text)
                    validated_output = response_schema.model_validate(parsed_json)
                    latency = (time.perf_counter() - start) * 1000
                    return LLMResponseModel(
                        text=output_text,
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
                last_error = ValueError("Empty output text received from model")

            if attempt < effective_max_retries:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation with the required schema.\n"
                    f"Previous output:\n{output_text or ''}\n"
                    f"Validation error:\n{last_error}\n"
                    f"Please correct the error and return only a valid JSON object matching the required schema."
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
        request_tools: list[FunctionToolParam] = []
        for tool in tools or []:
            request_tools.append(tool.to_openai_tool())
        start = time.perf_counter()
        request_options: dict[str, Any] = {"tools": request_tools} if request_tools else {}
        if request_tools:
            request_options["tool_choice"] = (tool_choice or ToolChoice()).to_openai()
        if self.temperature is not None:
            request_options["temperature"] = self.temperature

        response = await self.client.responses.create(
            model=self.model,
            input=cast(
                ResponseInputParam,
                [message.model_dump(exclude_none=True) for message in messages],
            ),
            **cast(Any, request_options),
        )
        calls: list[ToolCall] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            raw_arguments = getattr(item, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = raw_arguments
            calls.append(
                ToolCall(
                    id=getattr(item, "call_id", ""),
                    name=getattr(item, "name", ""),
                    arguments=arguments,
                )
            )
        usage = getattr(response, "usage", None)
        return LLMResponseModel(
            text=getattr(response, "output_text", None) if not calls else None,
            tool_calls=calls,
            model=getattr(response, "model", self.model),
            latency_ms=(time.perf_counter() - start) * 1000,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw_response=response,
        )

    async def stream(self, messages: list[ChatMessage] | str) -> AsyncIterator[str]:
        input_data: Any
        if isinstance(messages, str):
            input_data = messages
        else:
            input_data = cast(
                ResponseInputParam,
                [message.model_dump(exclude_none=True) for message in messages],
            )
        async with self.client.responses.stream(
            model=self.model,
            input=input_data,
        ) as stream:
            async for event in stream:
                if (
                    isinstance(event, ResponseTextDeltaEvent)
                    or hasattr(event, "delta")
                    and isinstance(event.delta, str)
                ):
                    yield event.delta
