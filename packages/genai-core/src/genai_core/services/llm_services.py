from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from pydantic import BaseModel

from ..core import get_settings
from ..exceptions import LLMValidationError
from ..llm import (
    AnthropicProvider,
    GoogleProvider,
    LLMProvider,
    OpenAIProvider,
)
from ..models import (
    ChatMessage,
    LLMError,
    LLMManagerRequest,
    LLMManagerResponse,
    LLMResponseModel,
    ToolChoice,
    ToolSpec,
)


class LLMService:
    """Provider-independent entry point for structured outputs, tool-aware completions, and streaming."""

    def __init__(
        self,
        providers: dict[str, LLMProvider] | None = None,
        default_provider: str | None = None,
    ) -> None:
        self.default_provider = default_provider
        if providers is not None:
            self.providers = providers
        else:
            self.providers = self._init_default_providers()

    def _init_default_providers(self) -> dict[str, LLMProvider]:
        providers: dict[str, LLMProvider] = {}
        with contextlib.suppress(Exception):
            providers["google"] = GoogleProvider()
        with contextlib.suppress(Exception):
            providers["openai"] = OpenAIProvider()
        with contextlib.suppress(Exception):
            providers["anthropic"] = AnthropicProvider()
        return providers

    async def complete(self, request: LLMManagerRequest) -> LLMManagerResponse:
        settings = get_settings()
        provider_name: str = str(
            request.provider
            or self.default_provider
            or getattr(settings, "default_llm_provider", "google")
            or "google"
        )
        provider = self.providers.get(provider_name)
        if provider is None:
            return LLMManagerResponse(
                error=LLMError(
                    provider=provider_name,
                    code="unsupported_provider",
                    message=f"Unsupported LLM provider '{provider_name}'.",
                )
            )

        try:
            if request.response_schema is not None:
                prompt = request.prompt or (
                    request.messages[-1].content if request.messages else ""
                )
                response = await provider.complete_structure(
                    prompt=prompt, response_schema=request.response_schema
                )
            else:
                messages = list(request.messages)
                if not messages and request.prompt:
                    messages = [ChatMessage(role="user", content=request.prompt)]
                response = await provider.complete_tool(
                    messages=messages,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                )
        except LLMValidationError as error:
            return LLMManagerResponse(
                error=LLMError(
                    provider=provider_name,
                    code="validation_error",
                    message=str(error.message),
                    status_code=422,
                    retryable=False,
                )
            )
        except Exception as error:  # noqa: BLE001 - this boundary converts all provider failures
            status_code = getattr(error, "status_code", None)
            body = getattr(error, "body", None)
            provider_error = body.get("error", {}) if isinstance(body, dict) else {}
            if not isinstance(provider_error, dict):
                provider_error = {}
            code = provider_error.get("code") or getattr(error, "code", None)
            message = provider_error.get("message") or getattr(error, "message", None)
            if not message:
                message = str(error).strip() or "The provider request failed."
            return LLMManagerResponse(
                error=LLMError(
                    provider=provider_name,
                    code=str(code or type(error).__name__),
                    message=str(message),
                    status_code=status_code if isinstance(status_code, int) else None,
                    retryable=status_code in {408, 429, 500, 502, 503, 504},
                )
            )

        return LLMManagerResponse(
            text=response.text,
            structured_response=response.structured_response,
            tool_calls=response.tool_calls,
            model=response.model,
            usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
            raw_response=response.raw_response,
        )

    async def complete_structure(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        provider: str | None = None,
    ) -> LLMManagerResponse:
        request = LLMManagerRequest(
            provider=provider,
            prompt=prompt,
            response_schema=response_schema,
        )
        return await self.complete(request)

    async def complete_tool(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
        provider: str | None = None,
    ) -> LLMManagerResponse:
        request = LLMManagerRequest(
            provider=provider,
            messages=messages,
            tools=tools or [],
            tool_choice=tool_choice or ToolChoice(),
        )
        return await self.complete(request)

    async def complete_all(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> list[LLMResponseModel | BaseException]:
        tasks = [
            provider.complete_structure(prompt=prompt, response_schema=response_schema)
            for provider in self.providers.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def stream(
        self, provider_name: str, messages: list[ChatMessage] | str
    ) -> AsyncIterator[str]:
        provider = self.providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unsupported provider: {provider_name}")
        if isinstance(messages, str):
            messages = [ChatMessage(role="user", content=messages)]
        async for token in provider.stream(messages):
            yield token


LLMServicemanager = LLMService
