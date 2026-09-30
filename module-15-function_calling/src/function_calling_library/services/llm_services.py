from __future__ import annotations

from collections.abc import AsyncIterator

from ..core import get_settings
from ..models import (
    LLMError,
    LLMManagerRequest,
    LLMManagerResponse,
)
from ..providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
)
from ..providers.llm_provider import LLMProvider


class LLMService:
    """Provider-independent entry point for text and tool-aware completions."""

    def __init__(self, providers: dict[str, LLMProvider] | None = None) -> None:
        self.providers = providers or {
            "google": GoogleProvider(),
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
        }

    async def complete(self, request: LLMManagerRequest) -> LLMManagerResponse:
        provider_name = request.provider or get_settings().default_llm_provider
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
            response = await provider.complete(request.messages, request.tools, request.tool_choice)
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
            tool_calls=response.tool_calls,
            model=response.model,
            usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
            raw_response=response.raw_response,
        )

    async def stream(self, provider_name: str, messages: list) -> AsyncIterator[str]:
        provider = self.providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unsupported provider: {provider_name}")
        async for token in provider.stream(messages):
            yield token


LLMServicemanager = LLMService
