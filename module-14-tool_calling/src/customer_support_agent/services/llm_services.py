from __future__ import annotations

from collections.abc import AsyncIterator

from customer_support_agent.core import get_settings
from customer_support_agent.models import (
    LLMManagerRequest,
    LLMManagerResponse,
)
from customer_support_agent.providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
)
from customer_support_agent.providers.llm_provider import LLMProvider


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
            raise ValueError(f"Unsupported provider: {provider_name}")
        response = await provider.complete(request.messages, request.tools)
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
