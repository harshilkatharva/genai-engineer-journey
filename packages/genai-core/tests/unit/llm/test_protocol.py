from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

from pydantic import BaseModel

from genai_core.llm import (
    AnthropicProvider,
    GoogleProvider,
    LLMProvider,
    OpenAIProvider,
)
from genai_core.models import ChatMessage, LLMResponseModel, ToolChoice, ToolSpec


class MockCustomProvider:
    async def complete_structure(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
        max_retries: int | None = None,
        model: str | None = None,
    ) -> LLMResponseModel:
        return LLMResponseModel(text="structured", model=model or "mock", latency_ms=1.0)

    async def complete_tool(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
        model: str | None = None,
    ) -> LLMResponseModel:
        return LLMResponseModel(text="tool", model=model or "mock", latency_ms=1.0)

    async def stream(
        self, messages: list[ChatMessage] | str, model: str | None = None
    ) -> AsyncIterator[str]:
        yield f"chunk_{model or 'mock'}"


def test_custom_provider_conforms_to_protocol() -> None:
    provider = MockCustomProvider()
    assert isinstance(provider, LLMProvider)


def test_google_provider_conforms_to_protocol() -> None:
    fake_client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace())))
    provider = GoogleProvider(client=fake_client)
    assert isinstance(provider, LLMProvider)


def test_openai_provider_conforms_to_protocol() -> None:
    fake_client = cast(Any, SimpleNamespace(responses=SimpleNamespace()))
    provider = OpenAIProvider(client=fake_client)
    assert isinstance(provider, LLMProvider)


def test_anthropic_provider_conforms_to_protocol() -> None:
    fake_client = cast(Any, SimpleNamespace(messages=SimpleNamespace()))
    provider = AnthropicProvider(client=fake_client)
    assert isinstance(provider, LLMProvider)
