from unittest.mock import AsyncMock, MagicMock
from typing import AsyncIterator, Any

from hypothesis import given, strategies as st

import pytest

from llm_client.services.providers import AnthropicProvider, OpenAIProvider, GoogleProvider


@given(st.text())
@pytest.mark.asyncio
async def test_openai_provider_complete_mock(mock_clients: dict[str, Any], text: str) -> None:
    mock_response = MagicMock()
    mock_response.output_text = "This response created for mock test OpenAI" + text
    mock_response.provider = "openai"
    mock_response.usage.total_tokens = 30

    mock_client = mock_clients["openai"].return_value
    mock_client.responses.create = AsyncMock(return_value=mock_response)

    provider = OpenAIProvider()
    result = await provider.complete("Hello")

    assert result.provider == "openai"
    assert len(result.text) > 0
    assert result.latency_ms > 0.0
    assert text in result.text


class MockStreamOpenAI:
    def __aiter__(self) -> AsyncIterator[Any]:
        async def gen() -> AsyncIterator[Any]:
            for text in ["Hello", "how", "are", "you"]:
                event = MagicMock()
                event.type = "response.output_text.delta"
                event.delta = text
                yield event

        return gen()


@pytest.mark.asyncio
async def test_opnai_provider_stream_mock(mock_clients: dict[str, Any]) -> None:
    mock_client = mock_clients["openai"].return_value
    mock_client.responses.create = AsyncMock(return_value=MockStreamOpenAI())

    provider = OpenAIProvider()

    chunks = [chunk async for chunk in provider.stream("Hello")]

    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_google_provider_complete_mock(mock_clients: dict[str, Any]) -> None:
    mock_response = MagicMock()
    mock_response.text = "This response is created for mock test"
    mock_response.usage_metadata.total_token_count = 30
    mock_response.provider = "google"
    mock_response.latency_ms = 150

    mock_client = mock_clients["google"].return_value
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GoogleProvider()
    result = await provider.complete("Hello")

    assert result.provider == "google"
    assert len(result.text) > 0
    assert result.latency_ms > 0.0


class MockStreamGoogle:
    def __aiter__(self) -> AsyncIterator[Any]:
        async def gen() -> AsyncIterator[Any]:
            for text in ["Hello", "how", "are", "you"]:
                chunk = MagicMock()
                chunk.text = text
                yield chunk

        return gen()


@pytest.mark.asyncio
async def test_google_provider_stream_mock(mock_clients: dict[str, Any]) -> None:
    mock_client = mock_clients["google"].return_value

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=MockStreamGoogle())

    provider = GoogleProvider()
    chunks = [chunk async for chunk in provider.stream("Hello")]

    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_anthropic_provider_complete_mock(mock_clients: dict[str, Any]) -> None:
    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "This response created for mock test"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 200

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    mock_response.usage = mock_usage

    mock_client = mock_clients["anthropic"].return_value
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = AnthropicProvider()
    result = await provider.complete("Hello")

    assert result.provider == "anthropic"
    assert len(result.text) > 0
    assert result.latency_ms >= 0.0


async def MockStreamAnthropic() -> AsyncIterator[Any]:
    for text in ["Hello", "how", "are", "you"]:
        yield text + " "


@pytest.mark.asyncio
async def test_anthropic_provider_stream_mock(mock_clients: dict[str, Any]) -> None:
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=context)
    context.__aexit__ = AsyncMock(return_value=None)
    context.text_stream = MockStreamAnthropic()

    mock_client = mock_clients["anthropic"].return_value
    mock_client.messages.stream.return_value = context

    provider = AnthropicProvider()
    result = [chunk async for chunk in provider.stream("Hello")]

    assert len(result) >= 2
