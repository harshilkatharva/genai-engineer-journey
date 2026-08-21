from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from rag_app.exceptions.llm_exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from rag_app.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.default_llm_provider = "anthropic"
    settings.default_llm_model = "claude-sonnet-4-6"
    return settings


@pytest.fixture
def provider(mock_settings):
    with (
        patch("rag_app.providers.anthropic_provider.AsyncAnthropic") as mock_client,
        patch(
            "rag_app.providers.anthropic_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = AnthropicProvider()
        provider.client = mock_client.return_value

        yield provider


@pytest.fixture
def anthropic_response():
    response = MagicMock()

    response.model = "claude-sonnet-4-6"

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "This is an Anthropic response."

    response.content = [text_block]

    response.usage.input_tokens = 25
    response.usage.output_tokens = 15

    return response


@pytest.mark.asyncio
async def test_complete_returns_llm_response(
    provider,
    anthropic_response,
):
    provider.client.messages.create = AsyncMock(return_value=anthropic_response)

    result = await provider.complete("Explain RAG in simple terms.")

    assert result.text == "This is an Anthropic response."
    assert result.model == "claude-sonnet-4-6"
    assert result.input_tokens == 25
    assert result.output_tokens == 15
    assert result.latency_ms >= 0

    provider.client.messages.create.assert_awaited_once_with(
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "user",
                "content": "Explain RAG in simple terms.",
            }
        ],
        max_tokens=1024,
    )


@pytest.mark.asyncio
async def test_complete_uses_configured_model(mock_settings):
    mock_settings.default_llm_provider = "anthropic"
    mock_settings.default_llm_model = "my-custom-claude-model"

    with (
        patch(
            "rag_app.providers.anthropic_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = AnthropicProvider()

        response = MagicMock()
        response.model = "my-custom-claude-model"

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Response"

        response.content = [text_block]
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5

        provider.client.messages.create = AsyncMock(return_value=response)

        result = await provider.complete("Hello")

    assert result.model == "my-custom-claude-model"

    provider.client.messages.create.assert_awaited_once_with(
        model="my-custom-claude-model",
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        max_tokens=1024,
    )


@pytest.mark.asyncio
async def test_complete_raises_llm_rate_limit_error(provider):
    provider.client.messages.create = AsyncMock(
        side_effect=RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(),
            body={},
        )
    )

    with pytest.raises(LLMRateLimitError):
        await provider.complete("Hello")


@pytest.mark.asyncio
async def test_complete_raises_llm_connection_error(provider):
    provider.client.messages.create = AsyncMock(
        side_effect=APIConnectionError(
            request=MagicMock(),
        )
    )

    with pytest.raises(LLMConnectionError):
        await provider.complete("Hello")


@pytest.mark.asyncio
async def test_complete_raises_llm_timeout_error(provider):
    provider.client.messages.create = AsyncMock(
        side_effect=APITimeoutError(
            request=MagicMock(),
        )
    )

    with pytest.raises(LLMTimeoutError):
        await provider.complete("Hello")


@pytest.mark.asyncio
async def test_complete_raises_llm_authentication_error(provider):
    provider.client.messages.create = AsyncMock(
        side_effect=AuthenticationError(
            message="Invalid API key",
            response=MagicMock(),
            body={},
        )
    )

    with pytest.raises(LLMAuthenticationError):
        await provider.complete("Hello")


@pytest.mark.asyncio
async def test_complete_raises_llm_error(provider):
    provider.client.messages.create = AsyncMock(
        side_effect=APIError(
            message="Anthropic API error",
            request=MagicMock(),
            body={},
        )
    )

    with pytest.raises(LLMError):
        await provider.complete("Hello")


@pytest.mark.asyncio
async def test_stream_returns_text_deltas(provider):
    stream = MagicMock()

    async def text_stream():
        yield "Hello"
        yield " world"
        yield "!"

    stream.text_stream = text_stream()

    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=stream)
    context_manager.__aexit__ = AsyncMock(return_value=None)

    provider.client.messages.stream = MagicMock(return_value=context_manager)

    chunks = []

    async for text in provider.stream("Say hello"):
        chunks.append(text)

    assert chunks == [
        "Hello",
        " world",
        "!",
    ]

    provider.client.messages.stream.assert_called_once_with(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Say hello",
            }
        ],
    )
