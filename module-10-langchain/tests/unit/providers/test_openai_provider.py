from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import (
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
from rag_app.providers.openai_provider import OpenAIProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.default_llm_provider = "openai"
    settings.default_llm_model = "gpt-4o-mini"

    return settings


@pytest.fixture
def provider(mock_settings):
    with (
        patch("rag_app.providers.openai_provider.AsyncOpenAI") as mock_client,
        patch(
            "rag_app.providers.openai_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = OpenAIProvider()

        provider.client = mock_client.return_value

        yield provider


@pytest.fixture
def openai_response():
    response = MagicMock()

    response.output_text = "This is an OpenAI response."
    response.model = "gpt-4o-mini"

    response.usage.input_tokens = 25
    response.usage.output_tokens = 15

    return response


# ---------------------------------------------------------------------------
# Complete - success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_llm_response(
    provider,
    openai_response,
):
    provider.client.responses.create = AsyncMock(return_value=openai_response)

    result = await provider.complete("Explain RAG in simple terms.")

    assert result.text == "This is an OpenAI response."
    assert result.model == "gpt-4o-mini"
    assert result.input_tokens == 25
    assert result.output_tokens == 15
    assert result.latency_ms >= 0

    provider.client.responses.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        input="Explain RAG in simple terms.",
    )


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_uses_configured_openai_model(
    mock_settings,
):
    mock_settings.default_llm_provider = "openai"
    mock_settings.default_llm_model = "gpt-custom-model"

    with (
        patch(
            "rag_app.providers.openai_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = OpenAIProvider()

        response = MagicMock()
        response.output_text = "Response"
        response.model = "gpt-custom-model"

        response.usage.input_tokens = 10
        response.usage.output_tokens = 5

        provider.client.responses.create = AsyncMock(return_value=response)

        result = await provider.complete("Hello")

    assert result.model == "gpt-custom-model"

    provider.client.responses.create.assert_awaited_once_with(
        model="gpt-custom-model",
        input="Hello",
    )


@pytest.mark.asyncio
async def test_complete_uses_fallback_model_when_provider_is_not_openai(
    mock_settings,
):
    mock_settings.default_llm_provider = "anthropic"
    mock_settings.default_llm_model = "claude-sonnet"

    with (
        patch(
            "rag_app.providers.openai_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = OpenAIProvider()

        response = MagicMock()
        response.output_text = "Response"
        response.model = "gpt-4"

        response.usage.input_tokens = 10
        response.usage.output_tokens = 5

        provider.client.responses.create = AsyncMock(return_value=response)

        result = await provider.complete("Hello")

    assert result.model == "gpt-4"

    provider.client.responses.create.assert_awaited_once_with(
        model="gpt-4",
        input="Hello",
    )


# ---------------------------------------------------------------------------
# Complete - response handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_empty_token_values_as_zero(
    provider,
):
    response = MagicMock()

    response.output_text = "Response"
    response.model = "gpt-4o-mini"

    response.usage.input_tokens = 0
    response.usage.output_tokens = 0

    provider.client.responses.create = AsyncMock(return_value=response)

    result = await provider.complete("Hello")

    assert result.input_tokens == 0
    assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_llm_rate_limit_error(provider):
    error = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(),
        body={},
    )

    provider.client.responses.create = AsyncMock(side_effect=error)

    with pytest.raises(LLMRateLimitError):
        await provider.complete("Hello")


# ---------------------------------------------------------------------------
# Connection error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_llm_connection_error(provider):
    error = APIConnectionError(
        request=MagicMock(),
    )

    provider.client.responses.create = AsyncMock(side_effect=error)

    with pytest.raises(LLMConnectionError):
        await provider.complete("Hello")


# ---------------------------------------------------------------------------
# Timeout error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_llm_timeout_error(provider):
    error = APITimeoutError(
        request=MagicMock(),
    )

    provider.client.responses.create = AsyncMock(side_effect=error)

    with pytest.raises(LLMTimeoutError):
        await provider.complete("Hello")


# ---------------------------------------------------------------------------
# Authentication error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_llm_authentication_error(provider):
    error = AuthenticationError(
        message="Invalid API key",
        response=MagicMock(),
        body={},
    )

    provider.client.responses.create = AsyncMock(side_effect=error)

    with pytest.raises(LLMAuthenticationError):
        await provider.complete("Hello")


# ---------------------------------------------------------------------------
# Generic API error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_llm_error(provider):
    error = APIError(
        message="OpenAI API error",
        request=MagicMock(),
        body={},
    )

    provider.client.responses.create = AsyncMock(side_effect=error)

    with pytest.raises(LLMError):
        await provider.complete("Hello")


# ---------------------------------------------------------------------------
# Stream - success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_returns_text_deltas(provider):
    event_1 = MagicMock()
    event_1.type = "response.output_text.delta"
    event_1.delta = "Hello"

    event_2 = MagicMock()
    event_2.type = "response.output_text.delta"
    event_2.delta = " world"

    event_3 = MagicMock()
    event_3.type = "response.completed"
    event_3.delta = None

    async def mock_stream():
        yield event_1
        yield event_2
        yield event_3

    provider.client.responses.create = AsyncMock(return_value=mock_stream())

    chunks = []

    async for text in provider.stream("Say hello"):
        chunks.append(text)

    assert chunks == [
        "Hello",
        " world",
    ]

    provider.client.responses.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        input="Say hello",
        stream=True,
    )


# ---------------------------------------------------------------------------
# Stream - ignores non-text events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_ignores_non_text_events(provider):
    event_1 = MagicMock()
    event_1.type = "response.created"
    event_1.delta = None

    event_2 = MagicMock()
    event_2.type = "response.output_text.delta"
    event_2.delta = "Hello"

    event_3 = MagicMock()
    event_3.type = "response.completed"
    event_3.delta = None

    async def mock_stream():
        yield event_1
        yield event_2
        yield event_3

    provider.client.responses.create = AsyncMock(return_value=mock_stream())

    chunks = []

    async for text in provider.stream("Hello"):
        chunks.append(text)

    assert chunks == ["Hello"]


# ---------------------------------------------------------------------------
# _get_model
# ---------------------------------------------------------------------------


def test_get_model_returns_configured_model_for_openai(
    mock_settings,
):
    mock_settings.default_llm_provider = "openai"
    mock_settings.default_llm_model = "gpt-4o-mini"

    with (
        patch("rag_app.providers.openai_provider.AsyncOpenAI"),
        patch(
            "rag_app.providers.openai_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = OpenAIProvider()

    assert provider._get_model() == "gpt-4o-mini"


def test_get_model_returns_fallback_for_non_openai_provider(
    mock_settings,
):
    mock_settings.default_llm_provider = "anthropic"
    mock_settings.default_llm_model = "claude-sonnet"

    with (
        patch("rag_app.providers.openai_provider.AsyncOpenAI"),
        patch(
            "rag_app.providers.openai_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = OpenAIProvider()

    assert provider._get_model() == "gpt-4"
