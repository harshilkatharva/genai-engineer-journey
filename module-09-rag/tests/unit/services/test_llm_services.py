from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_app.exceptions.llm_exceptions import LLMError
from rag_app.models import (
    LLMManagerRequest,
    LLMManagerResponse,
    LLMResponseModel,
)
from rag_app.services.llm_services import LLMServicemanager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_providers():
    openai = MagicMock()
    anthropic = MagicMock()
    google = MagicMock()

    return {
        "openai": openai,
        "anthropic": anthropic,
        "google": google,
    }


@pytest.fixture
def service(mock_providers):
    with (
        patch(
            "rag_app.services.llm_services.OpenAIProvider",
            return_value=mock_providers["openai"],
        ),
        patch(
            "rag_app.services.llm_services.AnthropicProvider",
            return_value=mock_providers["anthropic"],
        ),
        patch(
            "rag_app.services.llm_services.GoogleProvider",
            return_value=mock_providers["google"],
        ),
    ):
        service = LLMServicemanager()

        yield service


# ---------------------------------------------------------------------------
# complete - success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_successful_response(
    service,
    mock_providers,
):
    provider_response = LLMResponseModel(
        text="This is an OpenAI response.",
        model="gpt-4o-mini",
        latency_ms=120.5,
        input_tokens=20,
        output_tokens=10,
    )

    mock_providers["openai"].complete = AsyncMock(return_value=provider_response)

    request = LLMManagerRequest(
        provider="openai",
        prompt="Explain RAG.",
    )

    result = await service.complete(request)

    assert isinstance(result, LLMManagerResponse)
    assert result.text == "This is an OpenAI response."

    mock_providers["openai"].complete.assert_awaited_once_with("Explain RAG.")


# ---------------------------------------------------------------------------
# complete - provider selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_uses_requested_provider(
    service,
    mock_providers,
):
    provider_response = LLMResponseModel(
        text="Gemini response",
        model="gemini-2.5-flash",
        latency_ms=100,
        input_tokens=10,
        output_tokens=5,
    )

    mock_providers["google"].complete = AsyncMock(return_value=provider_response)

    request = LLMManagerRequest(
        provider="google",
        prompt="Explain embeddings.",
    )

    result = await service.complete(request)

    assert result.text == "Gemini response"

    mock_providers["google"].complete.assert_awaited_once_with("Explain embeddings.")

    mock_providers["openai"].complete.assert_not_called()
    mock_providers["anthropic"].complete.assert_not_called()


# ---------------------------------------------------------------------------
# complete - unsupported provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_raises_error_for_unsupported_provider(
    service,
):
    request = LLMManagerRequest(
        provider="unsupported",
        prompt="Hello",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported provider unsupported",
    ):
        await service.complete(request)


# ---------------------------------------------------------------------------
# complete - LLMError handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_user_error_message_when_provider_fails(
    service,
    mock_providers,
):
    error = LLMError("Internal provider error")

    mock_providers["openai"].complete = AsyncMock(side_effect=error)

    request = LLMManagerRequest(
        provider="openai",
        prompt="Explain RAG.",
    )

    result = await service.complete(request)

    assert isinstance(result, LLMManagerResponse)
    assert result.text == error.user_message

    mock_providers["openai"].complete.assert_awaited_once_with("Explain RAG.")


# ---------------------------------------------------------------------------
# complete_all - success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_all_calls_all_providers(
    service,
    mock_providers,
):
    openai_response = LLMResponseModel(
        text="OpenAI response",
        model="gpt-4o-mini",
        latency_ms=100,
        input_tokens=10,
        output_tokens=5,
    )

    anthropic_response = LLMResponseModel(
        text="Anthropic response",
        model="claude-sonnet",
        latency_ms=120,
        input_tokens=12,
        output_tokens=6,
    )

    google_response = LLMResponseModel(
        text="Google response",
        model="gemini-2.5-flash",
        latency_ms=110,
        input_tokens=11,
        output_tokens=7,
    )

    mock_providers["openai"].complete = AsyncMock(return_value=openai_response)

    mock_providers["anthropic"].complete = AsyncMock(return_value=anthropic_response)

    mock_providers["google"].complete = AsyncMock(return_value=google_response)

    results = await service.complete_all("Explain RAG.")

    assert len(results) == 3

    assert results[0] == openai_response
    assert results[1] == anthropic_response
    assert results[2] == google_response

    mock_providers["openai"].complete.assert_awaited_once_with("Explain RAG.")

    mock_providers["anthropic"].complete.assert_awaited_once_with("Explain RAG.")

    mock_providers["google"].complete.assert_awaited_once_with("Explain RAG.")


# ---------------------------------------------------------------------------
# complete_all - provider failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_all_returns_exception_without_stopping_other_providers(
    service,
    mock_providers,
):
    openai_response = LLMResponseModel(
        text="OpenAI response",
        model="gpt-4o-mini",
        latency_ms=100,
        input_tokens=10,
        output_tokens=5,
    )

    google_response = LLMResponseModel(
        text="Google response",
        model="gemini-2.5-flash",
        latency_ms=110,
        input_tokens=11,
        output_tokens=7,
    )

    error = LLMError("Anthropic failed")

    mock_providers["openai"].complete = AsyncMock(return_value=openai_response)

    mock_providers["anthropic"].complete = AsyncMock(side_effect=error)

    mock_providers["google"].complete = AsyncMock(return_value=google_response)

    results = await service.complete_all("Explain RAG.")

    assert len(results) == 3

    assert results[0] == openai_response

    assert isinstance(results[1], LLMError)
    assert str(results[1]) == "Anthropic failed"

    assert results[2] == google_response

    mock_providers["openai"].complete.assert_awaited_once()
    mock_providers["anthropic"].complete.assert_awaited_once()
    mock_providers["google"].complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# complete_all - all providers fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_all_returns_all_exceptions_when_all_fail(
    service,
    mock_providers,
):
    openai_error = LLMError("OpenAI failed")
    anthropic_error = LLMError("Anthropic failed")
    google_error = LLMError("Google failed")

    mock_providers["openai"].complete = AsyncMock(side_effect=openai_error)

    mock_providers["anthropic"].complete = AsyncMock(side_effect=anthropic_error)

    mock_providers["google"].complete = AsyncMock(side_effect=google_error)

    results = await service.complete_all("Hello")

    assert len(results) == 3

    assert results[0] is openai_error
    assert results[1] is anthropic_error
    assert results[2] is google_error


# ---------------------------------------------------------------------------
# stream - success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_returns_provider_tokens(
    service,
    mock_providers,
):
    async def mock_stream(prompt):
        yield "Hello"
        yield " world"
        yield "!"

    mock_providers["openai"].stream = mock_stream

    chunks = []

    async for token in service.stream(
        provider="openai",
        prompt="Say hello",
    ):
        chunks.append(token)

    assert chunks == [
        "Hello",
        " world",
        "!",
    ]


# ---------------------------------------------------------------------------
# stream - selected provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_uses_requested_provider(
    service,
    mock_providers,
):
    async def google_stream(prompt):
        yield "Google"
        yield " response"

    async def openai_stream(prompt):
        yield "OpenAI"
        yield " response"

    mock_providers["google"].stream = google_stream
    mock_providers["openai"].stream = openai_stream

    chunks = []

    async for token in service.stream(
        provider="google",
        prompt="Hello",
    ):
        chunks.append(token)

    assert chunks == [
        "Google",
        " response",
    ]


# ---------------------------------------------------------------------------
# stream - unsupported provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_raises_key_error_for_unsupported_provider(
    service,
):
    with pytest.raises(
        ValueError,
        match="Unsupported provider unsupported",
    ):
        async for token in service.stream(
            provider="unsupported",
            prompt="Hello",
        ):
            pass
