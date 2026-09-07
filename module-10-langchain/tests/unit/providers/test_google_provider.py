from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rag_app.providers.google_provider import GoogleProvider


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.default_llm_provider = "google"
    settings.default_llm_model = "gemini-2.5-flash"
    return settings


@pytest.fixture
def provider(mock_settings):
    with (
        patch("rag_app.providers.google_provider.ChatGoogleGenerativeAI") as mock_llm,
        patch(
            "rag_app.providers.google_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = GoogleProvider()
        provider.llm = mock_llm.return_value

        yield provider


@pytest.mark.asyncio
async def test_complete_returns_llm_response(provider):
    response = MagicMock()

    response.text = "This is a Gemini response."
    response.model_version = "gemini-2.5-flash"

    response.content = [{"text": "This is a Gemini response."}]
    response.usage_metadata = {"input_tokens": 25, "output_tokens": 15}

    provider.llm.ainvoke = AsyncMock(return_value=response)

    result = await provider.complete("Explain RAG in simple terms.")

    assert result.text == "This is a Gemini response."
    assert result.model == "gemini-2.5-flash"
    assert result.input_tokens == 25
    assert result.output_tokens == 15
    assert result.latency_ms >= 0

    provider.llm.ainvoke.assert_awaited_once_with("Explain RAG in simple terms.")


@pytest.mark.asyncio
async def test_complete_uses_configured_model(mock_settings):
    mock_settings.default_llm_provider = "google"
    mock_settings.default_llm_model = "gemini-custom-model"

    with (
        patch("rag_app.providers.google_provider.ChatGoogleGenerativeAI") as mock_llm,
        patch(
            "rag_app.providers.google_provider.get_settings",
            return_value=mock_settings,
        ),
    ):
        provider = GoogleProvider()
        provider.llm = mock_llm.return_value

        response = MagicMock()
        response.text = "Response"
        response.model_version = "gemini-custom-model"
        response.content = [{"text": "Response"}]
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        provider.llm.ainvoke = AsyncMock(return_value=response)

        result = await provider.complete("Hello")

    assert result.model == "gemini-custom-model"

    provider.llm.ainvoke.assert_awaited_once_with("Hello")


@pytest.mark.asyncio
async def test_complete_returns_empty_text_when_google_response_has_no_text(
    provider,
):
    response = MagicMock()

    response.text = None
    response.model_version = "gemini-2.5-flash"

    response.content = []
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 0}

    provider.llm.ainvoke = AsyncMock(return_value=response)

    result = await provider.complete("Hello")

    assert result.text is None


# def make_client_error(code: int) -> ClientError:
#     error = object.__new__(ClientError)
#     error.code = code
#     return error

# @pytest.mark.asyncio
# async def test_complete_raises_authentication_error_for_401(provider):
#     error = make_client_error(401)

#     provider.client.aio.models.generate_content = AsyncMock(
#         side_effect=error
#     )

#     with pytest.raises(LLMAuthenticationError):
#         await provider.complete("Hello")

# @pytest.mark.asyncio
# async def test_complete_raises_authentication_error_for_403(provider):
#     error = MagicMock(spec=ClientError)
#     error.code = 403

#     provider.client.aio.models.generate_content = AsyncMock(
#         side_effect=error
#     )

#     with pytest.raises(LLMAuthenticationError):
#         await provider.complete("Hello")


# @pytest.mark.asyncio
# async def test_complete_raises_rate_limit_error_for_429(provider):
#     error = MagicMock(spec=ClientError)
#     error.code = 429

#     provider.client.aio.models.generate_content = AsyncMock(
#         side_effect=error
#     )

#     with pytest.raises(LLMRateLimitError):
#         await provider.complete("Hello")

# @pytest.mark.asyncio
# async def test_complete_raises_timeout_error_for_408(provider):
#     error = MagicMock(spec=ClientError)
#     error.code = 408

#     provider.client.aio.models.generate_content = AsyncMock(
#         side_effect=error
#     )

#     with pytest.raises(TimeoutError):
#         await provider.complete("Hello")

# @pytest.mark.asyncio
# async def test_complete_raises_llm_error_for_other_client_error(
#     provider,
# ):
#     error = MagicMock(spec=ClientError)
#     error.code = 400

#     provider.client.aio.models.generate_content = AsyncMock(
#         side_effect=error
#     )

#     with pytest.raises(LLMError):
#         await provider.complete("Hello")


# @pytest.mark.asyncio
# async def test_complete_raises_connection_error_for_server_error(
#     provider,
# ):
#     error = MagicMock(spec=ServerError)

#     provider.client.aio.models.generate_content = AsyncMock(
#         side_effect=error
#     )

#     with pytest.raises(LLMConnectionError):
#         await provider.complete("Hello")


# @pytest.mark.asyncio
# async def test_complete_raises_llm_error_for_api_error(provider):
#     error = MagicMock(spec=APIError)

#     provider.client.aio.models.generate_content = AsyncMock(
#         side_effect=error
#     )

#     with pytest.raises(LLMError):
#         await provider.complete("Hello")


@pytest.mark.asyncio
async def test_stream_returns_text_chunks(provider):
    chunk_1 = MagicMock()
    chunk_1.content = "Hello"

    chunk_2 = MagicMock()
    chunk_2.content = " world"

    chunk_3 = MagicMock()
    chunk_3.content = None

    async def mock_stream():
        yield chunk_1
        yield chunk_2
        yield chunk_3

    provider.llm.astream = MagicMock(return_value=mock_stream())

    chunks = []

    async for text in provider.stream("Say hello"):
        chunks.append(text)

    assert chunks == [
        "Hello",
        " world",
    ]

    provider.llm.astream.assert_called_once_with("Say hello")
