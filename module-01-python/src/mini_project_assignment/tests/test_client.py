import pytest
from mini_project_assignment.models import CompletionResult
from mini_project_assignment.client import LLMClient
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
@patch("mini_project_assignment.client.OpenAIProvider")
async def test_client_complete_openai_mock(mock_openai_client):
    mock_response = CompletionResult(
        provider="openai",
        text="This mock response for client test",
        latency_ms=15.0,
        token_usage=10,
    )

    mock_client = mock_openai_client.return_value
    mock_client.complete = AsyncMock(return_value=mock_response)

    llm_client = LLMClient()
    result = await llm_client.complete(provider="openai", prompt="Hello")

    assert result.provider == "openai"


@pytest.mark.asyncio
@patch("mini_project_assignment.client.GoogleProvider")
async def test_client_complete_google_mock(mock_google_client):
    mock_response = CompletionResult(
        provider="google",
        text="This mock response for client test",
        latency_ms=15.0,
        token_usage=10,
    )

    mock_client = mock_google_client.return_value
    mock_client.complete = AsyncMock(return_value=mock_response)

    llm_client = LLMClient()
    result = await llm_client.complete(provider="google", prompt="Hello")

    assert result.provider == "google"


@pytest.mark.asyncio
@patch("mini_project_assignment.client.AnthropicProvider")
async def test_client_complete_anthropic_mock(mock_anthropic_client):
    mock_response = CompletionResult(
        provider="anthropic",
        text="This mock response for client test",
        latency_ms=15.0,
        token_usage=10,
    )

    mock_client = mock_anthropic_client.return_value
    mock_client.complete = AsyncMock(return_value=mock_response)

    llm_client = LLMClient()
    result = await llm_client.complete(provider="anthropic", prompt="Hello")

    assert result.provider == "anthropic"
