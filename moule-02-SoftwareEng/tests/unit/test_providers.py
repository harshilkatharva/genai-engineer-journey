import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from llm_client.services.providers import *


@pytest.mark.asyncio
@patch("llm_client.services.providers.openai_provider.AsyncOpenAI")
async def test_openai_provider_mock(mock_openai_client):
    mock_resposne = MagicMock()
    mock_resposne.output_text = "This response created for mock test OpenAI"
    mock_resposne.provider = "openai"
    mock_resposne.usage.total_tokens = 30
    mock_resposne.latency_ms = 303

    mock_client = mock_openai_client.return_value
    mock_client.responses.create = AsyncMock(return_value = mock_resposne)

    provider = OpenAIProvider()
    result = await provider.complete("Hello")

    assert result.provider == "openai"
    assert len(result.text) > 0
    assert result.latency_ms > 0.0


@pytest.mark.asyncio
@patch("llm_client.services.providers.google_provider.genai.Client")
async def test_google_provider_mock(mock_genai_client):

    mock_response = MagicMock()
    mock_response.text = "This response is created for mock test"
    mock_response.usage_metadata.total_token_count = 30
    mock_response.provider = "google"
    mock_response.latency_ms = 150

    mock_client = mock_genai_client.return_value
    mock_client.aio.models.generate_content = AsyncMock(return_value = mock_response)

    provider = GoogleProvider()
    result = await provider.complete("Hello")

    assert result.provider == "google"
    assert len(result.text) > 0
    assert result.latency_ms > 0.0


@pytest.mark.asyncio
@patch('llm_client.services.providers.anthropic_provider.AsyncAnthropic')
async def test_anthropic_provider_mock(mock_anthropic_client):
    mock_content_block = MagicMock()
    mock_content_block.text = "This response created for mock test"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 200

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    mock_response.usage = mock_usage

    mock_client = mock_anthropic_client.return_value
    mock_client.messages.create = AsyncMock(return_value= mock_response)

    provider = AnthropicProvider()
    result = await provider.complete("Hello")

    assert result.provider == "anthropic"
    assert len(result.text) > 0
    assert result.latency_ms >= 0.0





