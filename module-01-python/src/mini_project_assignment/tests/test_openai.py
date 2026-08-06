from unittest.mock import patch, MagicMock, AsyncMock
from mini_project_assignment.providers import OpenAIProvider
import pytest


@pytest.mark.asyncio
@patch("mini_project_assignment.providers.openai_provider.AsyncOpenAI")
async def test_openai_provider_mock(mock_openai_client):
    mock_resposne = MagicMock()
    mock_resposne.output_text = "This response created for mock test OpenAI"
    mock_resposne.provider = "openai"
    mock_resposne.usage.total_tokens = 30
    mock_resposne.latency_ms = 303

    mock_client = mock_openai_client.return_value
    mock_client.responses.create = AsyncMock(return_value=mock_resposne)

    provider = OpenAIProvider()
    result = await provider.complete("Hello")

    assert result.provider == "openai"
    assert len(result.text) > 0
    assert result.latency_ms > 0.0
