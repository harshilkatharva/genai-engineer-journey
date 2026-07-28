import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from mini_project_assignment.providers import AnthropicProvider


@pytest.mark.asyncio
@patch("mini_project_assignment.providers.anthropic_provider.anthropic.Anthropic")
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
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = AnthropicProvider()
    result = await provider.complete("Hello")

    assert result.provider == "anthropic"
    assert len(result.text) > 0
    assert result.latency_ms >= 0.0
