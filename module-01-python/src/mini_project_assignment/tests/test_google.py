import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mini_project_assignment.providers.google_provider import GoogleProvider


@pytest.mark.asyncio
@patch("mini_project_assignment.providers.google_provider.genai.Client")
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


