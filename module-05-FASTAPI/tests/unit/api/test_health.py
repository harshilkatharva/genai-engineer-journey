import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_health_providers():
    with (
        patch("llm_client.services.healthcheck.OpenAI") as openai,
        patch("llm_client.services.healthcheck.genai.Client") as google,
        patch("llm_client.services.healthcheck.Anthropic") as anthropic,
    ):
        # OpenAI
        openai.return_value.responses.create.return_value = MagicMock()

        # Google
        google.return_value.interactions.create.return_value = MagicMock()

        # Anthropic
        anthropic.return_value.messages.create.return_value = MagicMock()

        yield {
            "openai": openai,
            "google": google,
            "anthropic": anthropic,
        }


async def test_health(api_client, mock_health_providers):
    response = await api_client.get("/health/")
    print(response.json())
    assert response.status_code == 200

    response = response.json()
    assert isinstance(response, dict)
