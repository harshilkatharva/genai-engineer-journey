import pytest
from unittest.mock import patch

from collections.abc import Generator
from unittest.mock import MagicMock


@pytest.fixture(autouse=True, scope="session")
def mock_providers() -> Generator[dict[str, MagicMock], None, None]:
    with (
        patch("llm_client.services.llm_service.OpenAIProvider") as openai,
        patch("llm_client.services.llm_service.GoogleProvider") as google,
        patch("llm_client.services.llm_service.AnthropicProvider") as anthropic,
    ):
        yield {"openai": openai, "google": google, "anthropic": anthropic}


@pytest.fixture(autouse=True, scope="session")
def mock_clients() -> Generator[dict[str, MagicMock], None, None]:
    with (
        patch("llm_client.services.providers.openai_provider.AsyncOpenAI") as openai,
        patch("llm_client.services.providers.google_provider.genai.Client") as google,
        patch("llm_client.services.providers.anthropic_provider.AsyncAnthropic") as anthropic,
    ):
        yield {"openai": openai, "google": google, "anthropic": anthropic}
