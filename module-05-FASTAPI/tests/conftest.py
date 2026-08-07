import asyncio
from collections.abc import AsyncIterator, Generator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_client.api.app import app
from llm_client.api.routes.chat import get_llm_client
from llm_client.api.routes.prompts import get_prompt_test
from llm_client.models import LLMResponseModel


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


class FakeLLMClient:
    async def complete(self, provider: str, prompt: str) -> LLMResponseModel:
        return LLMResponseModel(
            text="Hello form fake client", provider=provider, latency_ms=1, token_usage=0
        )

    async def stream(self, provider: str, prompt: str) -> AsyncIterator[str]:
        sentences = ["Hello", "How", "are", "you!"]

        for token in sentences:
            await asyncio.sleep(1)
            yield token


async def override_llm_client():
    return FakeLLMClient()


@pytest_asyncio.fixture(autouse=True, scope="session")
async def api_client():
    app.dependency_overrides[get_llm_client] = override_llm_client

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


class FakePromptTest:
    async def process(self, provider: str):
        pass

    def check_result(self):
        return {
            "Pass Percentage": 100,
            "Comparison Result": [],
        }


def override_prompt_test():
    return FakePromptTest()


@pytest_asyncio.fixture(autouse=True, scope="session")
async def api_client_prompt():
    app.dependency_overrides[get_prompt_test] = override_prompt_test

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
