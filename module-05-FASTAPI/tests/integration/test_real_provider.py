import os

import pytest

from llm_client.services.llm_service import LLMClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Set RUN_INTEGRATION_TESTS=1 to enable integration test",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "google", "anthropic"])
async def test_all_provider_real(provider: str) -> None:
    client = LLMClient()

    responce = await client.complete(provider=provider, prompt="Say Hello")

    assert responce is not None
    assert len(responce.text) > 0
