import pytest
import os

from llm_client.services.llm_service import LLMClient


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Set RUN_INTEGRATION_TESTS=1 to enable integration test",
    ),
]
pytest.mark.asyncio


async def test_google_provider():
    client = LLMClient()

    responce = await client.complete(provider="google", prompt="Say Hello")

    assert responce is not None
    assert len(responce.text) > 0
