import pytest

from ..client import LLMClient


@pytest.mark.asyncio
async def test_client_complete():

    client = LLMClient()

    result = await client.complete(
        provider="openai",
        prompt="Hello"
    )

    assert result.provider == "openai"