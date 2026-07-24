import pytest

from ..providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_provider():

    provider = OpenAIProvider()

    result = await provider.complete("Hello")

    assert result.provider == "openai"
    assert result.text == "OpenAI response"