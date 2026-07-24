import pytest

from ..providers.google import GoogleProvider


@pytest.mark.asyncio
async def test_openai_provider():

    provider = GoogleProvider()

    result = await provider.complete("Hello")

    assert result.provider == "google"
    assert result.text == "OpenAI response"