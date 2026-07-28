import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from llm_client.models import CompletionResult
from llm_client.services import LLMClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,patch_target",
    [
        ("openai", "llm_client.services.llm_service.OpenAIProvider"),
        ("google", "llm_client.services.llm_service.GoogleProvider"),
        ("anthropic", "llm_client.services.llm_service.AnthropicProvider"),
    ],
)
async def test_client_complete(provider, patch_target):
    with patch(patch_target) as mock_provider:
        mock_response = CompletionResult(
            provider=provider,
            text=f"This mock response for client test of {provider}",
            latency_ms=15.0,
            token_usage=10,
        )

        mock_client = mock_provider.return_value
        mock_client.complete = AsyncMock(return_value=mock_response)

        llm_client = LLMClient()
        result = await llm_client.complete(provider=provider, prompt="Hello")

        assert result.provider == provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider, patch_target",
    [
        ("openai", "llm_client.services.llm_service.OpenAIProvider"),
        ("google", "llm_client.services.llm_service.GoogleProvider"),
        ("anthropic", "llm_client.services.llm_service.AnthropicProvider"),
    ],
)
async def test_client_stream(provider, patch_target):

    with patch(patch_target) as mock_provider:

        async def fake_text_stream(prompt):
            sent = "Hello How are you?"
            for word in sent.split():
                await asyncio.sleep(0.1)
                yield word + " "

        mock_client = mock_provider.return_value
        mock_client.stream = fake_text_stream

        llm_client = LLMClient()
        chunks = []

        async for chunk in llm_client.stream(provider=provider, prompt="Hello"):
            chunks.append(chunk)

        assert len(chunks) >= 2
