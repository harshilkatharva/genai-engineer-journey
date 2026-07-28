import pytest
from mini_project_assignment.client import LLMClient
from unittest.mock import AsyncMock
from mini_project_assignment.models import CompletionResult


@pytest.mark.asyncio
async def test_complete_all_mock():
    openai_res = CompletionResult(
        provider="openai",
        text="OpenAI mock response",
        latency_ms=300,
        token_usage=303,
    )
    google_res = CompletionResult(
        provider="google",
        text="Google mock response",
        latency_ms=300,
        token_usage=303,
    )

    llm_client = LLMClient()

    llm_client.providers["openai"].complete = AsyncMock(return_value=openai_res)
    llm_client.providers["google"].complete = AsyncMock(return_value=google_res)
    llm_client.providers["anthropic"].complete = AsyncMock(
        side_effect=Exception("Anthropic server is down!")
    )

    result = await llm_client.complete_all("Hello, Who are you ?")
    errors = [r for r in result if isinstance(r, Exception)]
    assert len(result) == 3
    assert len(errors) == 1
