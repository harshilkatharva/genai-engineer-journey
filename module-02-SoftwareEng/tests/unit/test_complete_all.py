from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from llm_client.models import CompletionResult
from llm_client.services import LLMClient


@pytest.mark.asyncio
async def test_complete_all_mock() -> None:
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

    openai = cast(Any, llm_client.providers["openai"])
    google = cast(Any, llm_client.providers["google"])
    anthropic = cast(Any, llm_client.providers["anthropic"])

    openai.complete = AsyncMock(return_value=openai_res)
    google.complete = AsyncMock(return_value=google_res)
    anthropic.complete = AsyncMock(side_effect=Exception("Anthropic server is down!"))

    result = await llm_client.complete_all("Hello, Who are you ?")
    errors = [r for r in result if isinstance(r, Exception)]
    assert len(result) == 3
    assert len(errors) == 1
