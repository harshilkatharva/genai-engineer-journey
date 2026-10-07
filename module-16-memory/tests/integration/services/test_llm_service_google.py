from __future__ import annotations

import pytest

from memory_system.core import Settings
from memory_system.models import ChatMessage, LLMManagerRequest
from memory_system.providers.google_provider import GoogleProvider
from memory_system.services.llm_services import LLMService

SETTINGS = Settings()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SETTINGS.integration_test or not SETTINGS.google_api_key,
        reason="Set INTEGRATION_TEST=1 and GOOGLE_API_KEY to enable Google integration tests",
    ),
]


@pytest.mark.asyncio
async def test_llm_service_completes_with_google_provider() -> None:
    service = LLMService({"google": GoogleProvider()})
    response = await service.complete(
        LLMManagerRequest(
            provider="google",
            messages=[
                ChatMessage(
                    role="user",
                    content="Reply with the exact token integration-ok and no other words.",
                )
            ],
        )
    )

    assert response.error is None
    assert response.text is not None
    assert "integration-ok" in response.text.casefold()
    assert response.model
    assert response.usage["input_tokens"] > 0
