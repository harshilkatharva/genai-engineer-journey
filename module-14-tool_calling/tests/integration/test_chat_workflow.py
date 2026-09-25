from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from customer_support_agent.api import app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("INTEGRATION_TEST") != "1",
        reason="Set INTEGRATION_TEST=1 to run real provider and database integration tests",
    ),
]


async def test_chat_tool_endpoint_real_provider_and_database_workflow():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat_tool",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Please explain the current cancellation policy for my order. "
                            "Use the support policy database before answering."
                        ),
                    }
                ]
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["error"] is None
    assert payload["exhausted"] is False
    assert payload["iterations"] >= 1, "The real LLM did not invoke a real support tool"
    assert payload["text"].strip()
