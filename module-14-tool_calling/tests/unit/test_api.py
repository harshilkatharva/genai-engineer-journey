from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from customer_support_agent.api import app


async def test_chat_tool_endpoint_rejects_empty_messages():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/chat_tool", json={"messages": []})
    assert response.status_code == 422
