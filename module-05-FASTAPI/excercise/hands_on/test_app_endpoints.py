import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from hands_on.tasks import app, get_llm_services

client = TestClient(app)


def test_chat_without_key():
    response = client.post("/chat_auth", json={"provider": "openai", "query": "What is ai"})

    assert response.status_code == 401


def test_chat_wrong_key():
    response = client.post(
        "/chat_auth",
        headers={"API-KEY": "wrong-key"},
        json={"provider": "openai", "query": "What is ai"},
    )

    assert response.status_code == 401


def test_chat_valid_key():
    response = client.post(
        "/chat_auth",
        headers={"API-KEY": "abcusbcduvdsvdcv"},
        json={"provider": "openai", "query": "What is ai"},
    )

    assert response.status_code == 200


class FakeLLMServices:
    async def complete(self, provider, query):
        return "Mock response"


@pytest.mark.asyncio
async def test_override_chat():
    app.dependency_overrides[get_llm_services] = lambda: FakeLLMServices()

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat_overrideable",
            json={"provider": "openai", "query": "Hello"},
        )

    assert response.status_code == 200
