import pytest

from llm_client.config import X_API_KEY


@pytest.mark.asyncio
async def test_chat_limit(api_client):
    provider = "openai"
    prompt = "hello"

    for i in range(10):
        response = await api_client.post(
            "/chat",
            headers={"x_api_key": X_API_KEY},
            json={"provider": provider, "prompt": prompt},
            follow_redirects=True,
        )

        assert response.status_code == 200 or response.status_code == 429


@pytest.mark.asyncio
async def test_chat_stream_limit(api_client):
    provider = "openai"
    prompt = "hello"

    for i in range(5):
        response = await api_client.post(
            "/chat/stream",
            headers={"x_api_key": X_API_KEY},
            json={"provider": provider, "prompt": prompt},
            follow_redirects=True,
        )

        assert response.status_code == 200 or response.status_code == 429
