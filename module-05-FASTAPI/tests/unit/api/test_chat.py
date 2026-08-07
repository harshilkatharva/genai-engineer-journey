import pytest
from llm_client.config import X_API_KEY


@pytest.mark.asyncio
async def test_chat_unauthorized(api_client):
    provider = "openai"
    prompt = "hello"

    response = await api_client.post(
        "/chat", json={"provider": provider, "prompt": prompt}, follow_redirects=True
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat(api_client):
    provider = "openai"
    prompt = "hello"

    response = await api_client.post(
        "/chat",
        headers={"x_api_key": X_API_KEY},
        json={"provider": provider, "prompt": prompt},
        follow_redirects=True,
    )
    assert response.status_code == 200
    response = response.json()
    assert response["provider"] == provider


@pytest.mark.asyncio
async def test_chat_stream_unauthorized(api_client):
    provider = "openai"
    prompt = "hello"

    async with api_client.stream(
        "POST",
        "/chat/stream",
        json={"provider": provider, "prompt": prompt},
    ) as response:
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_stream(api_client):
    provider = "openai"
    prompt = "hello"

    async with api_client.stream(
        "POST",
        "/chat/stream",
        headers={"x_api_key": X_API_KEY},
        json={"provider": provider, "prompt": prompt},
    ) as response:
        assert response.status_code == 200
        chunks = []
        async for chunk in response.aiter_text():
            chunks.append(chunk)

    assert "data:" in chunks[0]
