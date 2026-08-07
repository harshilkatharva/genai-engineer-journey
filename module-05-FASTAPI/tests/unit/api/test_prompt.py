import pytest

from llm_client.config import X_API_KEY


@pytest.mark.asyncio
async def test_prompt(api_client_prompt):
    response = await api_client_prompt.post(
        "/prompts/test", headers={"x_api_key": X_API_KEY}, json={"provider": "google"}
    )
    print(response.json())
    assert response.status_code == 200
