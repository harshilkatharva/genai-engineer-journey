import pytest

from customer_support_agent.models import ChatMessage, LLMManagerRequest, LLMResponseModel
from customer_support_agent.services.llm_services import LLMService


class FakeProvider:
    async def complete(self, messages, tools=None):
        return LLMResponseModel(
            model="fake", latency_ms=2, text="answer", input_tokens=3, output_tokens=4
        )

    async def stream(self, messages):
        yield "one"
        yield "two"


@pytest.mark.asyncio
async def test_llm_service_delegates_completion_and_maps_response():
    result = await LLMService({"fake": FakeProvider()}).complete(
        LLMManagerRequest(provider="fake", messages=[ChatMessage(role="user", content="hi")])
    )
    assert result.model == "fake"
    assert result.text == "answer"
    assert result.usage == {"input_tokens": 3, "output_tokens": 4}


@pytest.mark.asyncio
async def test_llm_service_streams_and_rejects_unknown_provider():
    service = LLMService({"fake": FakeProvider()})
    tokens = [token async for token in service.stream("fake", [])]
    assert tokens == ["one", "two"]
    with pytest.raises(ValueError, match="Unsupported provider"):
        await service.complete(LLMManagerRequest(provider="missing", messages=[]))
    with pytest.raises(ValueError, match="Unsupported provider"):
        [token async for token in service.stream("missing", [])]
