import asyncio

from function_calling_library.models import (
    ChatMessage,
    LLMManagerRequest,
    LLMResponseModel,
    ToolCall,
)
from function_calling_library.services import LLMService


class FailingProvider:
    async def complete(self, messages, tools=None, tool_choice=None):
        error = RuntimeError("provider temporarily unavailable")
        error.status_code = 503
        error.code = "service_unavailable"
        raise error

    async def stream(self, messages):
        if False:
            yield ""


class SuccessfulProvider:
    def __init__(self, response: LLMResponseModel) -> None:
        self.response = response

    async def complete(self, messages, tools=None, tool_choice=None):
        return self.response

    async def stream(self, messages):
        if False:
            yield ""


def make_request(provider: str) -> LLMManagerRequest:
    return LLMManagerRequest(
        provider=provider,
        messages=[ChatMessage(role="user", content="Hello")],
    )


def test_successful_text_response_is_returned() -> None:
    provider_response = LLMResponseModel(
        text="Hello there.",
        model="test-model",
        latency_ms=12.5,
        input_tokens=4,
        output_tokens=3,
    )
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    response = asyncio.run(service.complete(make_request("test")))

    assert response.error is None
    assert response.text == "Hello there."
    assert response.tool_calls == []
    assert response.model == "test-model"
    assert response.usage == {"input_tokens": 4, "output_tokens": 3}


def test_successful_single_tool_call_is_returned() -> None:
    provider_response = LLMResponseModel(
        tool_calls=[ToolCall(id="call-1", name="lookup", arguments={"query": "weather"})],
        model="test-model",
        latency_ms=12.5,
    )
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    response = asyncio.run(service.complete(make_request("test")))

    assert response.error is None
    assert response.text is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "weather"}


def test_successful_multiple_tool_calls_preserve_order_and_ids() -> None:
    provider_response = LLMResponseModel(
        tool_calls=[
            ToolCall(id="call-1", name="lookup_weather", arguments={"city": "Paris"}),
            ToolCall(id="call-2", name="lookup_time", arguments={"timezone": "UTC"}),
        ],
        model="test-model",
        latency_ms=18.0,
        input_tokens=8,
        output_tokens=6,
    )
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    response = asyncio.run(service.complete(make_request("test")))

    assert response.error is None
    assert [call.id for call in response.tool_calls] == ["call-1", "call-2"]
    assert [call.name for call in response.tool_calls] == ["lookup_weather", "lookup_time"]
    assert [call.arguments for call in response.tool_calls] == [
        {"city": "Paris"},
        {"timezone": "UTC"},
    ]
    assert response.usage == {"input_tokens": 8, "output_tokens": 6}


def test_provider_failure_returns_structured_retryable_error() -> None:
    service = LLMService(providers={"test": FailingProvider()})

    response = asyncio.run(service.complete(make_request("test")))

    assert response.error is not None
    assert response.error.provider == "test"
    assert response.error.code == "service_unavailable"
    assert response.error.message == "provider temporarily unavailable"
    assert response.error.status_code == 503
    assert response.error.retryable is True


def test_unsupported_provider_returns_structured_error() -> None:
    service = LLMService(providers={})

    response = asyncio.run(service.complete(make_request("missing")))

    assert response.error is not None
    assert response.error.provider == "missing"
    assert response.error.code == "unsupported_provider"
    assert "missing" in response.error.message
