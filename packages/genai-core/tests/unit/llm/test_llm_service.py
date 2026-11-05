from typing import Any, cast

import pytest
from pydantic import BaseModel

from genai_core.exceptions import LLMValidationError
from genai_core.models import (
    ChatMessage,
    LLMManagerRequest,
    LLMResponseModel,
    ToolCall,
)
from genai_core.services import LLMService


class CitySchema(BaseModel):
    city: str


class FailingProvider:
    async def complete_structure(
        self, prompt: str, response_schema=None, max_retries=None, model=None
    ):
        error: Any = RuntimeError("provider temporarily unavailable")
        error.status_code = 503
        error.code = "service_unavailable"
        raise error

    async def complete_tool(self, messages, tools=None, tool_choice=None, model=None):
        error: Any = RuntimeError("provider temporarily unavailable")
        error.status_code = 503
        error.code = "service_unavailable"
        raise error

    async def stream(self, messages, model=None):
        if False:
            yield ""


class ValidationFailingProvider:
    async def complete_structure(
        self, prompt: str, response_schema=None, max_retries=None, model=None
    ):
        raise LLMValidationError("Output failed schema validation after retries.")

    async def complete_tool(self, messages, tools=None, tool_choice=None, model=None):
        raise LLMValidationError("Tool output failed schema validation.")

    async def stream(self, messages, model=None):
        if False:
            yield ""


class SuccessfulProvider:
    def __init__(self, response: LLMResponseModel) -> None:
        self.response = response
        self.last_model: str | None = None

    async def complete_structure(
        self, prompt: str, response_schema=None, max_retries=None, model=None
    ):
        self.last_model = model
        resp = self.response.model_copy()
        if model:
            resp.model = model
        return resp

    async def complete_tool(self, messages, tools=None, tool_choice=None, model=None):
        self.last_model = model
        resp = self.response.model_copy()
        if model:
            resp.model = model
        return resp

    async def stream(self, messages, model=None):
        self.last_model = model
        yield "token1"
        yield "token2"


class LegacyProviderWithoutModel:
    """Simulates a third-party custom provider whose complete_* methods don't accept model."""

    async def complete_structure(self, prompt: str, response_schema=None):
        return LLMResponseModel(text="legacy_structured", model="legacy-model", latency_ms=1.0)

    async def complete_tool(self, messages, tools=None, tool_choice=None):
        return LLMResponseModel(text="legacy_tool", model="legacy-model", latency_ms=1.0)

    async def stream(self, messages):
        yield "legacy_chunk"


def make_request(provider: str) -> LLMManagerRequest:
    return LLMManagerRequest(
        provider=provider,
        messages=[ChatMessage(role="user", content="Hello")],
    )


@pytest.mark.asyncio
async def test_successful_text_response_is_returned() -> None:
    provider_response = LLMResponseModel(
        text="Hello there.",
        model="test-model",
        latency_ms=12.5,
        input_tokens=4,
        output_tokens=3,
    )
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    response = await service.complete(make_request("test"))

    assert response.error is None
    assert response.text == "Hello there."
    assert response.tool_calls == []
    assert response.model == "test-model"
    assert response.usage == {"input_tokens": 4, "output_tokens": 3}


@pytest.mark.asyncio
async def test_successful_structured_output_response() -> None:
    schema_instance = CitySchema(city="Kyoto")
    provider_response = LLMResponseModel(
        text='{"city":"Kyoto"}',
        structured_response=schema_instance,
        model="test-model",
        latency_ms=10.0,
    )
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    request = LLMManagerRequest(
        provider="test",
        prompt="Get city",
        response_schema=CitySchema,
    )
    response = await service.complete(request)

    assert response.error is None
    assert response.structured_response == schema_instance
    assert response.structured_response.city == "Kyoto"
    assert response.text == '{"city":"Kyoto"}'


@pytest.mark.asyncio
async def test_successful_single_tool_call_is_returned() -> None:
    provider_response = LLMResponseModel(
        tool_calls=[ToolCall(id="call-1", name="lookup", arguments={"query": "weather"})],
        model="test-model",
        latency_ms=12.5,
    )
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    response = await service.complete(make_request("test"))

    assert response.error is None
    assert response.text is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "weather"}


@pytest.mark.asyncio
async def test_successful_multiple_tool_calls_preserve_order_and_ids() -> None:
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

    response = await service.complete(make_request("test"))

    assert response.error is None
    assert [call.id for call in response.tool_calls] == ["call-1", "call-2"]
    assert [call.name for call in response.tool_calls] == [
        "lookup_weather",
        "lookup_time",
    ]


@pytest.mark.asyncio
async def test_unsupported_provider_returns_error_response() -> None:
    service = LLMService(
        providers={
            "google": SuccessfulProvider(LLMResponseModel(text="hi", model="m", latency_ms=1.0))
        }
    )

    response = await service.complete(make_request("unsupported"))

    assert response.error is not None
    assert response.error.code == "unsupported_provider"
    assert "Unsupported LLM provider" in response.error.message


@pytest.mark.asyncio
async def test_failing_provider_maps_error_response() -> None:
    service = LLMService(providers={"test": FailingProvider()})

    response = await service.complete(make_request("test"))

    assert response.error is not None
    assert response.error.code == "service_unavailable"
    assert response.error.status_code == 503
    assert response.error.retryable is True


@pytest.mark.asyncio
async def test_validation_failing_provider_maps_validation_error() -> None:
    service = LLMService(providers={"test": ValidationFailingProvider()})

    request = LLMManagerRequest(
        provider="test",
        prompt="Get city",
        response_schema=CitySchema,
    )
    response = await service.complete(request)

    assert response.error is not None
    assert response.error.code == "validation_error"
    assert response.error.status_code == 422
    assert response.error.retryable is False
    assert "Output failed schema validation" in response.error.message


@pytest.mark.asyncio
async def test_complete_structure_convenience_method() -> None:
    schema_instance = CitySchema(city="Rome")
    provider_response = LLMResponseModel(
        text='{"city":"Rome"}',
        structured_response=schema_instance,
        model="test-model",
        latency_ms=5.0,
    )
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    response = await service.complete_structure(
        "City query", response_schema=CitySchema, provider="test"
    )

    assert response.error is None
    assert response.structured_response == schema_instance


@pytest.mark.asyncio
async def test_complete_tool_convenience_method() -> None:
    provider_response = LLMResponseModel(
        tool_calls=[ToolCall(id="c1", name="test_tool", arguments={})],
        model="test-model",
        latency_ms=5.0,
    )
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    response = await service.complete_tool(
        messages=[ChatMessage(role="user", content="run tool")],
        provider="test",
    )

    assert response.error is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "test_tool"


@pytest.mark.asyncio
async def test_complete_all_runs_concurrently() -> None:
    p1 = SuccessfulProvider(LLMResponseModel(text="res1", model="m1", latency_ms=1.0))
    p2 = SuccessfulProvider(LLMResponseModel(text="res2", model="m2", latency_ms=1.0))
    service = LLMService(providers={"p1": p1, "p2": p2})

    results = await service.complete_all(prompt="Hello")

    assert len(results) == 2
    assert all(isinstance(r, LLMResponseModel) for r in results)


@pytest.mark.asyncio
async def test_stream_delegates_to_provider() -> None:
    provider_response = LLMResponseModel(text="", model="test-model", latency_ms=1.0)
    service = LLMService(providers={"test": SuccessfulProvider(provider_response)})

    chunks = []
    async for chunk in service.stream("test", [ChatMessage(role="user", content="hi")]):
        chunks.append(chunk)

    assert "".join(chunks) == "token1token2"


@pytest.mark.asyncio
async def test_stream_unsupported_provider_raises_value_error() -> None:
    service = LLMService(providers={})
    with pytest.raises(ValueError, match="Unsupported provider"):
        async for _ in service.stream("unknown", "hello"):
            pass


@pytest.mark.asyncio
async def test_llm_service_initialization_with_model() -> None:
    provider_response = LLMResponseModel(text="test", model="default-model", latency_ms=1.0)
    provider = SuccessfulProvider(provider_response)
    service = LLMService(
        providers={"test": provider},
        default_provider="test",
        model="custom-init-model",
    )

    assert service.default_model == "custom-init-model"

    response = await service.complete_structure(prompt="Hello")
    assert response.error is None
    assert provider.last_model == "custom-init-model"
    assert response.model == "custom-init-model"


@pytest.mark.asyncio
async def test_complete_structure_with_per_call_model_override() -> None:
    provider_response = LLMResponseModel(text="test", model="default-model", latency_ms=1.0)
    provider = SuccessfulProvider(provider_response)
    service = LLMService(
        providers={"test": provider},
        default_provider="test",
        default_model="service-default-model",
    )

    # Per-call model override must take precedence over service default
    response = await service.complete_structure(prompt="Hello", model="override-structure-model")
    assert response.error is None
    assert provider.last_model == "override-structure-model"
    assert response.model == "override-structure-model"


@pytest.mark.asyncio
async def test_complete_tool_with_per_call_model_override() -> None:
    provider_response = LLMResponseModel(
        tool_calls=[ToolCall(id="c1", name="search", arguments={})],
        model="default-model",
        latency_ms=1.0,
    )
    provider = SuccessfulProvider(provider_response)
    service = LLMService(
        providers={"test": provider},
        default_provider="test",
        model="service-default-model",
    )

    response = await service.complete_tool(
        messages=[ChatMessage(role="user", content="find info")],
        model="override-tool-model",
    )
    assert response.error is None
    assert provider.last_model == "override-tool-model"
    assert response.model == "override-tool-model"


@pytest.mark.asyncio
async def test_complete_with_request_model() -> None:
    provider_response = LLMResponseModel(text="test", model="default-model", latency_ms=1.0)
    provider = SuccessfulProvider(provider_response)
    service = LLMService(providers={"test": provider}, default_provider="test")

    req = LLMManagerRequest(
        provider="test",
        model="explicit-request-model",
        prompt="Hi",
    )
    response = await service.complete(req)
    assert response.error is None
    assert provider.last_model == "explicit-request-model"
    assert response.model == "explicit-request-model"


@pytest.mark.asyncio
async def test_stream_with_model_override() -> None:
    provider_response = LLMResponseModel(text="", model="default-model", latency_ms=1.0)
    provider = SuccessfulProvider(provider_response)
    service = LLMService(providers={"test": provider}, default_provider="test")

    chunks = []
    async for chunk in service.stream("test", "hi", model="stream-override-model"):
        chunks.append(chunk)

    assert provider.last_model == "stream-override-model"
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_legacy_provider_without_model_parameter_fallback() -> None:
    legacy_prov = LegacyProviderWithoutModel()
    service = LLMService(
        providers={"legacy": cast(Any, legacy_prov)},
        default_provider="legacy",
        model="ignored-by-legacy",
    )

    # complete_structure fallback
    resp1 = await service.complete_structure(
        prompt="Hi", response_schema=CitySchema, model="new-model"
    )
    assert resp1.error is None
    assert resp1.text == "legacy_structured"

    # complete_tool fallback
    resp2 = await service.complete_tool(
        messages=[ChatMessage(role="user", content="Hi")], model="new-model"
    )
    assert resp2.error is None
    assert resp2.text == "legacy_tool"

    # stream fallback
    chunks = []
    async for c in service.stream("legacy", "Hi", model="new-model"):
        chunks.append(c)
    assert chunks == ["legacy_chunk"]
