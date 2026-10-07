from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from memory_system.models import (
    ChatMessage,
    LLMManagerRequest,
    LLMResponseModel,
    ToolCall,
)
from memory_system.services.llm_services import LLMService
from tests.fakes import make_settings


class FakeProvider:
    def __init__(self, response: LLMResponseModel | None = None) -> None:
        self.response = response or LLMResponseModel(model="fake", latency_ms=1)
        self.complete_calls: list[tuple[object, ...]] = []

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponseModel:
        self.complete_calls.append((messages, tools, tool_choice))
        return self.response

    async def stream(self, messages) -> AsyncIterator[str]:
        for token in ("one", "two"):
            yield token


def make_request(provider: str | None = "fake") -> LLMManagerRequest:
    return LLMManagerRequest(provider=provider, messages=[ChatMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_complete_maps_provider_response_and_uses_requested_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = LLMResponseModel(
        text="answer",
        tool_calls=[ToolCall(id="call-1", name="lookup", arguments={"id": 1})],
        model="fake-model",
        latency_ms=3,
        input_tokens=6,
        output_tokens=2,
        raw_response={"id": "response-1"},
    )
    provider = FakeProvider(response)
    service = LLMService({"fake": provider})
    monkeypatch.setattr(
        "memory_system.services.llm_services.get_settings",
        lambda: make_settings(default_llm_provider="fake"),
    )

    result = await service.complete(make_request(provider=None))

    assert result.text == "answer"
    assert result.tool_calls == response.tool_calls
    assert result.usage == {"input_tokens": 6, "output_tokens": 2}
    assert result.raw_response == {"id": "response-1"}
    assert len(provider.complete_calls) == 1
    failure = await service.complete(
        LLMManagerRequest(provider="missing", messages=[ChatMessage(role="user", content="hi")])
    )

    assert failure.error is not None
    assert failure.error.code == "unsupported_provider"
    assert failure.error.provider == "missing"


@pytest.mark.asyncio
async def test_complete_converts_provider_exceptions() -> None:
    class FailingProvider(FakeProvider):
        def __init__(self, error: Exception) -> None:
            super().__init__()
            self.error = error

        async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponseModel:
            raise self.error

    failures = [
        (RuntimeError("offline"), "RuntimeError", "offline", None, False),
        (RuntimeError("fallback"), "upstream", "try later", 429, True),
    ]
    for error, code, message, status, retryable in failures:
        if status is not None:
            error.status_code = status  # type: ignore[attr-defined]
            error.body = {"error": {"code": code, "message": message}}  # type: ignore[attr-defined]
        provider = FailingProvider(error)
        result = await LLMService({"fake": provider}).complete(make_request())

        assert result.error is not None
        assert result.error.code == code
        assert result.error.message == message
        assert result.error.status_code == status
        assert result.error.retryable is retryable


@pytest.mark.asyncio
async def test_stream_yields_provider_tokens_and_rejects_unknown_provider() -> None:
    service = LLMService({"fake": FakeProvider()})

    assert [token async for token in service.stream("fake", [])] == ["one", "two"]
    with pytest.raises(ValueError, match="Unsupported provider"):
        _ = [token async for token in service.stream("missing", [])]
