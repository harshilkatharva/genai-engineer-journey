from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from memory_system.models import ChatMessage, ToolChoice, ToolSpec
from memory_system.providers import anthropic_provider, google_provider, openai_provider
from memory_system.providers.anthropic_provider import AnthropicProvider
from memory_system.providers.google_provider import GoogleProvider
from memory_system.providers.openai_provider import OpenAIProvider
from tests.fakes import make_settings


class Arguments(BaseModel):
    value: int


def make_tool() -> ToolSpec:
    return ToolSpec(name="sample", description="A sample tool", argument_model=Arguments)


def async_stream(*values: object) -> AsyncIterator[object]:
    async def generate() -> AsyncIterator[object]:
        for value in values:
            yield value

    return generate()


def test_provider_constructors_select_compatible_models(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(default_llm_model="custom-model", default_llm_temperature=0)
    monkeypatch.setattr(openai_provider, "get_settings", lambda: settings)
    monkeypatch.setattr(anthropic_provider, "get_settings", lambda: settings)
    monkeypatch.setattr(google_provider, "get_settings", lambda: settings)

    assert OpenAIProvider(client=cast(Any, object())).model == "gpt-4o-mini"
    assert AnthropicProvider(client=cast(Any, object())).model == "claude-3-5-sonnet-20241022"
    google = GoogleProvider(client=cast(Any, object()))
    assert google.model == "gemini-3.5-flash-lite"
    assert google.temperature == pytest.approx(0.2)

    compatible_settings = make_settings(default_llm_model="gpt-test", default_llm_temperature=0.7)
    monkeypatch.setattr(openai_provider, "get_settings", lambda: compatible_settings)
    assert OpenAIProvider(client=cast(Any, object())).model == "gpt-test"


@pytest.mark.asyncio
async def test_openai_complete_maps_text_tools_and_usage() -> None:
    response = SimpleNamespace(
        output_text="ignored when tools exist",
        output=[
            SimpleNamespace(
                type="function_call", arguments='{"value": 3}', call_id="call-1", name="sample"
            ),
            SimpleNamespace(
                type="function_call", arguments="not-json", call_id="call-2", name="sample"
            ),
            SimpleNamespace(type="message"),
        ],
        model="gpt-test",
        usage=SimpleNamespace(input_tokens=8, output_tokens=5),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=response)))
    provider = OpenAIProvider(client=cast(Any, client))

    result = await provider.complete(
        [ChatMessage(role="user", content="run")], [make_tool()], ToolChoice(mode="any")
    )

    assert result.text is None
    assert [call.arguments for call in result.tool_calls] == [{"value": 3}, "not-json"]
    assert result.model == "gpt-test"
    assert (result.input_tokens, result.output_tokens) == (8, 5)
    request = client.responses.create.await_args.kwargs
    assert request["tools"][0]["name"] == "sample"
    assert request["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_openai_complete_text_without_usage_and_stream_deltas() -> None:
    response = SimpleNamespace(output_text="answer", output=[], model="gpt-test", usage=None)
    events = [
        openai_provider.ResponseTextDeltaEvent.model_construct(delta="one"),
        SimpleNamespace(type="other"),
        openai_provider.ResponseTextDeltaEvent.model_construct(delta="two"),
    ]
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=async_stream(*events))
    stream.__aexit__ = AsyncMock(return_value=None)
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(return_value=response), stream=MagicMock(return_value=stream)
        )
    )
    provider = OpenAIProvider(client=cast(Any, client))

    result = await provider.complete([ChatMessage(role="user", content="hello")])
    tokens = [token async for token in provider.stream([ChatMessage(role="user", content="hello")])]

    assert result.text == "answer"
    assert (result.input_tokens, result.output_tokens) == (0, 0)
    assert tokens == ["one", "two"]


@pytest.mark.asyncio
async def test_anthropic_complete_separates_system_and_maps_text_and_tools() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="tool_use", id="tool-1", name="sample", input={"value": 4}),
            SimpleNamespace(type="text", text="there"),
        ],
        model="claude-test",
        usage=SimpleNamespace(input_tokens=11, output_tokens=6),
    )
    messages_api = SimpleNamespace(create=AsyncMock(return_value=response))
    client = SimpleNamespace(messages=messages_api)
    provider = AnthropicProvider(client=cast(Any, client))

    result = await provider.complete(
        [
            ChatMessage(role="system", content="system one"),
            ChatMessage(role="user", content="question"),
            ChatMessage(role="system", content="system two"),
        ],
        [make_tool()],
        ToolChoice(mode="specific", tool_name="sample"),
    )

    assert result.text == "Hello there"
    assert result.tool_calls[0].arguments == {"value": 4}
    request = messages_api.create.await_args.kwargs
    assert request["system"] == "system one\nsystem two"
    assert request["messages"] == [{"role": "user", "content": "question"}]
    assert request["tool_choice"] == {"type": "tool", "name": "sample"}


@pytest.mark.asyncio
async def test_anthropic_text_response_and_stream() -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="text only")],
        model="claude-test",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=SimpleNamespace(text_stream=async_stream("a", "b")))
    stream.__aexit__ = AsyncMock(return_value=None)
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=response), stream=MagicMock(return_value=stream)
        )
    )
    provider = AnthropicProvider(client=cast(Any, client))

    result = await provider.complete([ChatMessage(role="user", content="hello")])
    tokens = [token async for token in provider.stream([ChatMessage(role="user", content="hello")])]

    assert result.text == "text only"
    assert tokens == ["a", "b"]


@pytest.mark.asyncio
async def test_google_complete_maps_text_tool_calls_and_usage() -> None:
    function_call = SimpleNamespace(name="sample", args={"value": 5})
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(function_call=function_call)])
            ),
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(function_call=None)])),
        ],
        text="ignored with tool calls",
        usage_metadata=SimpleNamespace(prompt_token_count=13, candidates_token_count=7),
    )
    models = SimpleNamespace(generate_content=AsyncMock(return_value=response))
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    provider = GoogleProvider(client=cast(Any, client))

    result = await provider.complete(
        [ChatMessage(role="assistant", content="prior"), ChatMessage(role="user", content="go")],
        [make_tool()],
        ToolChoice(mode="specific", tool_name="sample"),
    )

    assert result.text is None
    assert result.tool_calls[0].id == "google-1"
    assert result.tool_calls[0].arguments == {"value": 5}
    assert (result.input_tokens, result.output_tokens) == (13, 7)
    request = models.generate_content.await_args.kwargs
    assert [content.role for content in request["contents"]] == ["model", "user"]
    assert request["config"].tool_config.function_calling_config.allowed_function_names == [
        "sample"
    ]


@pytest.mark.asyncio
async def test_google_text_response_and_stream_ignore_empty_chunks() -> None:
    response = SimpleNamespace(candidates=None, text="answer", usage_metadata=None)
    models = SimpleNamespace(
        generate_content=AsyncMock(return_value=response),
        generate_content_stream=AsyncMock(
            return_value=async_stream(
                SimpleNamespace(text="first"),
                SimpleNamespace(text=""),
                SimpleNamespace(text="last"),
            )
        ),
    )
    provider = GoogleProvider(client=cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models))))

    result = await provider.complete([ChatMessage(role="user", content="hello")])
    tokens = [token async for token in provider.stream([ChatMessage(role="user", content="hello")])]

    assert result.text == "answer"
    assert (result.input_tokens, result.output_tokens) == (0, 0)
    assert tokens == ["first", "last"]
