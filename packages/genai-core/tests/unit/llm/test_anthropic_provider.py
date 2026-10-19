from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from genai_core.exceptions import LLMValidationError
from genai_core.llm import AnthropicProvider
from genai_core.models import ChatMessage, ToolChoice, ToolSpec


class CityArguments(BaseModel):
    city: str


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if "tools" in kwargs:
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="call-anthropic-1",
                        name="lookup_weather",
                        input={"city": "Paris"},
                    )
                ],
                model="claude-3-5-sonnet-20241022",
                usage=SimpleNamespace(input_tokens=12, output_tokens=6),
            )
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text='{"city":"London"}',
                )
            ],
            model="claude-3-5-sonnet-20241022",
            usage=SimpleNamespace(input_tokens=20, output_tokens=10),
        )

    def stream(self, **kwargs: Any):
        return FakeAnthropicStreamContext()


class SequentialAnthropicMessages:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            model="claude-3-5-sonnet-20241022",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


class FakeAnthropicStreamContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def text_stream(self):
        async def _gen():
            yield "Hello"
            yield " from Claude"

        return _gen()


class FakeAnthropicClient:
    def __init__(self, messages: Any = None) -> None:
        self.messages = messages or FakeAnthropicMessages()


@pytest.mark.asyncio
async def test_anthropic_provider_complete_structure_with_schema() -> None:
    client = cast(Any, FakeAnthropicClient())
    provider = AnthropicProvider(client=client)

    response = await provider.complete_structure("Give me a city", response_schema=CityArguments)

    assert response.text == '{"city":"London"}'
    assert isinstance(response.structured_response, CityArguments)
    assert response.structured_response.city == "London"
    assert response.model == "claude-3-5-sonnet-20241022"
    assert response.input_tokens == 20
    assert response.output_tokens == 10


@pytest.mark.asyncio
async def test_anthropic_provider_complete_structure_retries_and_succeeds() -> None:
    seq_messages = SequentialAnthropicMessages(
        [
            '{"invalid": 999}',
            '{"city":"Manchester"}',
        ]
    )
    client = cast(Any, FakeAnthropicClient(messages=seq_messages))
    provider = AnthropicProvider(client=client)

    response = await provider.complete_structure("Give me a city", response_schema=CityArguments)

    assert len(seq_messages.calls) == 2
    assert isinstance(response.structured_response, CityArguments)
    assert response.structured_response.city == "Manchester"
    assert (
        "Your previous response failed validation"
        in seq_messages.calls[1]["messages"][0]["content"]
    )


@pytest.mark.asyncio
async def test_anthropic_provider_complete_structure_fails_after_retries() -> None:
    seq_messages = SequentialAnthropicMessages(
        [
            "bad json 1",
            "bad json 2",
            "bad json 3",
        ]
    )
    client = cast(Any, FakeAnthropicClient(messages=seq_messages))
    provider = AnthropicProvider(client=client)

    with pytest.raises(LLMValidationError) as exc_info:
        await provider.complete_structure("Give me a city", response_schema=CityArguments)

    assert len(seq_messages.calls) == 3
    assert "Failed to validate response against schema" in str(exc_info.value)


@pytest.mark.asyncio
async def test_anthropic_provider_complete_tool_mapping() -> None:
    client = cast(Any, FakeAnthropicClient())
    provider = AnthropicProvider(client=client)
    tool = ToolSpec(
        name="lookup_weather",
        description="Look up weather for a city.",
        argument_model=CityArguments,
    )

    response = await provider.complete_tool(
        [ChatMessage(role="user", content="Tell me the weather in Paris")],
        [tool],
        ToolChoice(mode="specific", tool_name="lookup_weather"),
    )

    assert response.text is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "lookup_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert client.messages.calls[0]["model"] == "claude-3-5-sonnet-20241022"
    assert client.messages.calls[0]["tool_choice"] == {
        "type": "tool",
        "name": "lookup_weather",
    }


@pytest.mark.asyncio
async def test_anthropic_provider_stream() -> None:
    client = cast(Any, FakeAnthropicClient())
    provider = AnthropicProvider(client=client)

    chunks = []
    async for chunk in provider.stream([ChatMessage(role="user", content="Hi")]):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello from Claude"
