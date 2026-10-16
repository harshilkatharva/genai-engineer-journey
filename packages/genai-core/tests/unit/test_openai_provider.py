from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from genai_core.exceptions import LLMValidationError
from genai_core.llm import OpenAIProvider
from genai_core.models import ChatMessage, ToolChoice, ToolSpec


class CityArguments(BaseModel):
    city: str


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if "tools" in kwargs:
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        call_id="call-openai-1",
                        name="lookup_weather",
                        arguments='{"city":"Paris"}',
                    )
                ],
                output_text="",
                model="gpt-4o-mini",
                usage=SimpleNamespace(input_tokens=11, output_tokens=5),
            )
        if "text" in kwargs:
            return SimpleNamespace(
                output=[],
                output_text='{"city":"Berlin"}',
                model="gpt-4o-mini",
                usage=SimpleNamespace(input_tokens=15, output_tokens=8),
            )
        return SimpleNamespace(
            output=[],
            output_text="OpenAI plain text",
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        )

    def stream(self, **kwargs: Any):
        return FakeOpenAIStreamContext()


class SequentialOpenAIResponses:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        return SimpleNamespace(
            output=[],
            output_text=text,
            model="gpt-4o-mini",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


class FakeOpenAIStreamContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __aiter__(self):
        self._events = [
            SimpleNamespace(delta="Hello", type="response.output_text.delta"),
            SimpleNamespace(delta=" world", type="response.output_text.delta"),
        ]
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        item = self._events[self._index]
        self._index += 1
        return item


class FakeOpenAIClient:
    def __init__(self, responses: Any = None) -> None:
        self.responses = responses or FakeOpenAIResponses()


@pytest.mark.asyncio
async def test_openai_provider_complete_structure_with_schema() -> None:
    client = cast(Any, FakeOpenAIClient())
    provider = OpenAIProvider(client=client)

    response = await provider.complete_structure("Give me a city", response_schema=CityArguments)

    assert response.text == '{"city":"Berlin"}'
    assert isinstance(response.structured_response, CityArguments)
    assert response.structured_response.city == "Berlin"
    assert response.model == "gpt-4o-mini"
    assert response.input_tokens == 15
    assert response.output_tokens == 8
    assert client.responses.calls[0]["text"]["format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_openai_provider_complete_structure_retries_and_succeeds() -> None:
    seq_responses = SequentialOpenAIResponses(
        [
            '{"invalid": "data"}',
            '{"city":"Munich"}',
        ]
    )
    client = cast(Any, FakeOpenAIClient(responses=seq_responses))
    provider = OpenAIProvider(client=client)

    response = await provider.complete_structure("Give me a city", response_schema=CityArguments)

    assert len(seq_responses.calls) == 2
    assert isinstance(response.structured_response, CityArguments)
    assert response.structured_response.city == "Munich"
    assert "Your previous response failed validation" in seq_responses.calls[1]["input"]


@pytest.mark.asyncio
async def test_openai_provider_complete_structure_fails_after_retries() -> None:
    seq_responses = SequentialOpenAIResponses(
        [
            "invalid json 1",
            "invalid json 2",
            "invalid json 3",
        ]
    )
    client = cast(Any, FakeOpenAIClient(responses=seq_responses))
    provider = OpenAIProvider(client=client)

    with pytest.raises(LLMValidationError) as exc_info:
        await provider.complete_structure("Give me a city", response_schema=CityArguments)

    assert len(seq_responses.calls) == 3  # 1 initial + 2 retries from settings
    assert "Failed to validate response against schema" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_provider_complete_structure_without_schema() -> None:
    client = cast(Any, FakeOpenAIClient())
    provider = OpenAIProvider(client=client)

    response = await provider.complete_structure("Hello")

    assert response.text == "OpenAI plain text"
    assert response.structured_response is None


@pytest.mark.asyncio
async def test_openai_provider_complete_tool_mapping() -> None:
    client = cast(Any, FakeOpenAIClient())
    provider = OpenAIProvider(client=client)
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
    assert client.responses.calls[0]["model"] == "gpt-4o-mini"
    assert client.responses.calls[0]["tool_choice"] == {
        "type": "function",
        "name": "lookup_weather",
    }


@pytest.mark.asyncio
async def test_openai_provider_stream() -> None:
    client = cast(Any, FakeOpenAIClient())
    provider = OpenAIProvider(client=client)

    chunks = []
    async for chunk in provider.stream([ChatMessage(role="user", content="Hi")]):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello world"
