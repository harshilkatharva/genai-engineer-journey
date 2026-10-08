import asyncio
from types import SimpleNamespace
from typing import Any, cast

from pydantic import BaseModel

from function_calling_library.models import ChatMessage, ToolChoice, ToolSpec
from function_calling_library.providers import OpenAIProvider


class CityArguments(BaseModel):
    city: str


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
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


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeOpenAIResponses()


def test_openai_provider_maps_tool_response_correctly() -> None:
    client = cast(Any, FakeOpenAIClient())
    provider = OpenAIProvider(client=client)
    tool = ToolSpec(
        name="lookup_weather",
        description="Look up weather for a city.",
        argument_model=CityArguments,
    )

    response = asyncio.run(
        provider.complete(
            [ChatMessage(role="user", content="Tell me the weather in Paris")],
            [tool],
            ToolChoice(mode="specific", tool_name="lookup_weather"),
        )
    )

    assert response.text is None
    assert response.tool_calls[0].name == "lookup_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert client.responses.calls[0]["model"] == "gpt-4o-mini"
    assert client.responses.calls[0]["tool_choice"] == {
        "type": "function",
        "name": "lookup_weather",
    }
