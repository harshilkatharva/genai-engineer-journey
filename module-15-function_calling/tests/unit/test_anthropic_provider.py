import asyncio
from types import SimpleNamespace
from typing import Any, cast

from pydantic import BaseModel

from function_calling_library.models import ChatMessage, ToolChoice, ToolSpec
from function_calling_library.providers import AnthropicProvider


class CityArguments(BaseModel):
    city: str


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
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


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()


def test_anthropic_provider_maps_tool_response_correctly() -> None:
    client = cast(Any, FakeAnthropicClient())
    provider = AnthropicProvider(client=client)
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
    assert client.messages.calls[0]["model"] == "claude-3-5-sonnet-20241022"
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "lookup_weather"}
