import asyncio
from types import SimpleNamespace

from function_calling_library.models import ChatMessage
from function_calling_library.providers import GoogleProvider


class FakeGoogleModels:
    def __init__(self) -> None:
        self.contents = None
        self.config = None

    async def generate_content(self, *, model, contents, config):
        self.contents = contents
        self.config = config
        return SimpleNamespace(candidates=[], usage_metadata=None, text="Done")


def test_google_provider_maps_system_and_tool_messages_correctly() -> None:
    models = FakeGoogleModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    provider = GoogleProvider(client=client)
    messages = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(role="user", content="What's the weather?"),
        ChatMessage(role="assistant", content="Checking now."),
        ChatMessage(
            role="tool",
            content='{"forecast":"sunny"}',
            tool_name="get_weather",
            tool_call_id="call-weather-1",
        ),
    ]

    response = asyncio.run(provider.complete(messages))

    assert response.text == "Done"
    assert models.config.system_instruction == "Be concise."
    assert [content.role for content in models.contents] == ["user", "model", "user"]
    function_response = models.contents[-1].parts[0].function_response
    assert function_response.name == "get_weather"
    assert function_response.id == "call-weather-1"
    assert function_response.response == {"forecast": "sunny"}


def test_google_provider_rejects_tool_result_without_name() -> None:
    models = FakeGoogleModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    provider = GoogleProvider(client=client)

    try:
        asyncio.run(provider.complete([ChatMessage(role="tool", content='{"forecast":"sunny"}')]))
    except ValueError as error:
        assert str(error) == "Google tool-result messages require tool_name"
    else:
        raise AssertionError("Expected missing tool_name to be rejected")
