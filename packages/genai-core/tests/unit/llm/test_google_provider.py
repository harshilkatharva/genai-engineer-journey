from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from genai_core.exceptions import LLMValidationError
from genai_core.llm import GoogleProvider
from genai_core.models import ChatMessage, ToolCall


class CitySchema(BaseModel):
    city: str
    country: str


class FakeGoogleModels:
    def __init__(
        self,
        response_text: str = "Done",
        function_call: Any = None,
        thought_signature: Any = None,
    ) -> None:
        self.contents: Any = None
        self.config: Any = None
        self.response_text = response_text
        self.function_call = function_call
        self.thought_signature = thought_signature
        self.call_history: list[dict[str, Any]] = []

    async def generate_content(self, *, model: Any, contents: Any, config: Any) -> Any:
        self.contents = contents
        self.config = config
        self.call_history.append({"contents": contents, "config": config})
        parts = []
        if self.function_call:
            parts.append(
                SimpleNamespace(
                    function_call=self.function_call,
                    thought_signature=self.thought_signature,
                )
            )
        candidates = [SimpleNamespace(content=SimpleNamespace(parts=parts))] if parts else []
        usage = SimpleNamespace(prompt_token_count=15, candidates_token_count=10)
        return SimpleNamespace(
            candidates=candidates,
            usage_metadata=usage,
            text=self.response_text,
            model_version="gemini-2.5-flash",
        )

    async def generate_content_stream(self, *, model, contents):
        for token in ["Hello", " ", "world"]:
            yield SimpleNamespace(text=token)


class SequentialGoogleModels:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.prompts: list[str] = []

    async def generate_content(self, *, model, contents, config):
        self.prompts.append(contents)
        text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
        return SimpleNamespace(
            candidates=[],
            usage_metadata=usage,
            text=text,
            model_version="gemini-2.5-flash",
        )


@pytest.mark.asyncio
async def test_google_provider_complete_structure_with_schema() -> None:
    models = FakeGoogleModels(response_text='{"city":"Tokyo","country":"Japan"}')
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client, model="gemini-2.5-flash")

    result = await provider.complete_structure("Name a city", response_schema=CitySchema)

    assert result.text == '{"city":"Tokyo","country":"Japan"}'
    assert isinstance(result.structured_response, CitySchema)
    assert result.structured_response.city == "Tokyo"
    assert result.structured_response.country == "Japan"
    assert result.model == "gemini-2.5-flash"
    assert result.input_tokens == 15
    assert result.output_tokens == 10
    assert result.latency_ms >= 0
    assert models.config.response_mime_type == "application/json"
    assert models.config.response_schema == CitySchema


@pytest.mark.asyncio
async def test_google_provider_complete_structure_retries_and_succeeds() -> None:
    # First response is invalid schema, second is valid
    models = SequentialGoogleModels(
        [
            '{"invalid_field": 123}',
            '{"city":"Kyoto","country":"Japan"}',
        ]
    )
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client)

    result = await provider.complete_structure("Give a city", response_schema=CitySchema)

    assert models.call_count == 2
    assert isinstance(result.structured_response, CitySchema)
    assert result.structured_response.city == "Kyoto"
    assert "Your previous response failed validation" in models.prompts[1]


@pytest.mark.asyncio
async def test_google_provider_complete_structure_fails_after_retries() -> None:
    # Always invalid JSON - uses default max_structured_error_retry from settings
    models = SequentialGoogleModels(
        [
            "Not valid json 1",
            "Not valid json 2",
            "Not valid json 3",
        ]
    )
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client)

    with pytest.raises(LLMValidationError) as exc_info:
        await provider.complete_structure("Give a city", response_schema=CitySchema)

    assert models.call_count == 3  # 1 initial + 2 retries from settings
    assert "Failed to validate response against schema" in str(exc_info.value)


@pytest.mark.asyncio
async def test_google_provider_complete_structure_without_schema() -> None:
    models = FakeGoogleModels(response_text="Plain text answer")
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client)

    result = await provider.complete_structure("Tell me something")

    assert result.text == "Plain text answer"
    assert result.structured_response is None
    assert models.config.response_mime_type is None


@pytest.mark.asyncio
async def test_google_provider_complete_tool_mapping() -> None:
    models = FakeGoogleModels(response_text="Done")
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
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

    response = await provider.complete_tool(messages)

    assert response.text == "Done"
    assert models.config.system_instruction == "Be concise."
    assert [content.role for content in models.contents] == ["user", "model", "user"]
    function_response = models.contents[-1].parts[0].function_response
    assert function_response.name == "get_weather"
    assert function_response.id == "call-weather-1"
    assert function_response.response == {"forecast": "sunny"}


@pytest.mark.asyncio
async def test_google_provider_complete_tool_parses_tool_calls() -> None:
    fake_func_call = SimpleNamespace(name="lookup_weather", args={"city": "Paris"})
    models = FakeGoogleModels(response_text="", function_call=fake_func_call)
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client)

    response = await provider.complete_tool([ChatMessage(role="user", content="Weather?")])

    assert response.text is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "lookup_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}


@pytest.mark.asyncio
async def test_google_provider_rejects_tool_result_without_name() -> None:
    models = FakeGoogleModels()
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client)

    with pytest.raises(ValueError, match="Google tool-result messages require tool_name"):
        await provider.complete_tool([ChatMessage(role="tool", content='{"forecast":"sunny"}')])


@pytest.mark.asyncio
async def test_google_provider_stream() -> None:
    models = FakeGoogleModels()
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client)

    chunks = []
    async for chunk in provider.stream([ChatMessage(role="user", content="Hi")]):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello world"


@pytest.mark.asyncio
async def test_google_provider_complete_tool_captures_thought_signature() -> None:
    fake_func_call = SimpleNamespace(id="call-123", name="lookup_weather", args={"city": "Paris"})
    models = FakeGoogleModels(
        response_text="",
        function_call=fake_func_call,
        thought_signature=b"test-sig-bytes",
    )
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client, min_call_interval=0.0)

    response = await provider.complete_tool([ChatMessage(role="user", content="Weather?")])

    assert response.text is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "lookup_weather"
    assert response.tool_calls[0].thought_signature == b"test-sig-bytes"


@pytest.mark.asyncio
async def test_google_provider_complete_tool_reconstructs_thought_signature() -> None:
    models = FakeGoogleModels(response_text="Final Answer")
    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    provider = GoogleProvider(client=client, min_call_interval=0.0)

    messages = [
        ChatMessage(role="user", content="Lookup ticket"),
        ChatMessage(
            role="assistant",
            content="Looking up",
            tool_calls=[
                ToolCall(
                    id="call_t1",
                    name="lookup_ticket",
                    arguments={"ticket_id": "T1"},
                    thought_signature=b"sig-roundtrip",
                )
            ],
        ),
        ChatMessage(
            role="tool",
            tool_name="lookup_ticket",
            tool_call_id="call_t1",
            content='{"status": "open"}',
        ),
    ]

    response = await provider.complete_tool(messages)

    assert response.text == "Final Answer"
    assert len(models.contents) == 3
    model_content = models.contents[1]
    assert model_content.role == "model"
    assert len(model_content.parts) == 2
    assert model_content.parts[0].text == "Looking up"
    fc_part = model_content.parts[1]
    assert fc_part.function_call.name == "lookup_ticket"
    assert fc_part.function_call.args == {"ticket_id": "T1"}
    assert fc_part.thought_signature == b"sig-roundtrip"


@pytest.mark.asyncio
async def test_google_provider_complete_tool_429_backoff() -> None:
    class FlakyRateLimitModels:
        def __init__(self) -> None:
            self.attempts = 0

        async def generate_content(self, *, model: Any, contents: Any, config: Any) -> Any:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED: Rate limit reached. Please retry in 0.01s."
                )
            return SimpleNamespace(
                candidates=[],
                usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=5),
                text="Recovered from 429",
                model_version="gemini-3.5-flash-lite",
            )

    client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=FlakyRateLimitModels())))
    provider = GoogleProvider(client=client, min_call_interval=0.0)

    response = await provider.complete_tool([ChatMessage(role="user", content="Hi")])
    assert response.text == "Recovered from 429"
