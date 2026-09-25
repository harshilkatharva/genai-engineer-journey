from types import SimpleNamespace

import pytest

from customer_support_agent.models import ChatMessage, ToolDefinition
from customer_support_agent.providers.anthropic_provider import AnthropicProvider
from customer_support_agent.providers.google_provider import GoogleProvider
from customer_support_agent.providers.openai_provider import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_adapter_maps_text_and_tool_call():
    response = SimpleNamespace(
        model="gpt",
        output_text="",
        output=[
            SimpleNamespace(type="function_call", call_id="c", name="lookup", arguments={"id": "1"})
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )

    class Responses:
        async def create(self, **kwargs):
            return response

    client = SimpleNamespace(responses=Responses())
    result = await OpenAIProvider(client).complete(
        [ChatMessage(role="user", content="hi")], [ToolDefinition(name="lookup", description="d")]
    )
    assert result.tool_calls[0].name == "lookup"


@pytest.mark.asyncio
async def test_anthropic_adapter_maps_text():
    response = SimpleNamespace(
        model="claude",
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )

    class Messages:
        async def create(self, **kwargs):
            return response

    result = await AnthropicProvider(SimpleNamespace(messages=Messages())).complete(
        [ChatMessage(role="user", content="hi")]
    )
    assert result.text == "hello"


@pytest.mark.asyncio
async def test_google_adapter_maps_text():
    response = SimpleNamespace(
        candidates=[],
        text="hello",
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2),
    )

    class Models:
        async def generate_content(self, **kwargs):
            return response

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    result = await GoogleProvider(client).complete([ChatMessage(role="user", content="hi")])
    assert result.text == "hello"
