from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from genai_core.core import get_settings
from genai_core.llm import GoogleProvider
from genai_core.models import ChatMessage, ToolChoice, ToolSpec


class CapitalCityInfo(BaseModel):
    capital: str = Field(description="The capital city")
    country: str = Field(description="The country name")


class WeatherReport(BaseModel):
    city: str = Field(description="The name of the city")


def _get_google_api_key() -> str | None:
    return (
        get_settings().google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )


def _handle_google_error(exc: Exception) -> None:
    err_str = str(exc).lower()
    if any(
        code in err_str
        for code in [
            "429",
            "503",
            "quota",
            "unavailable",
            "high demand",
            "resource_exhausted",
        ]
    ):
        pytest.skip(f"Google API transient rate-limit or high demand: {exc}")
    raise exc


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_complete_structure_live() -> None:
    api_key = _get_google_api_key()
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set")

    provider = GoogleProvider(api_key=api_key)
    try:
        response = await provider.complete_structure(
            prompt="What is the capital of France? Return JSON.",
            response_schema=CapitalCityInfo,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_google_error(exc)

    assert response.structured_response is not None
    assert isinstance(response.structured_response, CapitalCityInfo)
    assert response.structured_response.capital.strip().lower() == "paris"
    assert response.structured_response.country.strip().lower() == "france"
    assert response.latency_ms > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_complete_tool_live() -> None:
    api_key = _get_google_api_key()
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set")

    provider = GoogleProvider(api_key=api_key)
    weather_tool = ToolSpec(
        name="fetch_weather",
        description="Fetch current weather report for a city.",
        argument_model=WeatherReport,
    )
    messages = [
        ChatMessage(
            role="user",
            content="What is the weather in Tokyo? Call fetch_weather with city Tokyo.",
        )
    ]

    try:
        response = await provider.complete_tool(
            messages=messages,
            tools=[weather_tool],
            tool_choice=ToolChoice(mode="specific", tool_name="fetch_weather"),
        )
    except Exception as exc:  # noqa: BLE001
        _handle_google_error(exc)

    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0].name == "fetch_weather"
    assert "tokyo" in response.tool_calls[0].arguments.get("city", "").lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_stream_live() -> None:
    api_key = _get_google_api_key()
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set")

    provider = GoogleProvider(api_key=api_key)
    messages = [
        ChatMessage(
            role="user",
            content="Reply with exactly the words 'Hello Antigravity'.",
        )
    ]

    try:
        chunks: list[str] = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001
        _handle_google_error(exc)

    full_text = "".join(chunks)
    assert len(chunks) > 0
    assert "hello" in full_text.lower()


class TicketLookupArgs(BaseModel):
    ticket_id: str


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_multi_turn_tool_calling_with_thought_signature_live() -> None:
    api_key = _get_google_api_key()
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not set")

    provider = GoogleProvider(api_key=api_key, model="gemini-3.5-flash-lite")
    tool = ToolSpec(
        name="lookup_ticket",
        description="Lookup customer ticket details",
        argument_model=TicketLookupArgs,
    )

    try:
        # Turn 1: Gemini calls lookup_ticket
        resp1 = await provider.complete_tool(
            messages=[ChatMessage(role="user", content="What is the status of ticket T-1001?")],
            tools=[tool],
        )
        assert resp1.tool_calls, "Expected tool call on Turn 1"
        assert resp1.tool_calls[0].name == "lookup_ticket"

        # Turn 2: Return tool result back with thought_signature preserved
        resp2 = await provider.complete_tool(
            messages=[
                ChatMessage(role="user", content="What is the status of ticket T-1001?"),
                ChatMessage(role="assistant", content="", tool_calls=resp1.tool_calls),
                ChatMessage(
                    role="tool",
                    tool_name="lookup_ticket",
                    tool_call_id=resp1.tool_calls[0].id,
                    content='{"ticket_id": "T-1001", "status": "resolved"}',
                ),
            ],
            tools=[tool],
        )
        assert resp2.text, "Expected text answer on Turn 2"
        assert "resolved" in resp2.text.lower()
    except Exception as exc:  # noqa: BLE001
        _handle_google_error(exc)
