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
