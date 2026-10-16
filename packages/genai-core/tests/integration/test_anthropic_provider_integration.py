from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from genai_core.core import get_settings
from genai_core.llm import AnthropicProvider
from genai_core.models import ChatMessage, ToolChoice, ToolSpec


class CapitalCityInfo(BaseModel):
    capital: str = Field(description="The capital city")
    country: str = Field(description="The country name")


class WeatherReport(BaseModel):
    city: str = Field(description="The name of the city")


def _get_anthropic_api_key() -> str | None:
    return get_settings().anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")


def _handle_anthropic_error(exc: Exception) -> None:
    err_str = str(exc).lower()
    if any(
        term in err_str
        for term in [
            "401",
            "403",
            "429",
            "auth",
            "api_key",
            "credit",
            "quota",
            "rate",
            "not found",
            "permission",
            "unauthorized",
        ]
    ):
        pytest.skip(f"Anthropic live API skipped: {exc}")
    raise exc


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anthropic_complete_structure_live() -> None:
    api_key = _get_anthropic_api_key()
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    provider = AnthropicProvider(api_key=api_key)
    try:
        response = await provider.complete_structure(
            prompt="What is the capital of Italy? Return JSON matching the schema.",
            response_schema=CapitalCityInfo,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_anthropic_error(exc)

    assert response.structured_response is not None
    assert isinstance(response.structured_response, CapitalCityInfo)
    assert response.structured_response.capital.strip().lower() == "rome"
    assert response.structured_response.country.strip().lower() == "italy"
    assert response.latency_ms > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anthropic_complete_tool_live() -> None:
    api_key = _get_anthropic_api_key()
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    provider = AnthropicProvider(api_key=api_key)
    weather_tool = ToolSpec(
        name="fetch_weather",
        description="Fetch current weather report for a city.",
        argument_model=WeatherReport,
    )
    messages = [
        ChatMessage(
            role="user",
            content="Check the weather for Rome using fetch_weather.",
        )
    ]

    try:
        response = await provider.complete_tool(
            messages=messages,
            tools=[weather_tool],
            tool_choice=ToolChoice(mode="specific", tool_name="fetch_weather"),
        )
    except Exception as exc:  # noqa: BLE001
        _handle_anthropic_error(exc)

    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0].name == "fetch_weather"
    assert "rome" in response.tool_calls[0].arguments.get("city", "").lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anthropic_stream_live() -> None:
    api_key = _get_anthropic_api_key()
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    provider = AnthropicProvider(api_key=api_key)
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
        _handle_anthropic_error(exc)

    full_text = "".join(chunks)
    assert len(chunks) > 0
    assert "hello" in full_text.lower()
