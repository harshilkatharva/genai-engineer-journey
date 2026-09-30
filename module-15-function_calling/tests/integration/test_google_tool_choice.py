import pytest
from pydantic import BaseModel

from function_calling_library.core import get_settings
from function_calling_library.models import ChatMessage, ToolChoice, ToolSpec
from function_calling_library.providers import GoogleProvider


class CityArguments(BaseModel):
    city: str


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_forced_specific_tool_choice_live() -> None:
    if not get_settings().google_api_key:
        pytest.skip("GOOGLE_API_KEY is not configured")

    provider = GoogleProvider()
    response = await provider.complete(
        [ChatMessage(role="user", content="Call the tool for Paris, with city exactly Paris.")],
        [
            ToolSpec(
                name="report_city",
                description="Report the requested city.",
                argument_model=CityArguments,
            )
        ],
        ToolChoice(mode="specific", tool_name="report_city"),
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "report_city"
    assert response.tool_calls[0].arguments["city"].lower() == "paris"
