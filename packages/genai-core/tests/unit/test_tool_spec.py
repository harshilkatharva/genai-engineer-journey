from typing import Any, cast

from pydantic import BaseModel, Field

from genai_core.models import ToolChoice, ToolSpec


class SearchArguments(BaseModel):
    query: str = Field(description="Search terms")
    limit: int = Field(default=10, description="Max results")


def test_tool_spec_schema_conversions() -> None:
    tool = ToolSpec(
        name="search",
        description="Search database",
        argument_model=SearchArguments,
    )

    openai_tool = tool.to_openai_tool()
    assert openai_tool["type"] == "function"
    assert openai_tool["name"] == "search"
    parameters = cast(dict[str, Any], openai_tool["parameters"])
    assert "query" in parameters["properties"]

    anthropic_tool = tool.to_anthropic_tool()
    assert anthropic_tool["name"] == "search"
    assert "query" in anthropic_tool["input_schema"]["properties"]

    google_decl = tool.to_google_function_declaration()
    assert google_decl["name"] == "search"
    assert "query" in google_decl["parameters"]["properties"]


def test_tool_spec_parse_arguments_success() -> None:
    tool = ToolSpec(
        name="search",
        description="Search database",
        argument_model=SearchArguments,
    )

    model, error = tool.parse_arguments('{"query": "python", "limit": 5}')
    assert error is None
    assert isinstance(model, SearchArguments)
    assert model.query == "python"
    assert model.limit == 5


def test_tool_spec_parse_arguments_invalid_json() -> None:
    tool = ToolSpec(
        name="search",
        description="Search database",
        argument_model=SearchArguments,
    )

    model, error = tool.parse_arguments("invalid json")
    assert model is None
    assert error is not None
    assert error.code == "invalid_json"


def test_tool_choice_mappings() -> None:
    auto_choice = ToolChoice(mode="auto")
    assert auto_choice.to_openai() == "auto"
    assert auto_choice.to_anthropic() == {"type": "auto"}
    assert auto_choice.to_google() == {"mode": "AUTO"}

    any_choice = ToolChoice(mode="any")
    assert any_choice.to_openai() == "required"
    assert any_choice.to_anthropic() == {"type": "any"}
    assert any_choice.to_google() == {"mode": "ANY"}

    specific_choice = ToolChoice(mode="specific", tool_name="search")
    assert specific_choice.to_openai() == {"type": "function", "name": "search"}
    assert specific_choice.to_anthropic() == {"type": "tool", "name": "search"}
    assert specific_choice.to_google() == {
        "mode": "ANY",
        "allowed_function_names": ["search"],
    }
