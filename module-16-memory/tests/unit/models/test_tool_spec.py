from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from memory_system.models import ToolChoice, ToolSpec


class ToolArguments(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(gt=0)


def make_tool() -> ToolSpec:
    return ToolSpec(
        name="search",
        description="Search records",
        argument_model=ToolArguments,
        handler=lambda query, limit: (query, limit),
    )


def test_tool_choice_serializes_and_validates_modes() -> None:
    expected: dict[Literal["auto", "any", "specific"], tuple[Any, Any, Any, Any]] = {
        "auto": (None, "auto", {"type": "auto"}, {"mode": "AUTO"}),
        "any": (None, "required", {"type": "any"}, {"mode": "ANY"}),
        "specific": (
            "search",
            {"type": "function", "name": "search"},
            {"type": "tool", "name": "search"},
            {"mode": "ANY", "allowed_function_names": ["search"]},
        ),
    }
    for mode, (name, openai, anthropic, google) in expected.items():
        choice = ToolChoice(mode=mode, tool_name=name)
        assert choice.to_openai() == openai
        assert choice.to_anthropic() == anthropic
        assert choice.to_google() == google
    for values in (
        {"mode": "specific"},
        {"mode": "auto", "tool_name": "search"},
    ):
        with pytest.raises(ValidationError, match="tool_name"):
            ToolChoice.model_validate(values)


def test_tool_spec_exports_schema_for_each_provider() -> None:
    tool = make_tool()

    assert tool.parameters["properties"]["query"]["type"] == "string"
    assert tool.to_openai_tool() == {
        "type": "function",
        "name": "search",
        "description": "Search records",
        "parameters": tool.parameters,
        "strict": False,
    }
    assert tool.to_anthropic_tool() == {
        "name": "search",
        "description": "Search records",
        "input_schema": tool.parameters,
    }
    assert tool.to_google_function_declaration() == {
        "name": "search",
        "description": "Search records",
        "parameters": tool.parameters,
    }
    assert "handler" not in tool.model_dump()


def test_parse_arguments_accepts_valid_representations() -> None:
    valid_arguments: list[tuple[Any, dict[str, Any]]] = [
        ('{"query":"cats","limit":2}', {"query": "cats", "limit": 2}),
        ({"query": "dogs", "limit": 3}, {"query": "dogs", "limit": 3}),
        (ToolArguments(query="birds", limit=4), {"query": "birds", "limit": 4}),
    ]
    for raw, expected in valid_arguments:
        parsed, error = make_tool().parse_arguments(raw)
        assert error is None
        assert parsed is not None
        assert parsed.model_dump() == expected


def test_parse_arguments_reports_malformed_or_invalid_values() -> None:
    invalid_arguments: list[tuple[Any, str]] = [
        ("{", "invalid_json"),
        ("[]", "arguments_not_object"),
        (None, "invalid_argument_type"),
        ({"query": "", "limit": 0}, "validation_error"),
    ]
    for raw, error_code in invalid_arguments:
        parsed, error = make_tool().parse_arguments(raw)
        assert parsed is None
        assert error is not None
        assert error.code == error_code
