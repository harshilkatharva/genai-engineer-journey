from pydantic import BaseModel

from function_calling_library.models import ToolChoice, ToolSpec


class WeatherArguments(BaseModel):
    city: str


def make_spec() -> ToolSpec:
    return ToolSpec(
        name="get_weather",
        description="Look up weather for a city.",
        argument_model=WeatherArguments,
    )


def test_tool_spec_parses_and_validates_arguments() -> None:
    parsed, error = make_spec().parse_arguments('{"city":"Paris"}')

    assert error is None
    assert isinstance(parsed, WeatherArguments)
    assert parsed.city == "Paris"


def test_tool_spec_returns_structured_errors_without_raising() -> None:
    spec = make_spec()

    malformed, malformed_error = spec.parse_arguments('{"city":')
    invalid, invalid_error = spec.parse_arguments('{"city":42}')
    non_object, non_object_error = spec.parse_arguments('["Paris"]')

    assert malformed is None
    assert malformed_error is not None
    assert malformed_error.code == "invalid_json"
    assert invalid is None
    assert invalid_error is not None
    assert invalid_error.code == "validation_error"
    assert non_object is None
    assert non_object_error is not None
    assert non_object_error.code == "arguments_not_object"


def test_tool_spec_adapters_share_the_argument_model_schema() -> None:
    spec = make_spec()
    schema = spec.parameters

    assert spec.to_openai_tool()["parameters"] == schema
    assert spec.to_anthropic_tool()["input_schema"] == schema
    assert spec.to_google_function_declaration()["parameters"] == schema


def test_tool_choice_modes_map_to_provider_contracts() -> None:
    auto = ToolChoice(mode="auto")
    any_tool = ToolChoice(mode="any")
    specific = ToolChoice(mode="specific", tool_name="get_weather")

    assert auto.to_openai() == "auto"
    assert any_tool.to_openai() == "required"
    assert specific.to_openai() == {"type": "function", "name": "get_weather"}
    assert auto.to_anthropic() == {"type": "auto"}
    assert any_tool.to_anthropic() == {"type": "any"}
    assert specific.to_anthropic() == {"type": "tool", "name": "get_weather"}
    assert auto.to_google() == {"mode": "AUTO"}
    assert any_tool.to_google() == {"mode": "ANY"}
    assert specific.to_google() == {
        "mode": "ANY",
        "allowed_function_names": ["get_weather"],
    }
