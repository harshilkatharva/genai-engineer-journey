import asyncio

from pydantic import BaseModel

from function_calling_library.models import ChatMessage, LLMManagerResponse, ToolCall, ToolSpec
from function_calling_library.services import execute_tool_calls, extract_structured_output


# Task 4
class ExerciseArguments(BaseModel):
    count: int


def test_forced_tool_call_arguments_are_validated() -> None:
    class FakeService:
        async def complete(self, request):
            assert request.tool_choice is not None
            assert request.tool_choice.mode == "specific"
            assert request.tool_choice.tool_name == "emit_structured_output"
            return LLMManagerResponse(
                tool_calls=[
                    ToolCall(
                        id="forced-call",
                        name="emit_structured_output",
                        arguments={"count": "42"},
                    )
                ]
            )

    output, error = asyncio.run(
        extract_structured_output(
            FakeService(),
            [ChatMessage(role="user", content="Return the count")],
            ExerciseArguments,
        )
    )

    assert error is None
    assert isinstance(output, ExerciseArguments)
    assert output.count == 42


# Task 7
class CallArguments(BaseModel):
    value: int


def test_parallel_results_match_originating_calls_when_one_fails() -> None:
    async def handler(arguments: CallArguments) -> int:
        if arguments.value == 2:
            raise RuntimeError("call failed")
        await asyncio.sleep(0.02 if arguments.value == 1 else 0)
        return arguments.value * 10

    spec = ToolSpec(
        name="calculate",
        description="Calculate a value.",
        argument_model=CallArguments,
        handler=handler,
    )
    calls = [
        ToolCall(id="slow", name="calculate", arguments={"value": 1}),
        ToolCall(id="failed", name="calculate", arguments={"value": 2}),
        ToolCall(id="fast", name="calculate", arguments={"value": 3}),
    ]

    results = asyncio.run(execute_tool_calls(calls, [spec]))
    results_by_id = {result.call_id: result for result in results}

    assert results_by_id["slow"].result == 10
    assert results_by_id["fast"].result == 30
    assert results_by_id["failed"].error is not None
    assert results_by_id["failed"].error.code == "execution_error"


# Task 9
def test_tool_schemas_are_functionally_equivalent_across_providers() -> None:
    class ProviderArguments(BaseModel):
        city: str
        days: int
        include_forecast: bool = False

    spec = ToolSpec(
        name="get_weather",
        description="Look up weather.",
        argument_model=ProviderArguments,
    )
    openai = spec.to_openai_tool()
    anthropic = spec.to_anthropic_tool()
    google = spec.to_google_function_declaration()

    def meaning(schema):
        return (
            {field: definition["type"] for field, definition in schema["properties"].items()},
            set(schema.get("required", [])),
        )

    definitions = [openai, anthropic, google]
    assert {definition["name"] for definition in definitions} == {"get_weather"}
    assert {definition["description"] for definition in definitions} == {"Look up weather."}
    semantics = [
        meaning(openai["parameters"]),
        meaning(anthropic["input_schema"]),
        meaning(google["parameters"]),
    ]
    assert semantics[0] == semantics[1] == semantics[2]
