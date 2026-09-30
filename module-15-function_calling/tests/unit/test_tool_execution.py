import asyncio

from pydantic import BaseModel

from function_calling_library.models import ChatMessage, LLMManagerResponse, ToolCall, ToolSpec
from function_calling_library.services import execute_tool_calls, extract_structured_output


class NumberArguments(BaseModel):
    value: int


class ExtractedAnswer(BaseModel):
    answer: str


def test_parallel_execution_preserves_call_ids_and_input_order() -> None:
    async def handler(arguments: NumberArguments) -> int:
        await asyncio.sleep(0.02 if arguments.value == 1 else 0)
        return arguments.value * 2

    spec = ToolSpec(
        name="double",
        description="Double a number",
        argument_model=NumberArguments,
        handler=handler,
    )
    calls = [
        ToolCall(id="call-slow", name="double", arguments={"value": 1}),
        ToolCall(id="call-fast", name="double", arguments={"value": 2}),
        ToolCall(id="call-bad", name="double", arguments={"value": "invalid"}),
    ]

    results = asyncio.run(execute_tool_calls(calls, [spec]))

    assert [result.call_id for result in results] == ["call-slow", "call-fast", "call-bad"]
    assert [result.result for result in results[:2]] == [2, 4]
    assert results[2].error is not None
    assert results[2].error.code == "validation_error"


def test_tool_handler_exception_becomes_structured_error() -> None:
    def handler(arguments: NumberArguments) -> None:
        raise RuntimeError("unavailable")

    spec = ToolSpec(
        name="fail", description="Raises", argument_model=NumberArguments, handler=handler
    )
    result = asyncio.run(
        execute_tool_calls([ToolCall(id="call-1", name="fail", arguments={"value": 3})], [spec])
    )[0]

    assert result.call_id == "call-1"
    assert result.error is not None
    assert result.error.code == "execution_error"


def test_structured_output_uses_forced_function_call_and_validates() -> None:
    class FakeProvider:
        async def complete(self, messages, tools=None, tool_choice=None):
            assert tool_choice is not None
            assert tool_choice.mode == "specific"
            assert tools is not None
            return LLMManagerResponse(
                tool_calls=[
                    ToolCall(
                        id="forced-call",
                        name="emit_structured_output",
                        arguments={"answer": "42"},
                    )
                ]
            )

    class FakeService:
        async def complete(self, request):
            return await FakeProvider().complete(
                request.messages, request.tools, request.tool_choice
            )

    output, error = asyncio.run(
        extract_structured_output(
            FakeService(),
            [ChatMessage(role="user", content="Answer the question")],
            ExtractedAnswer,
        )
    )

    assert error is None
    assert isinstance(output, ExtractedAnswer)
    assert output.answer == "42"
