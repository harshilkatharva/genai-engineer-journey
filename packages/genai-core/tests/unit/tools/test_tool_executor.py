from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from genai_core.models import ToolCall
from genai_core.tools import (
    ToolExecutionStatus,
    ToolExecutor,
    ToolRegistry,
    tool,
)


@pytest.fixture
def executor():
    registry = ToolRegistry()

    @tool(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    @tool(name="slow_tool")
    async def slow_tool(delay: float) -> str:
        await asyncio.sleep(delay)
        return "finished"

    @tool(name="failing_tool")
    def failing_tool() -> str:
        raise RuntimeError("Something broke inside the tool!")

    @tool(name="large_output_tool")
    def large_output_tool(size: int) -> str:
        return "x" * size

    registry.register(add)
    registry.register(slow_tool)
    registry.register(failing_tool)
    registry.register(large_output_tool)

    return ToolExecutor(
        registry=registry,
        default_timeout_s=1.0,
        max_output_chars=50,
        max_concurrency=2,
    )


@pytest.mark.asyncio
async def test_executor_successful_run(executor: ToolExecutor):
    call = ToolCall(id="call_1", name="add", arguments={"a": 10, "b": 25})
    res = await executor.execute(call)

    assert res.is_success()
    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.content == 35
    assert res.error_message is None
    assert res.call_id == "call_1"
    assert res.tool_name == "add"
    assert res.latency_ms >= 0

    msg = res.to_chat_message()
    assert msg.role == "tool"
    assert msg.content == "35"
    assert msg.tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_executor_validation_error(executor: ToolExecutor):
    # Missing required argument 'b'
    call = ToolCall(id="call_2", name="add", arguments={"a": 10})
    res = await executor.execute(call)

    assert not res.is_success()
    assert res.status == ToolExecutionStatus.VALIDATION_ERROR
    assert "Validation failed" in (res.error_message or "")

    msg = res.to_chat_message()
    assert "Error:" in msg.content


@pytest.mark.asyncio
async def test_executor_unregistered_tool(executor: ToolExecutor):
    call = ToolCall(id="call_3", name="nonexistent", arguments={})
    res = await executor.execute(call)

    assert not res.is_success()
    assert res.status == ToolExecutionStatus.EXECUTION_ERROR
    assert "is not registered" in (res.error_message or "")


@pytest.mark.asyncio
async def test_executor_runtime_exception_isolated(executor: ToolExecutor):
    call = ToolCall(id="call_4", name="failing_tool", arguments={})
    res = await executor.execute(call)

    assert not res.is_success()
    assert res.status == ToolExecutionStatus.EXECUTION_ERROR
    assert "RuntimeError" in (res.error_message or "")
    assert "Something broke" in (res.error_message or "")


@pytest.mark.asyncio
async def test_executor_timeout_enforcement(executor: ToolExecutor):
    # Timeout is 0.2s, tool takes 0.5s
    call = ToolCall(id="call_5", name="slow_tool", arguments={"delay": 0.5})
    res = await executor.execute(call, timeout_s=0.2)

    assert not res.is_success()
    assert res.status == ToolExecutionStatus.TIMEOUT
    assert "timed out after 0.2s" in (res.error_message or "")


@pytest.mark.asyncio
async def test_executor_output_truncation(executor: ToolExecutor):
    # max_output_chars is configured to 50
    call = ToolCall(id="call_6", name="large_output_tool", arguments={"size": 200})
    res = await executor.execute(call)

    assert res.is_success()
    assert res.truncated is True
    assert isinstance(res.content, str)
    assert len(res.content) > 50
    assert "[TRUNCATED: output exceeded 50 characters]" in res.content


@pytest.mark.asyncio
async def test_executor_batch_execution(executor: ToolExecutor):
    calls = [
        ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
        ToolCall(id="c2", name="add", arguments={"a": 3, "b": 4}),
        ToolCall(id="c3", name="add", arguments={"a": 5, "b": 6}),
    ]
    results = await executor.execute_many(calls)
    assert len(results) == 3
    assert [r.content for r in results] == [3, 7, 11]
    assert all(r.is_success() for r in results)


@pytest.mark.asyncio
async def test_executor_to_chat_message_formatting():
    class TestModel(BaseModel):
        status: str
        count: int

    # Dict formatting
    r1 = ToolExecutor().registry

    @tool
    def return_dict() -> dict:
        return {"a": 1}

    r1.register(return_dict)
    exec1 = ToolExecutor(registry=r1)
    res1 = await exec1.execute(ToolCall(id="c1", name="return_dict", arguments={}))
    msg1 = res1.to_chat_message()
    assert '"a": 1' in msg1.content

    # BaseModel formatting
    @tool
    def return_model() -> TestModel:
        return TestModel(status="ok", count=42)

    r1.register(return_model)
    res2 = await exec1.execute(ToolCall(id="c2", name="return_model", arguments={}))
    msg2 = res2.to_chat_message()
    assert '"status":"ok"' in msg2.content or '"status": "ok"' in msg2.content
