from __future__ import annotations

import pytest
from fakes import FailingDB, FakeDB, FakeMCPClient, FakeProvider, SequencedProvider
from pydantic import BaseModel

from customer_support_agent.models import ChatMessage, LLMResponseModel, ToolCall
from customer_support_agent.orchestration import ToolChatResult, run_tool_chat
from customer_support_agent.tools import ToolContext, build_default_registry


async def test_tool_loop_executes_tool_and_returns_final_text():
    result = await run_tool_chat(
        [ChatMessage(role="user", content="Where is order-1?")],
        FakeProvider(),
        FakeMCPClient(build_default_registry(ToolContext(db=FakeDB()))),
    )
    assert result.text == "Your order is shipped."
    assert result.iterations == 2


async def test_model_selects_one_tool_and_only_that_tool_runs():
    db = FakeDB()
    result = await run_tool_chat(
        [ChatMessage(role="user", content="Where is order-1?")],
        FakeProvider(),
        FakeMCPClient(build_default_registry(ToolContext(db=db))),
    )
    assert result.error is None
    assert db.calls == [("lookup_order", "order-1")]


async def test_tool_loop_supports_genuine_sequential_two_tool_calls():
    db = FakeDB()
    provider = SequencedProvider(
        [
            LLMResponseModel(
                model="fake",
                latency_ms=0,
                tool_calls=[
                    ToolCall(
                        id="order-call", name="lookup_order", arguments={"order_id": "order-1"}
                    )
                ],
            ),
            LLMResponseModel(
                model="fake",
                latency_ms=0,
                tool_calls=[
                    ToolCall(
                        id="product-call", name="product_details", arguments={"product_id": "p-1"}
                    )
                ],
            ),
            LLMResponseModel(model="fake", latency_ms=0, text="I checked both details."),
        ]
    )
    result = await run_tool_chat(
        [ChatMessage(role="user", content="Check my order and cancellation policy.")],
        provider,
        FakeMCPClient(build_default_registry(ToolContext(db=db))),
    )
    assert result.text == "I checked both details."
    assert result.iterations == 3
    assert db.calls == [("lookup_order", "order-1"), ("product_details", "p-1")]
    assert [message.tool_name for message in result.messages if message.role == "tool"] == [
        "lookup_order",
        "product_details",
    ]


async def test_failing_tool_is_safe_and_model_can_recover():
    provider = SequencedProvider(
        [
            LLMResponseModel(
                model="fake",
                latency_ms=0,
                tool_calls=[
                    ToolCall(id="bad-call", name="lookup_order", arguments={"order_id": "order-1"})
                ],
            ),
            LLMResponseModel(model="fake", latency_ms=0, text="I could not look up that order."),
        ]
    )
    result = await run_tool_chat(
        [ChatMessage(role="user", content="Find my order.")],
        provider,
        FakeMCPClient(build_default_registry(ToolContext(db=FailingDB()))),
    )
    assert result.text == "I could not look up that order."
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert "database password leaked" not in tool_message.content
    assert "requested support tool failed" in tool_message.content


async def test_tool_loop_stops_at_max_iterations():
    provider = FakeProvider()
    result = await run_tool_chat(
        [ChatMessage(role="user", content="Where is order-1?")],
        provider,
        FakeMCPClient(build_default_registry(ToolContext(db=FakeDB()))),
        max_iterations=1,
    )
    assert result.exhausted is True
    assert provider.calls == 1


def test_tool_chat_result_is_a_pydantic_model():
    result = ToolChatResult(
        text="done", messages=[ChatMessage(role="user", content="hi")], iterations=1
    )
    assert isinstance(result, BaseModel)
    assert result.model_dump()["exhausted"] is False


async def test_tool_loop_normalizes_provider_errors_and_validates_iterations():
    class FailingProvider:
        async def complete(self, messages, tools=None):
            raise RuntimeError("provider internals")

    result = await run_tool_chat(
        [ChatMessage(role="user", content="hello")],
        FailingProvider(),
        FakeMCPClient(build_default_registry(ToolContext(db=FakeDB()))),
    )
    assert result.error == "model_error"
    assert "provider internals" not in result.text
    with pytest.raises(ValueError, match="at least 1"):
        await run_tool_chat(
            [],
            FailingProvider(),
            FakeMCPClient(build_default_registry(ToolContext(db=FakeDB()))),
            0,
        )
