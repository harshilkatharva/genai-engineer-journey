from __future__ import annotations

import pytest
from fakes import FakeDB
from pydantic import BaseModel

from customer_support_agent.tools import (
    RegisteredTool,
    ToolContext,
    ToolRegistry,
    build_default_registry,
)


def test_tool_context_and_registered_tool_are_pydantic_models():
    db = FakeDB()
    context = ToolContext(db=db, customer_id="customer-1")
    assert isinstance(context, BaseModel)
    assert context.model_dump()["db"] is db
    assert context.model_dump()["customer_id"] == "customer-1"

    async def handler(value: str) -> str:
        return value

    registered = RegisteredTool(
        definition={"name": "example", "description": "example", "parameters": {}},
        handler=handler,
    )
    assert isinstance(registered, BaseModel)
    assert registered.model_dump()["definition"]["name"] == "example"
    assert registered.model_dump()["handler"] is handler
    with pytest.raises(ValueError):
        ToolContext(db=db, customer_id=123)


def test_registry_definitions_explain_use_and_non_use_for_every_tool():
    definitions = build_default_registry(ToolContext(db=FakeDB())).definitions()
    assert {definition.name for definition in definitions} == {
        "lookup_order",
        "product_search",
        "product_details",
        "escalation",
    }
    for definition in definitions:
        description = definition.description.lower()
        assert "what it does" in description
        assert "use when" in description
        assert "do not use when" in description


async def test_registry_executes_each_default_tool_and_applies_constraints():
    db = FakeDB()
    registry = build_default_registry(ToolContext(db=db, customer_id="customer-1"))
    assert (await registry.execute("lookup_order", {"order_id": "order-1"}))["found"]
    products = await registry.execute(
        "product_search", {"query": "widget", "max_price": 20, "limit": 100}
    )
    assert products["products"][0]["product_id"] == "p-1"
    assert (await registry.execute("product_details", {"product_id": "p-1"}))["found"]
    escalation = await registry.execute(
        "escalation", {"reason": "complex", "summary": "Needs help", "priority": "urgent"}
    )
    assert escalation == {"ticket_id": "ticket-1", "status": "open"}
    assert ("product_search", "widget", 20, 25) in db.calls
    assert ("escalation", "complex") in db.calls


async def test_registry_rejects_unknown_unexpected_and_invalid_tools():
    registry = ToolRegistry(ToolContext(db=FakeDB()))

    async def handler(value: str) -> str:
        return value

    registry.register("example", "example", {}, handler)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("example", "duplicate", {}, handler)
    with pytest.raises(KeyError, match="Unknown tool"):
        await registry.execute("missing", {})
    with pytest.raises(ValueError, match="Unexpected arguments"):
        await registry.execute("example", {"unexpected": "value"})

    escalation = build_default_registry(ToolContext(db=FakeDB()))
    with pytest.raises(ValueError, match="priority"):
        await escalation.execute(
            "escalation", {"reason": "x", "summary": "y", "priority": "invalid"}
        )
