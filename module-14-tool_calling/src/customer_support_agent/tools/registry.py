from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from customer_support_agent.models import ToolDefinition

ToolHandler = Callable[..., Awaitable[Any]]


class ToolContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    db: Any
    customer_id: str | None = None


class RegisteredTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    definition: ToolDefinition
    handler: ToolHandler


class ToolRegistry:
    """Typed catalogue of tools exposed to the model and their handlers."""

    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self, name: str, description: str, parameters: dict[str, Any], handler: ToolHandler
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = RegisteredTool(
            definition=ToolDefinition(name=name, description=description, parameters=parameters),
            handler=handler,
        )

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        # Validation is intentionally performed by the handler's typed signature where
        # possible; reject unexpected arguments before calling application code.
        signature = inspect.signature(tool.handler)
        allowed = set(signature.parameters)
        unexpected = set(arguments) - allowed
        if unexpected:
            raise ValueError(f"Unexpected arguments for {name}: {', '.join(sorted(unexpected))}")
        return await tool.handler(**arguments)


def build_default_registry(context: ToolContext) -> ToolRegistry:
    registry = ToolRegistry(context)
    db = context.db

    async def lookup_order(order_id: str) -> dict[str, Any]:
        order = await db.get_order(order_id)
        return {
            "found": order is not None,
            "order": order.model_dump(mode="json") if order else None,
        }

    async def cancellation_policy() -> dict[str, Any]:
        policy = await db.get_cancellation_policy()
        return {"found": policy is not None, "policy": policy.model_dump() if policy else None}

    async def product_search(
        query: str, max_price: float | None = None, limit: int = 10
    ) -> dict[str, Any]:
        products = await db.search_products(
            query, max_price=max_price, limit=min(max(limit, 1), 25)
        )
        return {"products": [product.model_dump() for product in products]}

    async def product_details(product_id: str) -> dict[str, Any]:
        product = await db.get_product(product_id)
        return {"found": product is not None, "product": product.model_dump() if product else None}

    async def escalation(reason: str, summary: str, priority: str = "normal") -> dict[str, str]:
        if priority not in {"low", "normal", "high", "urgent"}:
            raise ValueError("priority must be low, normal, high, or urgent")
        ticket_id = await db.create_escalation(context.customer_id, reason, summary, priority)
        return {"ticket_id": ticket_id, "status": "open"}

    registry.register(
        "lookup_order",
        "What it does: looks up an order by its order ID. "
        "Use when: the customer asks about an order's status or details. "
        "Do not use when: the customer has not provided an order ID0 or ask about product details..",
        {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        lookup_order,
    )
    registry.register(
        "cancellation_policy",
        "What it does: retrieves the current order cancellation policy. "
        "Use when: the customer asks whether or how an order can be cancelled. "
        "Do not use when: the question is about another policy or a specific order status or status about cancelled orders.",
        {"type": "object", "properties": {}},
        cancellation_policy,
    )
    registry.register(
        "product_search",
        "What it does: searches products by text and an optional maximum price. "
        "Use when: the customer wants to find products matching a description or budget. "
        "Do not use when: the customer already knows the product ID.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_price": {"type": "number"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        product_search,
    )
    registry.register(
        "product_details",
        "What it does: retrieves details for one product by product ID. "
        "Use when: the customer provides a product ID and asks about that product. "
        "Do not use when: the customer is searching by name, description, or budget.",
        {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
        },
        product_details,
    )
    registry.register(
        "escalation",
        "What it does: creates a support escalation ticket. "
        "Use when: the issue cannot be resolved with the available support tools or the customer requests a human. "
        "Do not use when: a lookup or policy answer can resolve the request without escalation.",
        {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "summary": {"type": "string"},
                "priority": {"type": "string"},
            },
            "required": ["reason", "summary"],
        },
        escalation,
    )
    return registry
