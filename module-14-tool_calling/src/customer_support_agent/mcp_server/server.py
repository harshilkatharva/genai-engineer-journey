from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from customer_support_agent.db import CustomerSupportDB

mcp = MCPServer("E-commerce Customer Support")
db = CustomerSupportDB()


@mcp.tool(
    description=(
        "Look up one order by ID. Use when the customer asks about a specific order "
        "and provides its order ID. Do not use for product searches or general policy questions."
    )
)
async def lookup_order(order_id: str) -> dict[str, Any]:
    order = await db.get_order(order_id)
    return {"found": order is not None, "order": order.model_dump(mode="json") if order else None}


@mcp.tool(
    description=(
        "Search the product catalogue by description or budget. Use when the customer "
        "is looking for products. Do not use when a known product ID is available."
    )
)
async def product_search(
    query: str, max_price: float | None = None, limit: int = 10
) -> dict[str, Any]:
    products = await db.search_products(query, max_price=max_price, limit=min(max(limit, 1), 25))
    return {"products": [product.model_dump(mode="json") for product in products]}


@mcp.tool(
    description=(
        "Retrieve one product by product ID. Use when the customer provides a product ID "
        "and asks for its details. Do not use for name, description, or budget searches."
    )
)
async def product_details(product_id: str) -> dict[str, Any]:
    product = await db.get_product(product_id)
    return {
        "found": product is not None,
        "product": product.model_dump(mode="json") if product else None,
    }


@mcp.tool(
    description=(
        "Create a human-support escalation. Use when the customer requests a human or "
        "available tools cannot resolve the issue. Do not use when a lookup can answer it."
    )
)
async def escalation(reason: str, summary: str, priority: str = "normal") -> dict[str, str]:
    if priority not in {"low", "normal", "high", "urgent"}:
        raise ValueError("priority must be low, normal, high, or urgent")
    ticket_id = await db.create_escalation(
        os.getenv("CUSTOMER_ID") or None, reason, summary, priority
    )
    return {"ticket_id": ticket_id, "status": "open"}


@mcp.resource("support://cancellation-policy")
def cancellation_policy() -> str:
    # In a real application, this would likely be stored in a database or fetched from an API.
    policy_path = (
        Path(__file__).resolve().parents[3] / "data" / "daily_deal_product_cancellation_policy.txt"
    )
    return policy_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="stdio")
