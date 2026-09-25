from __future__ import annotations

from customer_support_agent.models import (
    CancellationPolicy,
    LLMResponseModel,
    Order,
    Product,
    ToolCall,
)


class FakeDB:
    def __init__(self):
        self.calls = []

    async def get_order(self, order_id: str):
        self.calls.append(("lookup_order", order_id))
        return Order(order_id=order_id, status="shipped")

    async def get_cancellation_policy(self):
        self.calls.append(("cancellation_policy",))
        return CancellationPolicy(policy_name="cancellation", content="Contact support.")

    async def search_products(self, query, max_price=None, limit=10):
        self.calls.append(("product_search", query, max_price, limit))
        return [Product(product_id="p-1", name="Widget", price=10)]

    async def get_product(self, product_id):
        self.calls.append(("product_details", product_id))
        return Product(product_id=product_id, name="Widget", price=10)

    async def create_escalation(self, customer_id, reason, summary, priority="normal"):
        self.calls.append(("escalation", reason))
        return "ticket-1"


class FakeProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponseModel(
                model="fake",
                latency_ms=0,
                tool_calls=[
                    ToolCall(id="call-1", name="lookup_order", arguments={"order_id": "order-1"})
                ],
            )
        return LLMResponseModel(model="fake", latency_ms=0, text="Your order is shipped.")


class SequencedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen_messages = []

    async def complete(self, messages, tools=None):
        self.seen_messages.append(list(messages))
        return next(self.responses)


class FailingDB(FakeDB):
    async def get_order(self, order_id: str):
        raise RuntimeError("database password leaked")
