import pytest
from pydantic import ValidationError

from customer_support_agent.models import (
    CancellationPolicy,
    ChatMessage,
    LLMManagerRequest,
    LLMManagerResponse,
    LLMResponseModel,
    Order,
    Product,
    SupportDocument,
    ToolCall,
    ToolDefinition,
)


def test_chat_and_tool_models_validate_and_dump():
    message = ChatMessage(role="user", content="hello")
    call = ToolCall(id="c1", name="lookup_order", arguments={"order_id": "o1"})
    definition = ToolDefinition(name="lookup_order", description="lookup", parameters={})
    assert message.model_dump()["role"] == "user"
    assert call.model_dump()["arguments"]["order_id"] == "o1"
    assert definition.model_dump()["name"] == "lookup_order"
    with pytest.raises(ValidationError):
        ChatMessage(role="invalid", content="hello")


def test_llm_request_response_models_defaults_and_nested_values():
    request = LLMManagerRequest(messages=[ChatMessage(role="user", content="hi")])
    response = LLMResponseModel(model="fake", latency_ms=1.0)
    managed = LLMManagerResponse(model="fake", tool_calls=[ToolCall(id="c", name="x")])
    assert request.tools == []
    assert response.tool_calls == []
    assert managed.model_dump()["tool_calls"][0]["name"] == "x"


def test_support_models_validate_and_serialize():
    order = Order(order_id="o1", status="shipped", items=[{"sku": "s1"}])
    product = Product(product_id="p1", name="Widget", price=10)
    document = SupportDocument(chunk_id="d1", content="policy", document_type="faq")
    policy = CancellationPolicy(policy_name="cancel", content="Contact support")
    assert order.model_dump()["items"] == [{"sku": "s1"}]
    assert product.model_dump()["price"] == 10
    assert document.model_dump()["document_type"] == "faq"
    assert policy.model_dump()["content"] == "Contact support"
