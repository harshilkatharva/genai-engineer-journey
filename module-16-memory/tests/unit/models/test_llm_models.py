from __future__ import annotations

import pytest
from pydantic import ValidationError

from memory_system.models import (
    ChatMessage,
    LLMError,
    LLMManagerRequest,
    LLMManagerResponse,
    LLMResponseModel,
    ToolCall,
    ToolChoice,
)


def test_llm_messages_and_tool_calls_have_expected_defaults() -> None:
    message = ChatMessage(role="tool", content="done", tool_call_id="call-1", tool_name="search")
    call = ToolCall(id="call-1", name="search")

    assert message.tool_call_id == "call-1"
    assert call.arguments == {}
    assert LLMResponseModel(model="model", latency_ms=1.5).input_tokens == 0


def test_llm_manager_request_and_response_defaults() -> None:
    request = LLMManagerRequest(messages=[ChatMessage(role="user", content="hello")])
    response = LLMManagerResponse(text="hello")

    assert request.tools == []
    assert request.tool_choice == ToolChoice()
    assert response.tool_calls == []
    assert response.usage == {}
    assert response.error is None
    for role in ("invalid", "developer"):
        with pytest.raises(ValidationError):
            ChatMessage.model_validate({"role": role, "content": "message"})
    error = LLMError(
        provider="openai", code="rate_limit", message="Try later", status_code=429, retryable=True
    )
    response = LLMManagerResponse(error=error, raw_response={"request": "id"})

    assert response.error == error
    assert response.raw_response == {"request": "id"}
    with pytest.raises(ValidationError):
        LLMError.model_validate(
            {"provider": "openai", "code": "bad", "message": "bad", "status_code": "not-a-code"}
        )
