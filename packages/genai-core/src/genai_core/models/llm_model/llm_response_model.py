from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Any = Field(default_factory=dict)
    thought_signature: bytes | str | None = None


class LLMErrorInfo(BaseModel):
    provider: str
    code: str
    message: str
    status_code: int | None = None
    retryable: bool = False


LLMError = LLMErrorInfo


class LLMResponseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    structured_response: Any | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    raw_response: Any | None = None
