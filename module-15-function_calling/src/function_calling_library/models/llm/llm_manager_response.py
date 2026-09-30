from typing import Any

from pydantic import BaseModel, Field

from .llm_response_model import LLMError, ToolCall


class LLMManagerResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    error: LLMError | None = None
    raw_response: Any | None = None
