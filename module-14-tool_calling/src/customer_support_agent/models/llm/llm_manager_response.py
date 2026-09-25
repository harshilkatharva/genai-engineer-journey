from typing import Any

from pydantic import BaseModel, Field

from .llm_response_model import ToolCall


class LLMManagerResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: Any | None = None
