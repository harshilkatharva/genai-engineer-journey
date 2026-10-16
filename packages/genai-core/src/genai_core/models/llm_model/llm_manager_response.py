from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .llm_response_model import LLMError, ToolCall


class LLMManagerResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    structured_response: Any | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    error: LLMError | None = None
    raw_response: Any | None = None
