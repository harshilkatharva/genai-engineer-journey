from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..tools_model.tool_spec import ToolChoice, ToolSpec
from .llm_response_model import ChatMessage


class LLMManagerRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: str | None = None
    prompt: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: ToolChoice = Field(default_factory=ToolChoice)
    response_schema: type[BaseModel] | None = None
