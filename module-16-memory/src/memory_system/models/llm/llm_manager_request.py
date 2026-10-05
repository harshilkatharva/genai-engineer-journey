from pydantic import BaseModel, Field

from ..tool import ToolChoice, ToolSpec
from .llm_response_model import ChatMessage


class LLMManagerRequest(BaseModel):
    provider: str | None = None
    messages: list[ChatMessage]
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: ToolChoice = Field(default_factory=ToolChoice)
