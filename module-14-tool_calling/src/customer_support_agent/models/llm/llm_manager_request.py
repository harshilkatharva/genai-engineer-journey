from pydantic import BaseModel, Field

from .llm_response_model import ChatMessage, ToolDefinition


class LLMManagerRequest(BaseModel):
    provider: str | None = None
    messages: list[ChatMessage]
    tools: list[ToolDefinition] = Field(default_factory=list)
