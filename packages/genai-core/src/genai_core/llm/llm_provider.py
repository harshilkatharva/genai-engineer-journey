from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import (
    ChatMessage,
    LLMResponseModel,
    ToolChoice,
    ToolSpec,
)


@runtime_checkable
class LLMProvider(Protocol):
    async def complete_structure(
        self,
        prompt: str,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponseModel: ...

    async def complete_tool(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponseModel: ...

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...
