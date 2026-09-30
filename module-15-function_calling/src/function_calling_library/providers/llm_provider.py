from collections.abc import AsyncIterator
from typing import Protocol

from ..models import (
    ChatMessage,
    LLMResponseModel,
    ToolChoice,
    ToolSpec,
)


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponseModel: ...

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...
