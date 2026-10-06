from collections.abc import AsyncIterator
from typing import Protocol

from ..models import (
    ChatMessage,
    LLMResponseModel,
    ToolChoice,
    ToolSpec,
)


class LLMProvider(Protocol):
    # Requests a complete model response with optional function tools.
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponseModel: ...

    # Streams response text incrementally from the selected model.
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...
