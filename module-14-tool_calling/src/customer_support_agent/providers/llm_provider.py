from collections.abc import AsyncIterator
from typing import Protocol

from customer_support_agent.models import (
    ChatMessage,
    LLMResponseModel,
    ToolDefinition,
)


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponseModel: ...

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...
