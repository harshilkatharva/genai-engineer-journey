from collections.abc import AsyncIterator
from typing import Protocol

from llm_client.models.response_model import CompletionResult


class LLMProvider(Protocol):
    async def complete(self, prompt: str) -> CompletionResult: ...

    def stream(self, prompt: str) -> AsyncIterator[str]: ...
