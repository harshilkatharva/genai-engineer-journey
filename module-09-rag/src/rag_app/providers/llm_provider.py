from collections.abc import AsyncIterator
from typing import Protocol

from rag_app.models import LLMResponseModel


class LLMProvider(Protocol):
    async def complete(self, prompt: str) -> LLMResponseModel: ...

    def stream(self, prompt: str) -> AsyncIterator[str]: ...
