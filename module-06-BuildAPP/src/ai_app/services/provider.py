from collections.abc import AsyncIterator
from typing import Protocol

from ai_app.models.llm_response_model import LLMResponseModel


class LLMProvider(Protocol):
    async def complete(self, prompt: str) -> LLMResponseModel: ...

    def stream(self, prompt: str) -> AsyncIterator[str]: ...
