from collections.abc import AsyncIterator
from typing import Protocol
from pydantic import BaseModel

from rag_app.models import LLMResponseModel


class LLMProvider(Protocol):
    async def complete(
        self, prompt: str, response_schema: type[BaseModel] | None = None
    ) -> LLMResponseModel: ...

    def stream(self, prompt: str) -> AsyncIterator[str]: ...
