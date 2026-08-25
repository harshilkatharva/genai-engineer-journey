from typing import Any

from pydantic import BaseModel


class LLMManagerResponse(BaseModel):
    text: str
    data: dict[str, Any] | None = None
