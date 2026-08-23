from pydantic import BaseModel
from typing import Any


class LLMManagerResponse(BaseModel):
    text: str
    data: dict[str, Any] | None = None
