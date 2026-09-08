from typing import Any

from pydantic import BaseModel, Field


class RAGExtractionResponse(BaseModel):
    fields: dict[str, Any] = Field(
        description="The meaningful fields extracted from the input text."
    )
