from typing import Any

from pydantic import BaseModel, Field


class LLMResponseModel(BaseModel):
    """
    Standard Response model for all providers
    """

    text: str | None = Field(..., description="Genrated Response text by provider")

    data: dict[str, Any] | None = Field(default=None, description="Formated data from LLMc")

    model: str = Field(..., description="Name of model")

    latency_ms: float = Field(..., description="Request latency in milisecond")

    input_tokens: int = Field(..., description="Input token consumed in request")

    output_tokens: int = Field(..., description="Output token consumed in request")
