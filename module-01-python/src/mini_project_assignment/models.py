from pydantic import BaseModel, Field


class CompletionResult(BaseModel):
    """
    Standard Response model for all providers
    """

    text: str = Field(..., description="Genrated Response text by provider")

    provider: str = Field(..., description="Name of Provider")

    latency_ms: float = Field(..., description="Request latency in milisecond")

    token_usage: int = Field(
        ..., description="Total number of token consumed in request"
    )
