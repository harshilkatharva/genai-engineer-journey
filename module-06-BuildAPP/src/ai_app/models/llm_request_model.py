from typing import Literal

from pydantic import BaseModel, Field


class LLMRequestModel(BaseModel):
    """
    Standard Request model for all providers
    """

    provider: Literal["google", "openai", "anthropic"] = Field(
        ..., description="Providers of LLM from this :- google, openai, anthropic"
    )

    prompt: str = Field(
        min_length=1,
        description="Prompt contains system prompt + conversation history + new message",
    )
