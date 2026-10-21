from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Normalized structured output schema emitted by LLM evaluators."""

    score: float = Field(
        ge=0.0, le=1.0, description="Score from 0.0 to 1.0 evaluating the target property."
    )
    passed: bool = Field(description="Whether the output meets acceptable quality threshold.")
    reasoning: str = Field(description="Concise rationale explaining the evaluation score.")
