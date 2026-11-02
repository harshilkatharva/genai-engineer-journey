"""Configuration settings for Self-Improving Report Writer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReportWriterConfig(BaseModel):
    """Runtime configuration for report generation, verification, and reflection."""

    generator_provider: str | None = Field(
        default=None, description="LLM provider for report generation and revision"
    )
    generator_model: str = Field(
        default="gemini-3.5-flash-lite", description="Model name for generation"
    )
    judge_provider: str | None = Field(
        default=None, description="LLM provider for rubric judging and semantic evaluation"
    )
    judge_model: str = Field(
        default="gemini-2.5-pro", description="Model name for independent judging"
    )
    default_top_k: int = Field(
        default=8, ge=1, le=50, description="Default number of source chunks to retrieve"
    )
    min_retrieval_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum relevance threshold for retrieved sources"
    )
    max_reflection_rounds: int = Field(
        default=3, ge=1, le=10, description="Maximum iterative reflection rounds"
    )
    max_total_tokens: int = Field(
        default=50000, ge=1000, description="Run-wide token consumption budget"
    )
    max_total_cost_usd: float = Field(
        default=0.20, ge=0.01, description="Run-wide USD financial cost budget"
    )
    max_wall_clock_seconds: float = Field(
        default=90.0, ge=5.0, description="Run-wide execution deadline in seconds"
    )
    max_consecutive_regressions: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum consecutive rounds where candidate quality degrades",
    )
    acceptance_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Composite score required for acceptance"
    )


def get_default_config() -> ReportWriterConfig:
    """Return default configuration instance."""
    return ReportWriterConfig()
