"""Centralized configuration for Parallel Research Team."""

from __future__ import annotations

import functools

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchTeamConfig(BaseSettings):
    """Centralized runtime and evaluation configuration for the Parallel Research Team."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # LLM & Generation Knobs
    # -------------------------------------------------------------------------
    default_provider: str = Field(default="google", description="Default LLM provider")
    default_model: str = Field(
        default="gemini-3.5-flash-lite", description="Default model for planning & workers"
    )
    synthesis_provider: str = Field(
        default="google", description="LLM provider for final report synthesis"
    )
    synthesis_model: str = Field(
        default="gemini-3.5-flash-lite", description="Model for final report synthesis"
    )

    # -------------------------------------------------------------------------
    # Reflection & Verification Judge Knobs
    # -------------------------------------------------------------------------
    judge_provider: str = Field(
        default="google", description="Provider for reflection verification judge"
    )
    judge_model: str = Field(
        default="gemini-3.1-flash-lite", description="Model for reflection verification judge"
    )
    enable_semantic_verification: bool = Field(
        default=True, description="Enable Gate 2 semantic grounding verification"
    )
    semantic_verification_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Minimum pass threshold for claim entailment"
    )

    # -------------------------------------------------------------------------
    # Reflection Loop Knobs (Self-Correction for Gate 2 Issues)
    # -------------------------------------------------------------------------
    enable_reflection: bool = Field(
        default=True,
        description="Enable self-correcting reflection loop on Gate 2 verification issues",
    )
    max_reflection_steps: int = Field(
        default=2, ge=1, le=5, description="Maximum self-correcting reflection revisions"
    )
    max_reflection_tokens: int = Field(
        default=15_000, ge=1_000, description="Dedicated token envelope for reflection loop"
    )
    max_reflection_cost_usd: float = Field(
        default=0.05, gt=0.0, description="Dedicated USD budget ceiling for reflection loop"
    )

    # -------------------------------------------------------------------------
    # Concurrency & Scaling Knobs
    # -------------------------------------------------------------------------
    max_concurrency: int = Field(
        default=3, ge=1, le=10, description="Max concurrent active workers"
    )
    max_workers: int = Field(
        default=4, ge=1, le=10, description="Max worker briefs per research run"
    )
    default_effort_level: str = Field(
        default="auto", description="Effort scaling mode: 'auto', 'direct', or 'parallel'"
    )

    # -------------------------------------------------------------------------
    # Worker Execution Envelope
    # -------------------------------------------------------------------------
    per_worker_timeout_seconds: float = Field(
        default=45.0, gt=0.0, description="Per-worker wall-clock deadline in seconds"
    )
    per_worker_max_steps: int = Field(
        default=5, ge=1, le=10, description="Per-worker tool iteration ceiling"
    )

    # -------------------------------------------------------------------------
    # Shared Run Budget Envelope
    # -------------------------------------------------------------------------
    max_total_tokens: int = Field(
        default=60_000, ge=1_000, description="Cumulative token limit across all agents"
    )
    max_total_cost_usd: float = Field(
        default=0.50, gt=0.0, description="Cumulative financial budget in USD"
    )
    max_wall_clock_seconds: float = Field(
        default=180.0, gt=0.0, description="Total run wall-clock timeout in seconds"
    )

    # -------------------------------------------------------------------------
    # Dual Retrieval Storage & Web Search
    # -------------------------------------------------------------------------
    tavily_api_key: str | None = Field(default=None, description="Tavily API key for web search")

    retrieval_mode: str = Field(
        default="in_memory", description="Retrieval mode: 'in_memory' or 'postgres'"
    )
    postgres_connection_string: str | None = Field(
        default=None, description="PostgreSQL connection string for PGVectorStoreAdapter"
    )
    embedding_provider: str = Field(
        default="google", description="Embedding provider for vector search"
    )
    embedding_model: str = Field(default="text-embedding-004", description="Embedding model name")
    embedding_dimensions: int = Field(default=768, ge=1, description="Embedding vector dimensions")

    # -------------------------------------------------------------------------
    # Checkpointing & Prompts
    # -------------------------------------------------------------------------
    checkpoint_dir: str = Field(
        default="checkpoints", description="Directory to store run checkpoints"
    )
    auto_checkpoint: bool = Field(
        default=True, description="Automatically persist checkpoints at milestones"
    )
    prompt_version: str = Field(default="v1", description="Active prompt template version")
    system_version: str = Field(
        default="1.0.0", description="Module 20 research team system generator version"
    )


@functools.lru_cache(maxsize=1)
def get_research_team_config() -> ResearchTeamConfig:
    """Return cached singleton instance of ResearchTeamConfig."""
    return ResearchTeamConfig()
