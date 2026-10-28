"""Configuration settings for Plan-and-Execute Research Assistant."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AssistantConfig(BaseSettings):
    """Central configuration for planning, execution, verification, and memory."""

    model_config = SettingsConfigDict(
        env_prefix="ASSISTANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model and Provider
    default_model: str = Field(
        default="gemini-3.5-flash-lite",
        validation_alias=AliasChoices(
            "assistant_default_model", "default_model", "default_llm_model"
        ),
        description="Default LLM model identifier for planning and verification",
    )
    default_provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "assistant_default_provider", "default_provider", "default_llm_provider"
        ),
        description="Optional provider identifier (e.g. 'google', 'anthropic')",
    )
    default_temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="Sampling temperature for planning"
    )

    # Planning & Retries
    max_planning_retries: int = Field(
        default=3, ge=1, description="Maximum automated structural repair attempts during planning"
    )

    # Concurrency & Scheduling
    max_concurrency: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices("assistant_max_concurrency", "max_concurrency"),
        description="Maximum concurrent active tasks during DAG execution",
    )
    auto_approve_review: bool = Field(
        default=False,
        description="Whether to automatically approve plans during review (for testing/CI)",
    )

    # Step Execution & Retries
    default_step_max_iterations: int = Field(
        default=5,
        ge=1,
        validation_alias=AliasChoices(
            "assistant_default_step_max_iterations",
            "default_step_max_iterations",
            "max_tool_iterations",
        ),
        description="Default max loop iterations per step execution",
    )
    max_step_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum execution retries for an individual step before replan",
    )
    max_replans: int = Field(
        default=3, ge=0, description="Maximum dynamic replanning cycles permitted per run"
    )

    # Global Run Limits
    max_total_tokens: int = Field(
        default=40_000,
        ge=100,
        validation_alias=AliasChoices("assistant_max_total_tokens", "max_total_tokens"),
        description="Global cumulative token ceiling across all phases",
    )
    max_total_cost_usd: float = Field(
        default=0.25,
        ge=0.001,
        validation_alias=AliasChoices("assistant_max_total_cost_usd", "max_total_cost_usd"),
        description="Global cumulative USD cost budget across all phases",
    )
    default_timeout_seconds: float = Field(
        default=180.0,
        gt=0.0,
        validation_alias=AliasChoices(
            "assistant_default_timeout_seconds",
            "default_timeout_seconds",
            "timeout_seconds",
        ),
        description="Maximum wall-clock execution duration in seconds",
    )
    max_web_retrieved_tokens: int = Field(
        default=5000,
        ge=100,
        description="Maximum token ceiling for web-retrieved data and search tool responses",
    )

    # Verification Thresholds
    verification_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required from model judge to verify a step",
    )

    # Memory Settings
    enable_episodic_similarity_search: bool = Field(
        default=False,
        description="Config-based toggle: whether to search past episodes via semantic similarity. "
        "Defaults to False until genai-core natively provides episodic vector search.",
    )
    max_recalled_preferences: int = Field(
        default=5, ge=1, description="Maximum long-term user preferences to recall during planning"
    )
    max_recalled_past_plans: int = Field(
        default=3, ge=1, description="Maximum similar past plan episodes to recall during planning"
    )

    # Persistence Directories
    runs_dir: Path = Field(
        default=Path("runs"), description="Directory where RunRecords are persisted as JSON"
    )
    transcripts_dir: Path = Field(
        default=Path("transcripts"), description="Directory where agent transcripts are saved"
    )


@lru_cache(maxsize=1)
def get_assistant_config() -> AssistantConfig:
    """Return a cached singleton instance of AssistantConfig."""
    return AssistantConfig()
