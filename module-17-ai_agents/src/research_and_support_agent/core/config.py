"""Configuration and settings for Module 17 Bounded Research and Support Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Runtime configuration for bounded agent execution, memory, and storage."""

    # Model & Provider
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gemini-3.5-flash-lite")
    )
    default_provider: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "google")
    )

    # Execution Bounds
    default_max_steps: int = 10
    default_max_tokens: int = 10_000
    default_max_cost_usd: float = 0.05
    default_timeout_seconds: float = 30.0
    default_max_repeated_calls: int = 3

    # Storage Connections
    postgres_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_CONNECTION_CONVERSATION_URL",
            "postgresql://postgres:postgres@127.0.0.1:5432/ai_search",
        )
    )
    schema: str = "public"
    memory_table_prefix: str = "memory_"
    approval_table_prefix: str = "hitl_"

    # Web Search Configuration
    web_search_max_results: int = 3
    web_search_timeout_seconds: float = 10.0
    web_search_max_chars: int = 4000

    # Paths
    transcripts_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "transcripts"
    )
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data")


def get_agent_config() -> AgentConfig:
    """Retrieve global agent configuration instance."""
    return AgentConfig()
