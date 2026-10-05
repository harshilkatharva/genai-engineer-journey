from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Multi-Layer Memory System"
    default_llm_provider: str = "google"
    default_llm_model: str = "gemini-3.5-flash-lite"
    default_llm_temperature: float = 0.2
    database_url: str = Field(
        default="postgresql://localhost:5432/memory",
        validation_alias=AliasChoices("DATABASE_CONNECTION_CONVERSATION_URL", "DATABASE_URL"),
    )
    pgvector_database_url: str | None = Field(
        default=None, validation_alias="DATABASE_CONNECTION_PGVECTOR_URL"
    )
    log_level: str = "INFO"
    log_directory: str = "logs"
    tool_call_log_file: str = "logs/tool_calls.jsonl"
    max_tool_iterations: int = 5
    google_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    memory_token_budget: int = Field(default=8_000, gt=0, validation_alias="MEMORY_TOKEN_BUDGET")
    memory_working_ttl_hours: int = Field(
        default=24, gt=0, validation_alias="MEMORY_WORKING_TTL_HOURS"
    )
    memory_episodic_retention_days: int = Field(
        default=90, gt=0, validation_alias="MEMORY_EPISODIC_RETENTION_DAYS"
    )
    memory_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
        validation_alias="MEMORY_EMBEDDING_MODEL",
    )
    memory_embedding_dimension: int = Field(
        default=384, gt=0, validation_alias="MEMORY_EMBEDDING_DIMENSION"
    )
    memory_retrieval_limit: int = Field(default=8, gt=0, validation_alias="MEMORY_RETRIEVAL_LIMIT")
    memory_recency_half_life_days: int = Field(
        default=30, gt=0, validation_alias="MEMORY_RECENCY_HALF_LIFE_DAYS"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
