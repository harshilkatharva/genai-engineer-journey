from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "GenAI Core"
    default_llm_provider: str = "google"
    default_llm_model: str = "gemini-3.5-flash-lite"
    default_llm_temperature: float = 0.2
    max_structured_error_retry: int = 2
    # Embedding settings
    default_embedding_provider: str = "local"
    default_embedding_model: str = "all-MiniLM-L6-v2"
    default_embedding_dimensions: int = 384
    default_embedding_batch_size: int = 32
    default_google_embedding_model: str = "gemini-embedding-001"
    default_openai_embedding_model: str = "text-embedding-3-small"

    # Embedding resilience settings
    embedding_max_retries: int = 3
    embedding_initial_backoff: float = 0.5
    embedding_max_backoff: float = 10.0
    embedding_timeout_seconds: float = 30.0
    embedding_rate_limit_rpm: int | None = None
    embedding_rate_limit_tpm: int | None = None

    # Retrieval settings
    default_retrieval_top_k: int = 5
    default_retrieval_score_threshold: float | None = None
    retrieval_timeout_seconds: float = 30.0
    default_vector_weight: float = 0.6
    default_keyword_weight: float = 0.4
    default_text_search_language: str = "english"
    enable_reranking: bool = False
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_batch_size: int = 32

    # Memory settings
    memory_working_ttl_hours: int = 24
    memory_token_budget: int = 4000
    memory_recency_half_life_days: int = 30
    memory_episodic_retention_days: int = 14

    # Integration and API keys
    integration_test: int = 0
    google_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
