from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "E-commerce Customer Support Agent"
    default_llm_provider: str = "google"
    default_llm_model: str = "gemini-3.5-flash-lite"
    default_llm_temperature: float = 0.2
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_search"
    log_level: str = "INFO"
    log_directory: str = "logs"
    tool_call_log_file: str = "logs/tool_calls.jsonl"
    max_tool_iterations: int = 5
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
