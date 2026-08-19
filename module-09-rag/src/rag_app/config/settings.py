import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "Semantic Search Engine"
    app_version: str = "0.1.0"
    debug: bool = False

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_directory: str = "src/rag_app/user_data/data"
    DATABASE_CONNECTION_CONVERSATION_URL: str = os.environ["DATABASE_CONNECTION_CONVERSATION_URL"]

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunking_strategy: str = "sentence"

    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Target chunk size in tokens",
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Number of overlapping tokens between chunks",
    )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"

    embedding_batch_size: int = Field(
        default=100,
        gt=0,
    )

    # ------------------------------------------------------------------
    # Embedding pricing
    #
    # Price is expressed as USD per 1 million input tokens.
    # Keep this configurable so we don't hardcode pricing in the
    # embedding manager.
    # ------------------------------------------------------------------
    embedding_cost_per_million_tokens: float = Field(
        default=0.0,
        ge=0,
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    default_top_k: int = Field(
        default=5,
        gt=0,
    )

    max_top_k: int = Field(
        default=100,
        gt=0,
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_directory: str = "src/rag_app/user_data/logs"

    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    evaluation_file: str = "src/rag_app/evaluation/evalution.json"

    # ------------------------------------------------------------------
    # Pydantic Settings
    # ------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
