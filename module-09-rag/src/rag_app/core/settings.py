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
    data_directory: str = "user_data/data"

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
    DATABASE_CONNECTION_CONVERSATION_URL: str = Field(
        default="postgresql://postgres:postgres@127.0.0.1:5432/ai_search"
    )

    embedding_model: str = "all-MiniLM-L6-v2"
    default_embedding_model: str = "all-MiniLM-L6-v2"

    embedding_batch_size: int = Field(
        default=100,
        gt=0,
    )

    # ------------------------------------------------------------------
    # Embedding pricing
    # Price is expressed as USD per 1 million input tokens.
    # ------------------------------------------------------------------
    embedding_cost_per_million_tokens: float = Field(
        default=0.0,
        ge=0,
    )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    default_query_strategy: str | None = Field(
        default="query_HyDE",
    )

    default_hyde_strategy: str = Field(
        default="Google",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    default_retrieval_strategy: str = Field(
        default="vector_search",
    )

    canidate_default_top_k: int = Field(
        default=5,
        gt=0,
    )

    canidate_max_top_k: int = Field(
        default=100,
        gt=0,
    )

    # ------------------------------------------------------------------
    # ReRanker
    # ------------------------------------------------------------------

    re_ranker_availability: bool = Field(
        default=False,
    )

    re_ranker_model: str = Field(
        default="",
    )

    re_ranker_default_top_k: int = Field(
        default=5,
        gt=0,
    )

    re_ranker_max_top_k: int = Field(
        default=20,
        gt=0,
    )

    # ------------------------------------------------------------------
    # LLM Providers
    # ------------------------------------------------------------------
    default_llm_provider: str = Field(default="google")
    default_llm_model: str = Field(default="gemini-3.5-flash-lite")

    default_fallback_provider: str = Field(default="google")
    default_fallback_model: str = Field(default="gemini-3.5-flash")

    default_llm_temperature: float = Field(default=0.2)

    # ------------------------------------------------------------------
    # Latest Prompt file/version
    # ------------------------------------------------------------------
    rag_prompt_running_version: str = Field(default="rag_v1.md")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_directory: str = "logs"

    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    evaluation_file_path: str = "src/rag_app/evaluation/evalution.json"

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
