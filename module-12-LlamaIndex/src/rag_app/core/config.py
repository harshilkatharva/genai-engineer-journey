from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the RAG application."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "multi-index-rag"
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    index_storage_dir: Path = Path(__file__).resolve().parent.parent / "storage"
    postgres_url: str = Field(validation_alias="DATABASE_CONNECTION_CONVERSATION_URL")
    vector_table_name: str = "rag_documents_LlmaIndex"
    vector_schema_name: str = "public"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_top_n: int = 5
    similarity_top_k: int = 20
    llm_model: str = "gemini-3.5-flash-lite"
    chunk_size: int = 500
    chunk_overlap: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
