from pydantic import BaseModel, Field


class EmbeddingTracker(BaseModel):
    tenant_id: str

    embedding_model: str

    total_chunks: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    latency_ms: float = Field(ge=0)

    estimated_cost: float = Field(default=0.0, ge=0)
