from pydantic import BaseModel, Field


class QueryTracker(BaseModel):
    conversation_id: str

    query: str

    embedding_model: str

    query_token_count: int = Field(ge=0)

    embedding_latency_ms: float = Field(ge=0)
    retrieval_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)

    estimated_cost: float = Field(default=0.0, ge=0)

    top_k: int = Field(ge=1)
