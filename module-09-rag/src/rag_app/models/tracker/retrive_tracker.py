from pydantic import BaseModel, Field


class RetriveTracker(BaseModel):
    tenant_id: str

    query: str

    top_k: int = Field(ge=1)
    results_count: int = Field(ge=0)

    retrieval_latency_ms: float = Field(ge=0)

    embedding_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
