from pydantic import BaseModel, Field


class DBQueryTracker(BaseModel):
    tenant_id: str

    top_k: int = Field(ge=1)
    results_count: int = Field(ge=0)

    query_latency_ms: float = Field(ge=0)
    chunk_ids: list[str]
