from pydantic import BaseModel, Field


class IndexBatchTracker(BaseModel):
    tenant_id: str

    batch_number: int = Field(ge=1)
    batch_size: int = Field(gt=0)
    total_chunks: int = Field(gt=0)

    insertion_latency_ms: float = Field(ge=0)
