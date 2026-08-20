from pydantic import BaseModel, Field


class ChunkingTracker(BaseModel):
    tenant_id: str

    document_count: int = Field(ge=0)
    total_chunks: int = Field(ge=0)

    chunking_strategy: str
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)

    total_input_tokens: int = Field(default=0, ge=0)

    latency_ms: float = Field(ge=0)
