from uuid import UUID

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: UUID
    tenant_id: UUID

    document_type: str
    metadata: dict

    chunk_index: int = Field(ge=0)

    text: str
    token_count: int = Field(ge=0)

    start_position: int | None = Field(default=None, ge=0)
    end_position: int | None = Field(default=None, ge=0)
