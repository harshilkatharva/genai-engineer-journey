from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    conversation_id: str

    chunk_index: int = Field(ge=0)

    text: str
    token_count: int = Field(ge=0)

    start_position: int | None = Field(default=None, ge=0)
    end_position: int | None = Field(default=None, ge=0)
