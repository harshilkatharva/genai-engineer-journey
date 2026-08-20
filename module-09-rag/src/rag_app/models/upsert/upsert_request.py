from uuid import UUID

from pydantic import BaseModel, model_validator


class UpsertRequest(BaseModel):
    tenant_id: UUID
    document_ids: list[UUID]
    chunk_ids: list[str]
    updated_chunks: list[str]

    @model_validator(mode="after")
    def validate_equal_lengths(self) -> "UpsertRequest":
        n_docs = len(self.document_ids)
        n_chunks = len(self.chunk_ids)
        n_texts = len(self.updated_chunks)
        if not (n_docs == n_chunks == n_texts):
            raise ValueError(
                f"Array length mismatch: document_ids ({n_docs}), "
                f"chunk_ids ({n_chunks}), updated_chunks ({n_texts}) must have identical lengths."
            )
        return self
