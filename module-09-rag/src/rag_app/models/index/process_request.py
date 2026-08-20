from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ProcessRequest(BaseModel):
    # tenant_id: str = Field(min_length=1)
    # documents: list[str] = Field(min_length=1)
    tenant_id: UUID = Field(description="Users tenant id")
    documents: list[str] = Field(min_length=1, description="Names of files or direct text")
    documents_type: list[str] = Field(description="Document type")
    meta_data: list[dict] = Field(description="Add meta data of each document")

    @model_validator(mode="after")
    def validate_document_fields(self):
        if not (len(self.documents) == len(self.documents_type) == len(self.meta_data)):
            raise ValueError("documents, documents_type, and meta_data must have the same length")

        return self
