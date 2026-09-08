from uuid import UUID

from pydantic import BaseModel, field_validator


class RAGEndpointRequest(BaseModel):
    query: str
    tenant_id: UUID
    session_id: str = "default"

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value):
        if isinstance(value, dict):
            value = value.get("query", value.get("text"))

        return value
