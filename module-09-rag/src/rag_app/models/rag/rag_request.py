from uuid import UUID

from pydantic import BaseModel


class RAGRequest(BaseModel):
    query: str
    tenant_id: UUID
    request_id: UUID
