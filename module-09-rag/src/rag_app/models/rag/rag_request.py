from pydantic import BaseModel
from uuid import UUID


class RAGRequest(BaseModel):
    query: str
    tenant_id: UUID
    request_id: UUID
