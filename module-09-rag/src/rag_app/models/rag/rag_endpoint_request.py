from pydantic import BaseModel
from uuid import UUID


class RAGEndpointRequest(BaseModel):
    query: str
    tenant_id: UUID
