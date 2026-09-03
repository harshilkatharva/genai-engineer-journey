from uuid import UUID

from pydantic import BaseModel


class RAGEndpointRequest(BaseModel):
    query: str
    tenant_id: UUID
