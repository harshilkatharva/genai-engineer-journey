from pydantic import BaseModel, Field
from uuid import UUID

from rag_app.config import get_settings

settings = get_settings()


class CandidateRequest(BaseModel):
    tenant_id: UUID
    queries: list[str]
    top_k_candidate: int = Field(
        default=settings.canidate_default_top_k, ge=1, le=settings.canidate_max_top_k
    )
