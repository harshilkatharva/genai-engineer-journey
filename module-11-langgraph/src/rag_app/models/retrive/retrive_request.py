from uuid import UUID

from pydantic import BaseModel, Field

from rag_app.core import get_settings

settings = get_settings()


class RetriveRequest(BaseModel):
    tenant_id: UUID
    query: str = Field(description="The query to retrieve relevant documents for.")
    top_k_candidate: int = Field(
        default=settings.canidate_default_top_k, ge=1, le=settings.canidate_max_top_k
    )
    top_k_re_ranker: int = Field(
        default=settings.re_ranker_default_top_k, ge=1, le=settings.re_ranker_max_top_k
    )
