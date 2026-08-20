from pydantic import BaseModel
from rag_app.models import RetriveResult


class CandidateResponse(BaseModel):
    chunks: list[RetriveResult]
