from pydantic import BaseModel
from rag_app.models import RetriveResult


class ReRankerResponse(BaseModel):
    chunks: list[RetriveResult]
