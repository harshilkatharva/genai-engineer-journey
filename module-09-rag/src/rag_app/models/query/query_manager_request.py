from pydantic import BaseModel
from typing import Literal


class QueryManagerRequest(BaseModel):
    query: str
    technique: (
        Literal[
            "query_expansion",
            "query_hyde",
        ]
        | None
    ) = None
