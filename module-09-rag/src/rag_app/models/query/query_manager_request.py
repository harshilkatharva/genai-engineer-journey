from typing import Literal

from pydantic import BaseModel


class QueryManagerRequest(BaseModel):
    query: str
    technique: (
        Literal[
            "query_expansion",
            "query_hyde",
        ]
        | None
    ) = None
