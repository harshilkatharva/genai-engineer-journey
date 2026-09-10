from pydantic import BaseModel


class RAGResposne(BaseModel):
    text: str
    source: list[str] | str | None = None
