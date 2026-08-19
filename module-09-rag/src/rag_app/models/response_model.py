from pydantic import BaseModel


class ResponseModel(BaseModel):
    success: bool = True
    message: str | None = None
