from pydantic import BaseModel, Field


class RequestModel(BaseModel):
    tenant_id: str = Field(min_length=1)
