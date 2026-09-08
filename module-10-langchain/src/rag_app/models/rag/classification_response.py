from pydantic import BaseModel, Field


class RAGClassificationResponse(BaseModel):
    category: str = Field(description="The best matching category for the input text.")
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(description="A concise reason for the classification.")
