from datetime import datetime

from pydantic import BaseModel, Field


class TrackerModel(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
