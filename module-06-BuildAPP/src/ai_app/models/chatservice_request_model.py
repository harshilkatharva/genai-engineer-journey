from pydantic import BaseModel
from uuid import UUID


class ChatServiceRequestModel(BaseModel):
    conversation_id: UUID
    user_id: str
    request_id: UUID
    user_message: str
