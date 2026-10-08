from uuid import UUID

from pydantic import BaseModel


class ChatServiceRequestModel(BaseModel):
    conversation_id: UUID
    user_id: str
    request_id: UUID
    user_message: str
