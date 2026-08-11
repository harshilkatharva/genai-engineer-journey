from fastapi import APIRouter
from uuid import UUID
from ai_app.core.conversation_manager import ConversationManager

router = APIRouter()


@router.post("/conversation")
async def start_conversation(user_id: UUID):
    conversation_manager = ConversationManager()

    conversation_id = await conversation_manager.start_conversation(user_id)

    return conversation_id
