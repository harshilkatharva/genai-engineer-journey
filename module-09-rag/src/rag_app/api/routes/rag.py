from fastapi import APIRouter
import uuid
from rag_app.features.naive_rag_chat import RAGChat
from rag_app.models import RAGRequest, RAGEndpointRequest


rag_chat = RAGChat()

router = APIRouter()


@router.post("/chat_answer")
async def get_answer(request: RAGEndpointRequest):
    request_id = uuid.uuid4()
    answer = await rag_chat.get_chat_answer(
        RAGRequest(query=request.query, request_id=request_id, tenant_id=request.tenant_id)
    )

    return answer
