import time

from fastapi import APIRouter

from rag_app.features.rag_chat import RAGChat
from rag_app.models import RAGEndpointRequest, RAGRequest
from rag_app.observability.logger import logger

rag_chat = RAGChat()

router = APIRouter()


@router.post("/chat_answer")
async def get_answer(request: RAGEndpointRequest):
    logger.info(
        "RAG chat request started",
        event="chat_request_started",
        component="api",
        endpoint="/rag/chat_answer",
        tenant_id=str(request.tenant_id),
    )

    try:
        start = time.perf_counter()
        answer = await rag_chat.get_chat_answer(
            RAGRequest(query=request.query, tenant_id=request.tenant_id)
        )
        end = time.perf_counter()
        logger.info(
            "RAG chat request completed",
            event="chat_request_completed",
            component="api",
            endpoint="/rag/chat_answer",
            latency_ms=(end - start) * 1000,
            status="success",
        )

        return answer

    except Exception as exc:
        logger.exception(
            "RAG request failed",
            event="request_failed",
            component="api",
            endpoint="/rag/chat_answer",
            status="error",
            error_type=type(exc).__name__,
        )

        raise
