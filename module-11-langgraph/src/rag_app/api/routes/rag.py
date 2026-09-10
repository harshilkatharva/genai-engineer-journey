import time

from fastapi import APIRouter

from rag_app.features.rag_classification import RAGClassification
from rag_app.features.rag_chat_langchain import RAGChatLangchain
from rag_app.features.rag_extraction import RAGExtraction
from rag_app.models import RAGEndpointRequest, RAGRequest, RAGTextRequest
from rag_app.observability.logger import logger

rag_chat = RAGChatLangchain()
rag_classification = RAGClassification()
rag_extraction = RAGExtraction()
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
            RAGRequest(
                query=request.query,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
            )
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


@router.post("/classification")
async def classify(request: RAGTextRequest):
    return await rag_classification.classify(request.text)


@router.post("/extraction")
async def extract(request: RAGTextRequest):
    return await rag_extraction.extract(request.text)
