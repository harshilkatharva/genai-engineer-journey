import json
import time
from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_app.features.rag_classification import RAGClassification
from rag_app.features.rag_extraction import RAGExtraction

# from rag_app.features.rag_chat_langchain import RAGChatLangchain
from rag_app.features.rag_langgraph import RAGChatLanggraph
from rag_app.models import RAGEndpointRequest, RAGRequest, RAGTextRequest
from rag_app.observability.logger import logger


class RAGReviewRequest(BaseModel):
    query: str
    tenant_id: UUID
    session_id: str = "default"
    decision: Literal["approved", "rejected"]


class LazyRAGChat:
    def __init__(self):
        self._service: RAGChatLanggraph | None = None

    async def _get_service(self) -> RAGChatLanggraph:
        if self._service is None:
            self._service = await RAGChatLanggraph.from_postgres()
        return self._service

    async def get_chat_answer(self, request: RAGRequest):
        return await (await self._get_service()).get_chat_answer(request)

    async def stream_chat_answer(self, request: RAGRequest):
        service = await self._get_service()
        async for update in service.stream_chat_answer(request):
            yield update

    async def resume_chat_answer(self, request: RAGRequest, decision: str):
        return await (await self._get_service()).resume_chat_answer(request, decision)


rag_chat = LazyRAGChat()
rag_classification = RAGClassification()
rag_extraction = RAGExtraction()
router = APIRouter()


def get_rag_chat() -> LazyRAGChat:
    return rag_chat


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
        answer = await get_rag_chat().get_chat_answer(
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


@router.post("/chat_answer/stream")
async def stream_answer(request: RAGEndpointRequest):
    async def events():
        async for update in get_rag_chat().stream_chat_answer(
            RAGRequest(
                query=request.query,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
            )
        ):
            yield json.dumps(update, default=str) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.post("/chat_answer/review")
async def review_answer(request: RAGReviewRequest):
    return await get_rag_chat().resume_chat_answer(
        RAGRequest(
            query=request.query,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
        ),
        request.decision,
    )


@router.post("/classification")
async def classify(request: RAGTextRequest):
    return await rag_classification.classify(request.text)


@router.post("/extraction")
async def extract(request: RAGTextRequest):
    return await rag_extraction.extract(request.text)
