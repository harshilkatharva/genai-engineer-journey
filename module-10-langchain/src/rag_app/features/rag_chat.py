import time

from rag_app.core.settings import get_settings
from rag_app.models import (
    LLMManagerRequest,
    PromptRequest,
    QueryPerformanceTracker,
    RAGRequest,
    RAGResposne,
    RetriveRequest,
)
from rag_app.observability.events import EventName
from rag_app.observability.logger import logger
from rag_app.prompts.prompt_manager import PromptManager
from rag_app.query.query_manager import QueryManager
from rag_app.retrieval.retriver_manager import RetriverManager
from rag_app.services.llm_services import LLMServicemanager
from rag_app.tracker.query_performance_tracker import QueryPerformanceTrackerLogger


class RAGChat:
    def __init__(self):
        self.settings = get_settings()
        self.query_manager = QueryManager()
        self.retriver_manager = RetriverManager()
        self.prompt_manager = PromptManager()
        self.llm_manager = LLMServicemanager()
        self.query_performance_tracker = QueryPerformanceTrackerLogger()

    async def get_chat_answer(
        self,
        request: RAGRequest,
    ) -> RAGResposne:
        # =====================================================
        # Retrieval
        # =====================================================

        retrieval_start = time.perf_counter()

        context = await self.retriver_manager.retrieve(
            request=RetriveRequest(
                tenant_id=request.tenant_id,
                query=request.query,
            )
        )

        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        no_of_chunks = len(context.results)
        logger.info(
            "Retrieval completed",
            event=EventName.RETRIEVAL_COMPLETED,
            component="rag_feature",
            latency_ms=retrieval_latency_ms,
            no_of_chunks=no_of_chunks,
        )

        # =====================================================
        # Prompt
        # =====================================================

        prompt_start = time.perf_counter()

        prompt, prompt_version = self.prompt_manager.build_rag_prompt(
            request=PromptRequest(
                query=request.query,
                chunks=context.results,
            )
        )

        prompt_latency_ms = (time.perf_counter() - prompt_start) * 1000

        logger.info(
            "Prompt building completed",
            event=EventName.PROMPT_COMPLETED,
            component="rag_feature",
            latency_ms=prompt_latency_ms,
            prompt_version=prompt_version,
        )

        # =====================================================
        # LLM
        # =====================================================

        llm_start = time.perf_counter()

        answer = await self.llm_manager.complete(request=LLMManagerRequest(prompt=prompt))

        llm_latency_ms = (time.perf_counter() - llm_start) * 1000

        logger.info(
            "LLM request completed",
            event=EventName.LLM_COMPLETED,
            component="rag_feature",
            latency_ms=llm_latency_ms,
        )

        tracker = QueryPerformanceTracker(
            app_version=self.settings.app_version,
            query=request.query,
            chunk_ids=[chunk.chunk_id for chunk in context.results],
            llm_answer=answer.text,
        )

        self.query_performance_tracker.track(tracker)

        # =====================================================
        # Response
        # =====================================================

        return RAGResposne(text=answer.text)
