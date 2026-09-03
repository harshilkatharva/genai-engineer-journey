import time

from rag_app.core.settings import get_settings
from rag_app.models import (
    LLMManagerRequest,
    QueryManagerRequest,
    QueryPerformanceTracker,
    RAGRequest,
    RAGResposne,
)
from rag_app.observability.events import EventName
from rag_app.observability.logger import logger
from rag_app.prompts.prompt_manager import PromptManager
from rag_app.query.query_manager import QueryManager
from rag_app.retrieval.retriver_manager import LangchainRetriever
from rag_app.services.llm_services import LLMServicemanager
from rag_app.tracker.query_performance_tracker import QueryPerformanceTrackerLogger


class RAGChat:
    def __init__(self):
        self.settings = get_settings()
        self.query_manager = QueryManager()
        self.langchain_retriver = LangchainRetriever()
        self.prompt_manager = PromptManager()
        self.llm_manager = LLMServicemanager()
        self.query_performance_tracker = QueryPerformanceTrackerLogger()

    async def get_chat_answer(
        self,
        request: RAGRequest,
    ) -> RAGResposne:
        # =====================================================
        # Query processing
        # =====================================================

        query_start = time.perf_counter()

        queries = await self.query_manager.get_queries(
            request=QueryManagerRequest(query=request.query)
        )

        query_latency_ms = (time.perf_counter() - query_start) * 1000

        logger.info(
            "Query processing completed",
            event=EventName.QUERY_COMPLETED,
            component="rag_feature",
            latency_ms=query_latency_ms,
            no_of_queries=len(queries.queries),
        )

        # =====================================================
        # Retrieval
        # =====================================================

        retrieval_start = time.perf_counter()

        documents = await self.langchain_retriver.ainvoke(request.tenant_id, queries.queries)

        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        no_of_chunks = len(documents)
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

        prompt = self.prompt_manager.build_rag_prompt_langchain()

        prompt_latency_ms = (time.perf_counter() - prompt_start) * 1000

        logger.info(
            "Prompt building completed",
            event=EventName.PROMPT_COMPLETED,
            component="rag_feature",
            latency_ms=prompt_latency_ms,
            prompt_version=self.settings.rag_prompt_running_version,
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
            no_of_queries=len(queries.queries),
            chunk_ids=[chunk.metadata["chunk_id"] for chunk in documents],
            llm_answer=answer.text,
        )

        self.query_performance_tracker.track(tracker)

        # =====================================================
        # Response
        # =====================================================

        return RAGResposne(text=answer.text)
