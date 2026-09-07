from rag_app.core.settings import get_settings
from rag_app.models import (
    QueryPerformanceTracker,
    RAGRequest,
    RAGResposne,
)

from rag_app.prompts.prompt_manager import PromptManager
from rag_app.query.query_manager import QueryManager
from rag_app.retrieval.retriver_manager import LangchainRetriever
from rag_app.services.llm_services import LLMServicemanager
from rag_app.tracker.query_performance_tracker import QueryPerformanceTrackerLogger

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


class RAGChatLangchain:
    def __init__(self):
        self.settings = get_settings()
        self.query_manager = QueryManager()
        self.langchain_retriver = LangchainRetriever()
        self.prompt_manager = PromptManager()
        self.llm_manager = LLMServicemanager()
        self.query_performance_tracker = QueryPerformanceTrackerLogger()
        self.chain = self._build_chain()

    def _build_chain(self):
        # -----------------------------------------
        # Retrieval
        # -----------------------------------------

        retrieval = self.langchain_retriver

        # -----------------------------------------
        # Documents → context
        # -----------------------------------------

        def context(documents):
            return "\n\n".join(doc.page_content for doc in documents)

        # -----------------------------------------
        # Prepare prompt input
        # -----------------------------------------

        prompt_input = RunnableParallel(
            query=lambda x: x,
            context=retrieval | context,
        )

        # -----------------------------------------
        # Prompt → LLM → Parser
        # -----------------------------------------

        prompt = self.prompt_manager.build_rag_prompt_langchain()

        llm = self.llm_manager.get_chat_model()

        parser = StrOutputParser()

        return prompt_input | prompt | llm | parser

    async def get_chat_answer(
        self,
        request: RAGRequest,
    ) -> RAGResposne:
        answer = await self.chain.ainvoke(
            request.query,
            config={
                "metadata": {
                    "tenant_id": request.tenant_id,
                }
            },
        )

        print(answer)

        tracker = QueryPerformanceTracker(
            app_version=self.settings.app_version,
            query=request.query,
            no_of_queries=None,
            chunk_ids=None,
            llm_answer=answer,
        )

        self.query_performance_tracker.track(tracker)

        return RAGResposne(text=answer)
