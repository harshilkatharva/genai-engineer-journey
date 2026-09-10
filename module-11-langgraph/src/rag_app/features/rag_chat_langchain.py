from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableWithMessageHistory,
)

from rag_app.core.settings import get_settings
from rag_app.models import (
    QueryPerformanceTracker,
    RAGRequest,
    RAGResposne,
)
from rag_app.prompts.prompt_manager import PromptManager
from rag_app.retrieval.retriver_manager import LangchainRetriever
from rag_app.services.llm_services import LLMServicemanager
from rag_app.services.redis_history import get_redis_message_history
from rag_app.tracker.query_performance_tracker import QueryPerformanceTrackerLogger


class RAGChatLangchain:
    def __init__(self):
        self.settings = get_settings()
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
            return "\n\n".join(
                f"Document Name(Source Name): {doc.metadata.get('document_name')}\nContent :- {doc.page_content}\n\n"
                if doc.metadata.get("document_name")
                else doc.page_content
                for doc in documents
            )

        # -----------------------------------------
        # Prepare prompt input
        # -----------------------------------------

        prompt_input = RunnableParallel(
            query=lambda x: x["query"],
            context=RunnableLambda(lambda x: x["query"]) | retrieval | context,
        )

        # -----------------------------------------
        # Prompt → LLM → Parser
        # -----------------------------------------

        parser = PydanticOutputParser(pydantic_object=RAGResposne)

        prompt = self.prompt_manager.build_rag_prompt_langchain(
            format_instructions=parser.get_format_instructions()
        )

        llm = self.llm_manager.get_chat_model()

        history_chain = RunnableWithMessageHistory(
            prompt_input | prompt | llm,
            get_session_history=get_redis_message_history,
            input_messages_key="query",
            history_messages_key="history",
        )

        return history_chain | parser

    async def get_chat_answer(
        self,
        request: RAGRequest,
    ) -> RAGResposne:
        answer = await self.chain.ainvoke(
            {"query": request.query},
            config={
                "metadata": {
                    "tenant_id": request.tenant_id,
                },
                "configurable": {
                    "session_id": f"{request.tenant_id}:{request.session_id}",
                },
            },
        )

        tracker = QueryPerformanceTracker(
            app_version=self.settings.app_version,
            query=request.query,
            no_of_queries=None,
            chunk_ids=None,
            llm_answer=answer.text,
        )

        self.query_performance_tracker.track(tracker)

        return RAGResposne(text=answer.text, source=answer.source)
