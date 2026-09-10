from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import (
    RunnableWithMessageHistory,
)
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.documents import Document


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


class RAGState(TypedDict, total=False):
    tenant_id: dict
    session_id: dict
    query: str
    retrived_docs = Annotated[list[Document], operator.add]
    retrive_attemps = int
    relevance_sufficient: bool
    prompt: str
    answer: str
    requires_review: bool


class RAGChatLanggraph:
    def __init__(self):
        self.settings = get_settings()
        self.langchain_retriver = LangchainRetriever()
        self.prompt_manager = PromptManager()
        self.llm_manager = LLMServicemanager()
        self.query_performance_tracker = QueryPerformanceTrackerLogger()
        self.graph = self._build_graph()
        self.parser = PydanticOutputParser(pydantic_object=RAGResposne)

    def _build_graph(self):
        # -----------------------------------------
        # Retrieval Node
        # -----------------------------------------
        async def retrive_docs(state: RAGState) -> dict:
            print(f"[RETRIEVE] Attempt {state.get('retrieval_attempts', 0) + 1}")

            docs = await self.langchain_retriver.ainvoke(
                {"query": state["query"]},
                config={
                    "metadata": {
                        "tenant_id": state["tenant_id"],
                    },
                    "configurable": {
                        "session_id": state["session_id"],
                    },
                },
            )

            return {"retrived_docs": docs, "retrive_attemps": state["retrive_attemps"] + 1}

        # -----------------------------------------
        # Prepare prompt
        # -----------------------------------------
        def build_prompt(state: RAGState) -> dict:
            prompt = self.prompt_manager.build_rag_prompt_langchain(
                format_instructions=self.parser.get_format_instructions()
            )

            prompt_text = prompt.invoke(
                {"query": state["query"], "context": self._build_context(state["retrived_docs"])}
            )

            return {"prompt": prompt_text}

        # -----------------------------------------
        # Genrate Answer
        # -----------------------------------------
        async def genrate_answer(state: RAGState):
            llm = self.llm_manager.get_chat_model()

            llm_model = RunnableWithMessageHistory(
                llm,
                get_session_history=get_redis_message_history,
                input_messages_key="query",
                history_messages_key="history",
            )

            answer = await llm_model.ainvoke({"prompt": state["prompt"]})

            return {"answer": answer}

        graph = StateGraph(RAGState)

        graph.add_node("retrive_docs", retrive_docs)
        graph.add_node("build_prompt", build_prompt)
        graph.add_node("genrate_answer", genrate_answer)

        graph.set_entry_point("retrive_docs")
        graph.add_edge("retrive_docs", "build_prompt")
        graph.add_edge("build_prompt", "genrate_answer")
        graph.add_edge("genrate_answer", END)

        self.graph = graph.compile()

    def _build_context(documents):
        return "\n\n".join(
            f"Document Name(Source Name): {doc.metadata.get('document_name')}\nContent :- {doc.page_content}\n\n"
            if doc.metadata.get("document_name")
            else doc.page_content
            for doc in documents
        )

    async def get_chat_answer(
        self,
        request: RAGRequest,
    ) -> RAGResposne:
        initial_state: RAGState = {
            "tenant_id": request.tenant_id,
            "session_id": request.session_id,
            "query": request.query,
            "retrived_docs": [],
            "retrive_attemps": 0,
            "prompt": "",
            "answer": "",
        }
        answer = await self.graph.ainvoke(initial_state, version="v2")

        tracker = QueryPerformanceTracker(
            app_version=self.settings.app_version,
            query=request.query,
            no_of_queries=None,
            chunk_ids=None,
            llm_answer=answer.text,
        )

        self.query_performance_tracker.track(tracker)

        return RAGResposne(text=answer.text, source=answer.source)
