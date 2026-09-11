from __future__ import annotations

import inspect
import operator
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from rag_app.core.settings import get_settings
from rag_app.models import (
    QueryManagerRequest,
    QueryPerformanceTracker,
    RAGRequest,
    RAGResposne,
    RetriveRequest,
)
from rag_app.prompts.prompt_manager import PromptManager
from rag_app.query.query_manager import QueryManager
from rag_app.retrieval.retriver_manager import RetriverManager
from rag_app.services.llm_services import LLMServicemanager
from rag_app.tracker.query_performance_tracker import QueryPerformanceTrackerLogger


class RAGGraphState(TypedDict, total=False):
    query: str
    current_query: str
    tenant_id: str
    retrieved_documents: Annotated[list[dict[str, Any]], operator.add]
    retrieval_attempts: int
    relevance_sufficient: bool
    category: str
    requires_review: bool
    review_approved: bool
    answer: str
    sources: list[str] | str | None
    stage: str


class RAGChatLanggraph:
    """Retryable, checkpointed RAG workflow."""

    def __init__(
        self,
        retriever_manager: RetriverManager | None = None,
        llm: Runnable | None = None,
        checkpointer: Any | None = None,
        query_transformer: Callable[[str], Any] | None = None,
        review_categories: set[str] | None = None,
    ):
        self.settings = get_settings()
        self.retriever_manager = retriever_manager or RetriverManager()
        self.query_manager = QueryManager()
        self.prompt_manager = PromptManager()
        self.llm_manager = LLMServicemanager()
        self.query_performance_tracker = QueryPerformanceTrackerLogger()
        self.llm = llm or self.llm_manager.get_chat_model()
        self.query_transformer = query_transformer or self._transform_query
        self.review_categories = review_categories or {"requires_review"}
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph()

    @classmethod
    async def from_postgres(
        cls, connection_url: str | None = None, **kwargs: Any
    ) -> RAGChatLanggraph:
        """Build a long-lived graph whose checkpoints survive process restarts."""
        settings = get_settings()
        context = AsyncPostgresSaver.from_conn_string(
            connection_url or settings.DATABASE_CONNECTION_CONVERSATION_URL
        )
        checkpointer = await context.__aenter__()
        await checkpointer.setup()
        instance = cls(checkpointer=checkpointer, **kwargs)
        instance._checkpointer_context = context
        return instance

    def _build_graph(self):
        graph = StateGraph(RAGGraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("reformulate", self._reformulate)
        graph.add_node("review", self._review)
        graph.add_node("generate", self._generate)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"reformulate": "reformulate", "review": "review", "generate": "generate"},
        )
        graph.add_edge("reformulate", "retrieve")
        graph.add_conditional_edges(
            "review",
            lambda state: "generate" if state.get("review_approved", False) else "end",
            {"generate": "generate", "end": END},
        )
        graph.add_edge("generate", END)
        return graph.compile(checkpointer=self.checkpointer)

    async def _retrieve(self, state: RAGGraphState) -> dict[str, Any]:
        response = await self.retriever_manager.retrieve(
            RetriveRequest(tenant_id=state["tenant_id"], query=state["current_query"])
        )
        documents = [
            {
                "page_content": result.chunk_text,
                "document_name": result.document_name,
                "chunk_id": str(result.chunk_id),
                "similarity_score": result.similarity_score,
            }
            for result in response.results
        ]
        return {
            "retrieved_documents": documents,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
            "stage": "retrieve",
        }

    async def _grade(self, state: RAGGraphState) -> dict[str, Any]:
        documents = state.get("retrieved_documents", [])
        sufficient = (
            bool(documents)
            and max(float(document.get("similarity_score", 0)) for document in documents) >= 0.5
        )
        query = state["query"].lower()
        category = (
            "requires_review"
            if any(term in query for term in ("refund", "delete my account", "legal advice"))
            else "standard"
        )
        return {
            "relevance_sufficient": sufficient,
            "category": category,
            "requires_review": category in self.review_categories,
            "stage": "grade",
        }

    def _route_after_grade(self, state: RAGGraphState) -> str:
        if not state.get("relevance_sufficient", False) and state.get("retrieval_attempts", 0) < 2:
            return "reformulate"
        if state.get("requires_review", False):
            return "review"
        return "generate"

    async def _reformulate(self, state: RAGGraphState) -> dict[str, Any]:
        query = self.query_transformer(state["current_query"])
        if inspect.isawaitable(query):
            query = await query
        return {"current_query": query, "stage": "reformulate"}

    async def _review(self, state: RAGGraphState) -> dict[str, Any]:
        decision = interrupt(
            {
                "type": "requires_review",
                "category": state.get("category"),
                "query": state["query"],
                "message": "Approve generation for this requires-review query?",
            }
        )
        approved = decision in (True, "approve", "approved")
        return {
            "review_approved": approved,
            "answer": "Generation was not approved by a human reviewer." if not approved else "",
            "stage": "review",
        }

    async def _generate(self, state: RAGGraphState) -> dict[str, Any]:
        parser = PydanticOutputParser(pydantic_object=RAGResposne)
        prompt = self.prompt_manager.build_rag_prompt_langchain(
            format_instructions=parser.get_format_instructions()
        )
        documents = [
            Document(page_content=document["page_content"], metadata=document)
            for document in state.get("retrieved_documents", [])
        ]
        context = "\n\n".join(
            f"Document Name(Source Name): {document.metadata.get('document_name')}\n"
            f"Content :- {document.page_content}"
            for document in documents
        )
        prompt_value = prompt.invoke({"query": state["query"], "context": context})
        response = await self.llm.ainvoke(prompt_value)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        parsed = parser.parse(content)
        sources = parsed.source
        if sources is None:
            sources = list(
                dict.fromkeys(
                    document["document_name"]
                    for document in state.get("retrieved_documents", [])
                    if document.get("document_name")
                )
            )
        return {
            "answer": parsed.text,
            "sources": sources,
            "stage": "generate",
        }

    async def _transform_query(self, query: str) -> str:
        response = await self.query_manager.get_queries(
            QueryManagerRequest(query=query, technique="query_HyDE")
        )
        return response.queries[1] if len(response.queries) >= 1 else query

    async def get_chat_answer(
        self,
        request: RAGRequest,
    ) -> RAGResposne:
        result = await self.graph.ainvoke(
            {
                "query": request.query,
                "current_query": request.query,
                "tenant_id": str(request.tenant_id),
                "retrieved_documents": [],
                "retrieval_attempts": 0,
            },
            config={
                "metadata": {"tenant_id": request.tenant_id},
                "configurable": {"thread_id": f"{request.tenant_id}:{request.session_id}"},
            },
        )

        tracker = QueryPerformanceTracker(
            app_version=self.settings.app_version,
            query=request.query,
            no_of_queries=None,
            chunk_ids=None,
            llm_answer=result.get("answer", ""),
        )

        self.query_performance_tracker.track(tracker)

        return RAGResposne(text=result.get("answer", ""), source=result.get("sources", []))

    async def stream_chat_answer(self, request: RAGRequest) -> AsyncIterator[dict[str, Any]]:
        initial_state = {
            "query": request.query,
            "current_query": request.query,
            "tenant_id": str(request.tenant_id),
            "retrieved_documents": [],
            "retrieval_attempts": 0,
        }
        config = {
            "metadata": {"tenant_id": request.tenant_id},
            "configurable": {"thread_id": f"{request.tenant_id}:{request.session_id}"},
        }
        async for update in self.graph.astream(initial_state, config=config, stream_mode="updates"):
            if "generate" in update:
                yield {"generate": update["generate"]}

    async def resume_chat_answer(self, request: RAGRequest, decision: Any) -> RAGResposne:
        """Resume a paused human-review thread using its durable checkpoint."""
        result = await self.graph.ainvoke(
            Command(resume=decision),
            config={
                "metadata": {"tenant_id": request.tenant_id},
                "configurable": {"thread_id": f"{request.tenant_id}:{request.session_id}"},
            },
        )
        return RAGResposne(text=result.get("answer", ""), source=result.get("sources", []))
