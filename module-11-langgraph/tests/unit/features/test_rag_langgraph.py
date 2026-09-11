from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from rag_app.features.rag_langgraph import RAGChatLanggraph
from rag_app.models import RAGRequest, RetriveResponse, RetriveResult


class FakeRetriever:
    def __init__(self):
        self.queries = []

    async def retrieve(self, request):
        self.queries.append(request.query)
        score = 0.4 if len(self.queries) == 1 else 0.8
        return RetriveResponse(
            tenant_id=request.tenant_id,
            queries=[request.query],
            results=[
                RetriveResult(
                    chunk_id=str(len(self.queries)),
                    chunk_text=f"Evidence for {request.query}",
                    document_name=f"attempt-{len(self.queries)}.md",
                    similarity_score=score,
                )
            ],
        )


def test_unique_sources_are_used_when_llm_source_is_none():
    state = {
        "retrieved_documents": [
            {"document_name": "policy.pdf"},
            {"document_name": "policy.pdf"},
            {"document_name": "guide.pdf"},
            {"document_name": None},
        ]
    }

    sources = list(
        dict.fromkeys(
            document["document_name"]
            for document in state["retrieved_documents"]
            if document.get("document_name")
        )
    )

    assert sources == ["policy.pdf", "guide.pdf"]


@pytest.fixture
def graph(monkeypatch):
    retriever = FakeRetriever()
    llm = RunnableLambda(
        lambda _: AIMessage(content='{"text":"grounded answer", "source": ["answer.pdf"]}')
    )
    service = RAGChatLanggraph(
        retriever_manager=retriever,
        llm=llm,
        checkpointer=InMemorySaver(),
        query_transformer=lambda _: "reformulated query",
    )
    service.prompt_manager.build_rag_prompt_langchain = lambda format_instructions: (
        ChatPromptTemplate.from_messages([("system", "Context: {context}"), ("human", "{query}")])
    )
    return service, retriever


@pytest.mark.asyncio
async def test_retry_accumulates_documents_and_overwrites_attempt_state(graph):
    service, retriever = graph
    tenant_id = uuid4()

    result = await service.get_chat_answer(
        RAGRequest(query="Which policy applies?", tenant_id=tenant_id)
    )
    state = await service.graph.aget_state({"configurable": {"thread_id": f"{tenant_id}:default"}})

    assert result.text == "grounded answer"
    assert result.source == ["answer.pdf"]
    assert retriever.queries == ["Which policy applies?", "reformulated query"]
    assert state.values["retrieval_attempts"] == 2
    assert state.values["relevance_sufficient"] is True
    assert len(state.values["retrieved_documents"]) == 2


@pytest.mark.asyncio
async def test_stream_returns_only_generate_update(graph):
    service, _ = graph
    updates = [
        update
        async for update in service.stream_chat_answer(
            RAGRequest(query="Which policy applies?", tenant_id=uuid4())
        )
    ]

    assert len(updates) == 1
    assert set(updates[0]) == {"generate"}
    assert updates[0]["generate"]["answer"] == "grounded answer"


@pytest.mark.asyncio
async def test_requires_review_interrupts_before_generation(monkeypatch):
    retriever = FakeRetriever()
    llm_called = False

    def fail_if_called(_):
        nonlocal llm_called
        llm_called = True
        raise AssertionError("generation must wait for human review")

    service = RAGChatLanggraph(
        retriever_manager=retriever,
        llm=RunnableLambda(fail_if_called),
        checkpointer=InMemorySaver(),
        query_transformer=lambda query: query,
    )
    service.prompt_manager.build_rag_prompt_langchain = lambda format_instructions: (
        ChatPromptTemplate.from_messages([("human", "{query}")])
    )
    tenant_id = uuid4()
    config = {"configurable": {"thread_id": f"{tenant_id}:review"}}

    result = await service.graph.ainvoke(
        {
            "query": "Can I get a refund?",
            "current_query": "Can I get a refund?",
            "tenant_id": str(tenant_id),
            "retrieved_documents": [],
            "retrieval_attempts": 0,
        },
        config=config,
    )

    assert "__interrupt__" in result
    assert llm_called is False
