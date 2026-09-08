from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from rag_app.features.rag_chat_langchain import RAGChatLangchain
from rag_app.models import RAGRequest, RAGResposne


class StubRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        return self.documents


def test_build_chain_injects_schema_and_document_name_context(monkeypatch):
    document = Document(
        page_content="Refunds are available within 30 days.",
        metadata={"document_name": "refund-policy.pdf"},
    )
    prompt_manager = MagicMock()
    prompt_manager.build_rag_prompt_langchain.side_effect = lambda format_instructions: (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Context:\n{context}\nSchema:\n{format_instructions}",
                ),
                ("human", "{query}"),
            ]
        ).partial(format_instructions=format_instructions)
    )

    llm = RunnableLambda(lambda _: '{"text":"grounded answer","source":"refund-policy.pdf"}')
    rag_chat = RAGChatLangchain.__new__(RAGChatLangchain)
    rag_chat.langchain_retriver = StubRetriever(documents=[document])
    rag_chat.prompt_manager = prompt_manager
    rag_chat.llm_manager = MagicMock()
    rag_chat.llm_manager.get_chat_model.return_value = llm
    history = MagicMock()
    history.messages = []
    monkeypatch.setattr(
        "rag_app.features.rag_chat_langchain.get_redis_message_history",
        lambda session_id: history,
    )

    chain = rag_chat._build_chain()
    result = chain.invoke(
        {"query": "What is the refund policy?"},
        config={"configurable": {"session_id": "tenant:session"}},
    )

    format_instructions = prompt_manager.build_rag_prompt_langchain.call_args.kwargs[
        "format_instructions"
    ]
    assert '"text"' in format_instructions
    assert '"source"' in format_instructions
    assert result == RAGResposne(
        text="grounded answer",
        source="refund-policy.pdf",
    )


@pytest.mark.asyncio
async def test_get_chat_answer_tracks_and_returns_parsed_response():
    tenant_id = uuid4()
    request = RAGRequest(query="What is the refund policy?", tenant_id=tenant_id)
    answer = RAGResposne(text="Refunds are available within 30 days.", source="refund-policy.pdf")

    rag_chat = RAGChatLangchain.__new__(RAGChatLangchain)
    rag_chat.chain = MagicMock()
    rag_chat.chain.ainvoke = AsyncMock(return_value=answer)
    rag_chat.settings = MagicMock(app_version="test-version")
    rag_chat.query_performance_tracker = MagicMock()

    result = await rag_chat.get_chat_answer(request)

    assert result == answer
    rag_chat.chain.ainvoke.assert_awaited_once_with(
        {"query": request.query},
        config={
            "metadata": {"tenant_id": tenant_id},
            "configurable": {"session_id": f"{tenant_id}:{request.session_id}"},
        },
    )
    rag_chat.query_performance_tracker.track.assert_called_once()
    tracker = rag_chat.query_performance_tracker.track.call_args.args[0]
    assert tracker.query == request.query
    assert tracker.llm_answer == answer.text
