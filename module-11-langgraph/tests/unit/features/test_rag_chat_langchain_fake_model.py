from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from rag_app.features.rag_chat_langchain import RAGChatLangchain
from rag_app.models import RAGResposne
from rag_app.prompts.prompt_manager import PromptManager


class StubRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        return self.documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        return self.documents


def build_chat_chain(monkeypatch, model):
    chat = RAGChatLangchain.__new__(RAGChatLangchain)
    chat.langchain_retriver = StubRetriever(
        documents=[
            Document(
                page_content="Refunds are available within 30 days.",
                metadata={"document_name": "refund-policy.pdf"},
            )
        ]
    )
    chat.prompt_manager = PromptManager()
    chat.llm_manager = MagicMock()
    chat.llm_manager.get_chat_model.return_value = model

    history = MagicMock()
    history.messages = []
    history.aget_messages = AsyncMock(return_value=[])
    history.aadd_messages = AsyncMock()
    monkeypatch.setattr(
        "rag_app.features.rag_chat_langchain.get_redis_message_history",
        lambda session_id: history,
    )

    return chat._build_chain()


@pytest.mark.asyncio
async def test_fake_model_returns_parsed_response(monkeypatch):
    model = FakeListChatModel(
        responses=['{"text":"Refunds are available within 30 days.","source":"refund-policy.pdf"}']
    )
    chain = build_chat_chain(monkeypatch, model)

    result = await chain.ainvoke(
        {"query": "What is the refund policy?"},
        config={"configurable": {"session_id": "fake-session"}},
    )

    assert result == RAGResposne(
        text="Refunds are available within 30 days.",
        source="refund-policy.pdf",
    )


@pytest.mark.asyncio
async def test_fake_model_is_deterministic_for_repeated_inputs(monkeypatch):
    model = FakeListChatModel(
        responses=[
            '{"text":"First deterministic answer","source":null}',
            '{"text":"Second deterministic answer","source":null}',
        ]
    )
    chain = build_chat_chain(monkeypatch, model)
    config = {"configurable": {"session_id": "fake-session"}}

    first = await chain.ainvoke({"query": "Question"}, config=config)
    second = await chain.ainvoke({"query": "Question"}, config=config)

    assert first.text == "First deterministic answer"
    assert second.text == "Second deterministic answer"
