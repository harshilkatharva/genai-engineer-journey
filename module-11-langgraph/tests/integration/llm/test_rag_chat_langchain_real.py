import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_app.core.config import GOOGLE_API_KEY
from rag_app.core.settings import get_settings
from rag_app.features.rag_chat_langchain import RAGChatLangchain
from rag_app.prompts.prompt_manager import PromptManager
from rag_app.models import RAGResposne


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_TESTS") != "1" or not GOOGLE_API_KEY,
    reason="Set RUN_REAL_LLM_TESTS=1 and GOOGLE_API_KEY to run provider integration tests.",
)


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


def build_real_chat_chain(monkeypatch):
    settings = get_settings()
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
    chat.llm_manager.get_chat_model.return_value = ChatGoogleGenerativeAI(
        model=settings.default_llm_model,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

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
async def test_real_google_model_returns_valid_rag_response(monkeypatch):
    chain = build_real_chat_chain(monkeypatch)

    result = await chain.ainvoke(
        {"query": "What is the refund policy?"},
        config={"configurable": {"session_id": "real-provider-test"}},
    )

    assert isinstance(result, RAGResposne)
    assert result.text
    assert isinstance(result.text, str)
    assert result.source is None or isinstance(result.source, (str, list))


@pytest.mark.asyncio
async def test_real_google_model_rejects_question_unrelated_to_context(monkeypatch):
    chain = build_real_chat_chain(monkeypatch)

    result = await chain.ainvoke(
        {"query": "What is the capital of France?"},
        config={"configurable": {"session_id": "real-provider-unrelated-test"}},
    )

    response_text = result.text.lower()
    insufficiency_signals = (
        "not contain",
        "does not contain",
        "insufficient",
        "not enough",
        "cannot answer",
        "unable to answer",
    )

    assert any(signal in response_text for signal in insufficiency_signals)
    assert result.source is None
