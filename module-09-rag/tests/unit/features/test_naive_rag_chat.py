import pytest
from unittest.mock import AsyncMock, MagicMock

from uuid import uuid4

from rag_app.features.rag_chat import RAGChat
from rag_app.models import (
    LLMManagerRequest,
    PromptRequest,
    QueryManagerRequest,
    RAGRequest,
    RAGResposne,
    RetriveRequest,
    RetriveResult,
)


@pytest.mark.asyncio
async def test_get_chat_answer():
    # Arrange
    rag_chat = RAGChat()

    rag_chat.query_manager = MagicMock()
    rag_chat.query_manager.get_queries = AsyncMock()

    rag_chat.retriver_manager = MagicMock()
    rag_chat.retriver_manager.retrieve = AsyncMock()

    rag_chat.prompt_manager = MagicMock()

    rag_chat.llm_manager = MagicMock()
    rag_chat.llm_manager.complete = AsyncMock()

    request = RAGRequest(query="What is RAG?", tenant_id=uuid4(), request_id=uuid4())

    queries = MagicMock()
    queries.queries = ["What is retrieval augmented generation?"]

    context = MagicMock()
    context.results = [
        RetriveResult(
            chunk_text="RAG combines retrieval with generation.",
            similarity_score=0.95,
        )
    ]

    prompt = "Answer the question using the provided context."

    answer = MagicMock()
    answer.text = "RAG stands for Retrieval-Augmented Generation."

    rag_chat.query_manager.get_queries.return_value = queries
    rag_chat.retriver_manager.retrieve.return_value = context
    rag_chat.prompt_manager.build_rag_prompt.return_value = prompt
    rag_chat.llm_manager.complete.return_value = answer

    # Act
    result = await rag_chat.get_chat_answer(request)

    # Assert
    assert isinstance(result, RAGResposne)
    assert result.text == "RAG stands for Retrieval-Augmented Generation."

    rag_chat.query_manager.get_queries.assert_awaited_once_with(
        request=QueryManagerRequest(
            query="What is RAG?",
        )
    )

    rag_chat.retriver_manager.retrieve.assert_awaited_once_with(
        request=RetriveRequest(
            tenant_id=request.tenant_id,
            queries=["What is retrieval augmented generation?"],
        )
    )

    rag_chat.prompt_manager.build_rag_prompt.assert_called_once_with(
        request=PromptRequest(
            query="What is RAG?",
            chunks=[
                RetriveResult(
                    chunk_text="RAG combines retrieval with generation.",
                    similarity_score=0.95,
                )
            ],
        )
    )

    rag_chat.llm_manager.complete.assert_awaited_once_with(
        request=LLMManagerRequest(
            prompt=prompt,
        )
    )
