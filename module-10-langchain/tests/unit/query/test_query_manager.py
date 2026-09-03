import pytest

from rag_app.models import QueryManagerRequest
from rag_app.query.query_manager import QueryManager


@pytest.mark.asyncio
async def test_get_queries_with_default_None(monkeypatch):
    class MockSettings:
        default_query_strategy = None

    monkeypatch.setattr(
        "rag_app.query.query_manager.get_settings",
        lambda: MockSettings(),
    )

    manager = QueryManager()

    request = QueryManagerRequest(
        query="What is RAG?",
    )

    response = await manager.get_queries(request)

    assert response.queries == ["What is RAG?"]
