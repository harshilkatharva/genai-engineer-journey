import pytest
from uuid import UUID

from internal_tools_mcp.rag_service.rag_service import ask_question
from internal_tools_mcp.retrive_service.retrive_service import retrieve_documents


class FakeRequest:
    def __init__(self, **values):
        self.values = values


class FakeResponse:
    def model_dump(self):
        return {"ok": True, "result": "sample"}


class FakeChat:
    last_request = None

    async def get_chat_answer(self, request):
        self.last_request = request
        return FakeResponse()


class FakeRetriever:
    last_request = None

    async def retrive_chunks(self, request):
        self.last_request = request
        return FakeResponse()


@pytest.mark.asyncio
async def test_rag_adapter_maps_request_and_response():
    service = FakeChat()

    result = await ask_question(
        "What is MCP?",
        "00000000-0000-4000-8000-000000000001",
        services_factory=lambda: (lambda: service, FakeRequest),
    )

    assert result == {"ok": True, "result": "sample"}
    assert service.last_request.values == {
        "query": "What is MCP?",
        "tenant_id": UUID("00000000-0000-4000-8000-000000000001"),
    }


@pytest.mark.asyncio
async def test_retrieval_adapter_maps_request_and_redirects_stdout(capsys):
    service = FakeRetriever()

    result = await retrieve_documents(
        "tenant-1",
        "vector search",
        "policy",
        7,
        services_factory=lambda: (lambda: service, FakeRequest),
    )

    assert result == {"ok": True, "result": "sample"}
    assert service.last_request.values == {
        "tenant_id": "tenant-1",
        "query": "vector search",
        "document_type": "policy",
        "top_k": 7,
    }
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_rag_adapter_restores_working_directory():
    from pathlib import Path

    original = Path.cwd()
    await ask_question(
        "question",
        "00000000-0000-4000-8000-000000000001",
        services_factory=lambda: (FakeChat, FakeRequest),
    )
    assert Path.cwd() == original
