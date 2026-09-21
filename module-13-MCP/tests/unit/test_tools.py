import pytest
from mcp.server.mcpserver import MCPServer

from internal_tools_mcp.mcp.tools.tools import _tenant_uuid, _text, register_tools


@pytest.mark.asyncio
async def test_register_tools_exposes_both_tools():
    async def answer(query, tenant_id):
        return {"answer": query, "tenant_id": tenant_id}

    async def retrieval(tenant_id, query, document_type, top_k):
        return {
            "tenant_id": tenant_id,
            "query": query,
            "document_type": document_type,
            "top_k": top_k,
        }

    server = MCPServer("test-tools")
    register_tools(server, answer_service=answer, retrieval_service=retrieval)

    tools = await server.list_tools()
    assert [tool.name for tool in tools] == [
        "module9_chat_answer",
        "module8_retrieve_documents",
    ]

    answer_result = await server.call_tool(
        "module9_chat_answer",
        {"query": "hello", "tenant_id": "00000000-0000-4000-8000-000000000001"},
    )
    retrieval_result = await server.call_tool(
        "module8_retrieve_documents",
        {"tenant_id": "tenant-1", "query": "search", "top_k": 3},
    )

    assert answer_result.structured_content == {
        "answer": "hello",
        "tenant_id": "00000000-0000-4000-8000-000000000001",
    }
    assert retrieval_result.structured_content == {
        "tenant_id": "tenant-1",
        "query": "search",
        "document_type": None,
        "top_k": 3,
    }


def test_tool_validation_accepts_and_normalizes_text():
    assert _text("  query  ", "query", 10) == "query"
    assert _tenant_uuid("00000000-0000-4000-8000-000000000001")


@pytest.mark.parametrize(
    "value, name, maximum",
    [("", "query", 10), ("   ", "query", 10), ("too long", "query", 3)],
)
def test_text_validation_rejects_invalid_values(value, name, maximum):
    with pytest.raises(ValueError):
        _text(value, name, maximum)


def test_tenant_validation_rejects_non_uuid():
    with pytest.raises(ValueError, match="valid UUID"):
        _tenant_uuid("tenant-1")


@pytest.mark.asyncio
async def test_tool_validation_rejects_invalid_top_k():
    server = MCPServer("test-validation")
    register_tools(
        server, answer_service=lambda *_: None, retrieval_service=lambda *_: None
    )

    with pytest.raises(
        Exception, match="Error executing tool module8_retrieve_documents"
    ):
        await server.call_tool(
            "module8_retrieve_documents",
            {"tenant_id": "tenant-1", "query": "search", "top_k": 0},
        )
