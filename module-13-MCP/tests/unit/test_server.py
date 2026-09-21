import asyncio

from starlette.testclient import TestClient

from internal_tools_mcp import server as server_module

mcp = server_module.mcp


def test_server_registers_tools_and_resources():
    tools = asyncio.run(mcp.list_tools())
    resources = asyncio.run(mcp.list_resources())

    assert {tool.name for tool in tools} == {
        "module9_chat_answer",
        "module8_retrieve_documents",
    }
    assert len(resources) == 5
    assert all(str(resource.uri).startswith("internal://") for resource in resources)


def test_sse_http_requires_bearer_token():
    app = server_module.build_http_app("sse", host="127.0.0.1", api_token="test-token")

    with TestClient(app) as client:
        response = client.get("/sse")

    assert response.status_code == 401
    assert "www-authenticate" in response.headers
