import asyncio

from internal_tools_mcp.server import mcp


def test_server_registers_tools_and_resources():
    tools = asyncio.run(mcp.list_tools())
    resources = asyncio.run(mcp.list_resources())

    assert {tool.name for tool in tools} == {
        "module9_chat_answer",
        "module8_retrieve_documents",
    }
    assert len(resources) == 5
    assert all(str(resource.uri).startswith("internal://") for resource in resources)
