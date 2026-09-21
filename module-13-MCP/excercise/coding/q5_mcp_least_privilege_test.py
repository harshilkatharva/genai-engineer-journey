import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="python3",
    args=["excercise/coding/q5_least_privilege_server.py"],
)


@pytest.mark.asyncio
async def test_mcp_tool_least_privilege():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Verify the MCP tool exists
            tools = await session.list_tools()

            tool_names = [tool.name for tool in tools.tools]

            assert "get_employee" in tool_names

            # Allowed operation
            result = await session.call_tool("get_employee", {"employee_id": "EMP-001"})

            assert not result.is_error

            # Operation outside the documented scope
            result = await session.call_tool("get_employee", {"employee_id": "EMP-002"})

            assert result.is_error
