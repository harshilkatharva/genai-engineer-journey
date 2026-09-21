import asyncio
import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="python3",
    args=["excercise/coding/q9_slow_server.py"],
)


async def call_with_timeout(session, tool_name, arguments, timeout=2):
    try:
        return await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return None


@pytest.mark.asyncio
async def test_slow_server_degrades_gracefully():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await call_with_timeout(
                session,
                "slow_tool",
                {"query": "test"},
                timeout=2,
            )

            # Client should not hang indefinitely
            assert result is None
