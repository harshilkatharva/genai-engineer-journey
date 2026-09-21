import asyncio
import sys

from mcp.server.mcpserver import MCPServer


mcp = MCPServer("Slow Server")


@mcp.tool()
async def slow_tool(query: str) -> dict:
    """
    Simulates a slow/unresponsive server.
    The client timeout is 5 seconds,
    but this tool takes 10 seconds.
    """

    print("slow_tool started...", file=sys.stderr)

    await asyncio.sleep(10)

    print("slow_tool finished.", file=sys.stderr)

    return {
        "query": query,
        "result": "Response from slow server",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
