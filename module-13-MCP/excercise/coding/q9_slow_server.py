import asyncio
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("SlowServer")


@mcp.tool()
async def slow_tool(query: str):
    # Simulate an unresponsive/slow server
    await asyncio.sleep(10)

    return {"result": query}


if __name__ == "__main__":
    mcp.run(transport="stdio")
