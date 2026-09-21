from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoServer")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    return a + b


@mcp.resource("greeting://hello")
def greeting() -> str:
    return "Hello from MCP Server!"


if __name__ == "__main__":
    mcp.run(transport="stdio")
