from mcp.server import MCPServer

mcp = MCPServer("Company Knowledge Server")


@mcp.tool()
def search_internal_docs(query: str, max_results: int = 5) -> list[str]:
    """Search internal company documentation."""

    # Pretend this calls your existing RAG service.
    return [f"Document result for: {query}" for _ in range(max_results)]


@mcp.resource("docs://company/about")
def company_about() -> str:
    """Return company information."""
    return "Our company builds AI-powered software."


@mcp.prompt()
def summarize_document(document: str) -> str:
    """Create a reusable document summarization prompt."""
    return f"Summarize the following document:\n\n{document}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
