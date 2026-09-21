import os
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("PaginationServer")

# Simulate a large dataset
DATASET = [{"id": i, "name": f"Record {i}"} for i in range(1, 101)]


@mcp.resource("records://page/{page}")
def get_records_page(page: int) -> dict:
    page_size = 10

    if page == 0:
        os._exit(1)

    if not 1 <= page <= 10:
        raise ValueError("Page range must be 1 to 100")

    start = (page - 1) * page_size
    end = start + page_size

    results = DATASET[start:end]

    return {
        "page": page,
        "page_size": page_size,
        "total": len(DATASET),
        "results": results,
        "has_next": end < len(DATASET),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
