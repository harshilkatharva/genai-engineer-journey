"""MCP server exposing Module 8 retrieval and Module 9 RAG chat."""

import argparse
import logging

from mcp.server.mcpserver import MCPServer

from internal_tools_mcp.mcp.tools.tools import register_tools
from internal_tools_mcp.mcp.resources.resource_catalog import register_resources

mcp = MCPServer(
    "Internal Tools MCP",
    description="Read-only Module 8 document retrieval and Module 9 RAG chat.",
    version="0.1.0",
)
register_tools(mcp)
register_resources(mcp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport", choices=("stdio", "sse", "streamable-http"), default="stdio"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
