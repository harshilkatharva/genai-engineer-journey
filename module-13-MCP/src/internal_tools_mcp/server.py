"""MCP server exposing Module 8 retrieval and Module 9 RAG chat."""

import argparse
import asyncio
import logging
import os
import secrets

from starlette.responses import JSONResponse

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


def require_api_key(app, api_token: str):
    """Protect HTTP/SSE app endpoints with a shared bearer token."""

    async def auth_middleware(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        authorization = headers.get("authorization")

        if authorization is None or not authorization.lower().startswith("bearer "):
            response = JSONResponse(
                {
                    "error": "unauthorized",
                    "error_description": "Authentication required",
                },
                status_code=401,
            )
            response.headers["WWW-Authenticate"] = (
                'Bearer realm="mcp", error="invalid_token"'
            )
            await response(scope, receive, send)
            return

        provided_token = authorization[7:].strip()
        if not secrets.compare_digest(provided_token, api_token):
            response = JSONResponse(
                {"error": "unauthorized", "error_description": "Invalid bearer token"},
                status_code=401,
            )
            response.headers["WWW-Authenticate"] = (
                'Bearer realm="mcp", error="invalid_token"'
            )
            await response(scope, receive, send)
            return

        await app(scope, receive, send)

    return auth_middleware


def build_http_app(transport: str, *, host: str, api_token: str):
    if transport == "sse":
        return require_api_key(mcp.sse_app(host=host), api_token)
    if transport == "streamable-http":
        return require_api_key(mcp.streamable_http_app(host=host), api_token)
    raise ValueError(f"Unsupported transport for authenticated HTTP app: {transport}")


async def run_http_transport(
    transport: str, *, host: str, port: int, api_token: str
) -> None:
    import uvicorn

    app = build_http_app(transport, host=host, api_token=api_token)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport", choices=("stdio", "sse", "streamable-http"), default="stdio"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-token",
        default=os.getenv("MCP_API_TOKEN"),
        help="Shared bearer token required for HTTP/SSE transports. Set MCP_API_TOKEN as an environment variable.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    if args.transport in {"sse", "streamable-http"}:
        if not args.api_token:
            parser.error(
                "--api-token or MCP_API_TOKEN is required when using the HTTP/SSE transports."
            )
        asyncio.run(
            run_http_transport(
                args.transport, host=args.host, port=args.port, api_token=args.api_token
            )
        )
        return

    mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
