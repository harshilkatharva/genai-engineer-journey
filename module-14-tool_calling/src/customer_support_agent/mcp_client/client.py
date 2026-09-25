from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, Self

from mcp import Client, StdioServerParameters

from customer_support_agent.models import ToolDefinition


class MCPToolClient(Protocol):
    async def list_tools(self) -> list[ToolDefinition]: ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...

    async def read_resource(self, uri: str) -> str: ...


class MCPClientError(RuntimeError):
    """Raised when the MCP server cannot complete a client operation."""


class MCPClient:
    """MCP client using a subprocess and the SDK's stdio transport."""

    def __init__(self, customer_id: str | None = None, timeout: float = 30.0) -> None:
        self.customer_id = customer_id
        self.timeout = timeout
        self._session: Client | None = None

    async def __aenter__(self) -> Self:
        root = Path(__file__).resolve().parents[3]
        source_path = str(root / "src")
        python_path = os.pathsep.join(filter(None, [source_path, os.environ.get("PYTHONPATH", "")]))
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "customer_support_agent.mcp_server.server"],
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": python_path,
                "CUSTOMER_ID": self.customer_id or "",
            },
        )
        self._session = Client(server, read_timeout_seconds=self.timeout, raise_exceptions=True)
        try:
            await self._session.__aenter__()
        except Exception as exc:
            await self.__aexit__(None, None, None)
            raise MCPClientError("Unable to initialize the support MCP server") from exc
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._session is not None:
            await self._session.__aexit__(exc_type, exc, traceback)
            self._session = None
        self._session = None

    def _require_session(self) -> Client:
        if self._session is None:
            raise MCPClientError("MCP client is not connected")
        return self._session

    async def list_tools(self) -> list[ToolDefinition]:
        try:
            result = await asyncio.wait_for(
                self._require_session().list_tools(), timeout=self.timeout
            )
            return [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.input_schema,
                )
                for tool in result.tools
            ]
        except Exception as exc:
            raise MCPClientError("Unable to discover support tools") from exc

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        try:
            result = await asyncio.wait_for(
                self._require_session().call_tool(name, dict(arguments)), timeout=self.timeout
            )
        except Exception as exc:
            raise MCPClientError(f"MCP tool call failed: {name}") from exc
        if result.isError:
            raise MCPClientError(f"MCP tool call failed: {name}")
        if result.structured_content:
            return result.structured_content
        text_parts = [item.text for item in result.content if hasattr(item, "text")]
        return "\n".join(text_parts)

    async def read_resource(self, uri: str) -> str:
        try:
            result = await asyncio.wait_for(
                self._require_session().read_resource(uri), timeout=self.timeout
            )
        except Exception as exc:
            raise MCPClientError("Unable to read support policy resource") from exc
        text_parts = [item.text for item in result.contents if hasattr(item, "text")]
        return "\n".join(text_parts)
