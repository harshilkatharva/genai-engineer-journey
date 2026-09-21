"""Module 9 RAG chat adapter."""

import asyncio
import os
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from uuid import UUID, uuid4

from internal_tools_mcp.integrations import module9_services

_RAG_CALL_LOCK = asyncio.Lock()
_MODULE9_ROOT = Path(__file__).resolve().parents[4] / "module-09-rag"


async def ask_question(
    query: str,
    tenant_id: str,
    services_factory: Callable[[], tuple[type, type]] | None = None,
) -> dict[str, object]:
    """Run Module 9 RAGChat and return its response as a dictionary."""

    async with _RAG_CALL_LOCK:
        previous_directory = Path.cwd()
        os.chdir(_MODULE9_ROOT)
        try:
            # Module 9's logger and query helpers write diagnostics to stdout.
            # stdio MCP reserves stdout for JSON-RPC, so keep those diagnostics on stderr.
            with redirect_stdout(sys.stderr):
                factory = services_factory or module9_services
                rag_chat_class, request_class = factory()
                request_token = None
                reset_request_id = None
                if services_factory is None:
                    from rag_app.observability.context import (
                        reset_request_id,
                        set_request_id,
                    )

                    request_token = set_request_id(uuid4())
                service = rag_chat_class()
                try:
                    response = await service.get_chat_answer(
                        request_class(query=query, tenant_id=UUID(tenant_id))
                    )
                    return response.model_dump()
                finally:
                    if reset_request_id is not None and request_token is not None:
                        reset_request_id(request_token)
        finally:
            os.chdir(previous_directory)
