"""Module 8 document retrieval adapter."""

import sys
from collections.abc import Callable
from contextlib import redirect_stdout

from internal_tools_mcp.integrations import module8_services


async def retrieve_documents(
    tenant_id: str,
    query: str,
    document_type: str | None = None,
    top_k: int = 5,
    services_factory: Callable[[], tuple[type, type]] | None = None,
) -> dict[str, object]:
    """Run Module 8 RetriveServiceManager and return its response."""

    # Module 8 contains diagnostic prints; stdio MCP reserves stdout for JSON-RPC.
    with redirect_stdout(sys.stderr):
        factory = services_factory or module8_services
        manager_class, request_class = factory()
        service = manager_class()
        response = await service.retrive_chunks(
            request_class(
                tenant_id=tenant_id,
                query=query,
                document_type=document_type,
                top_k=top_k,
            )
        )
    return response.model_dump()
