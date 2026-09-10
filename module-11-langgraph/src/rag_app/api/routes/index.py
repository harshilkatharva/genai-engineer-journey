from fastapi import APIRouter, HTTPException

from rag_app.models import (
    ProcessRequest,
)
from rag_app.observability.logger import logger
from rag_app.services.index_services import IndexServiceManager

router = APIRouter()


index_service_manager = IndexServiceManager()


@router.post(
    "/process",
)
async def process_documents(
    request: ProcessRequest,
) -> dict:
    try:
        logger.info(
            "Index request",
            event="Index Request",
            component="api",
            endpoint="/index/process",
            tenant_id=request.tenant_id,
        )
        return await index_service_manager.index(request)

    except Exception as exc:
        logger.exception(
            "Exrror raise in Indexing",
            event="Index Error",
            component="api",
            endpoint="/index/process",
            tenant_id=request.tenant_id,
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
