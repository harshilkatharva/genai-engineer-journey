from fastapi import APIRouter, HTTPException

from rag_app.models import UpsertRequest
from rag_app.observability.logger import logger
from rag_app.services.upsert_services import UpsertServiceManager

router = APIRouter()

update_service_manager = UpsertServiceManager()


@router.post("/chunks")
async def upsert_chunks(request: UpsertRequest):
    try:
        logger.info(
            "Upsert Request",
            event="Upsert Request",
            component="api",
            endpoint="/upsert/chunks",
            tenant_id=request.tenant_id,
        )
        return await update_service_manager.upsert_chunks(request=request)

    except Exception as exc:
        logger.exception(
            "Exrror raise in Upsert",
            event="upsert Error",
            component="api",
            endpoint="/upsert/chunks",
            tenant_id=request.tenant_id,
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
