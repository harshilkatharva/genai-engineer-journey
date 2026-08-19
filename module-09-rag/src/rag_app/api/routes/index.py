from fastapi import APIRouter, HTTPException

from rag_app.models import (
    ProcessRequest,
)
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
        return await index_service_manager.index(request)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
