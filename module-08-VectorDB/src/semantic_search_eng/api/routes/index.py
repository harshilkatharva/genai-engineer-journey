from fastapi import APIRouter, HTTPException

from semantic_search_eng.models import (
    ProcessRequest,
)
from semantic_search_eng.services.index_services import IndexServiceManager

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
