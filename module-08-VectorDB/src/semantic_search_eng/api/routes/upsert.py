from fastapi import APIRouter, HTTPException

from semantic_search_eng.models import UpsertRequest
from semantic_search_eng.services.upsert_services import UpsertServiceManager

router = APIRouter()

update_service_manager = UpsertServiceManager()


@router.post("/chunks")
async def upsert_chunks(request: UpsertRequest):
    try:
        return await update_service_manager.upsert_chunks(request=request)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
