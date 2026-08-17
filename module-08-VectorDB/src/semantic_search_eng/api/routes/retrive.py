from fastapi import APIRouter, HTTPException

from semantic_search_eng.models import RetriveRequest, RetriveResponse
from semantic_search_eng.services.retrive_services import RetriveServiceManager

router = APIRouter()

retrive_service_manager = RetriveServiceManager()


@router.post(
    "/",
    response_model=RetriveResponse,
)
async def search(
    request: RetriveRequest,
) -> RetriveResponse:
    try:
        return await retrive_service_manager.retrive_chunks(request)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
