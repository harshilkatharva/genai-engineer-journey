import uuid

from fastapi import FastAPI, Request

from rag_app.api.routes.evalution import router as evalution_router
from rag_app.api.routes.index import router as index_router
from rag_app.api.routes.rag import router as rag_router
from rag_app.api.routes.retrive import router as retrive_router
from rag_app.api.routes.upsert import router as upsert_router
from rag_app.core import get_settings
from rag_app.observability.context import (
    reset_request_id,
    set_request_id,
)
from rag_app.observability.logger import logger

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)


@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next,
):
    request_id = uuid.uuid4()

    token = set_request_id(request_id)

    try:
        response = await call_next(request)

        return response

    # except Exception as exc:
    #     logger.exception(
    #         "Request failed",
    #         event="request_failed",
    #         component="api",
    #         endpoint=request.url.path,
    #         status="error",
    #         error_type=type(exc).__name__,
    #     )

    #     raise

    finally:
        reset_request_id(token)


app.include_router(rag_router, prefix="/rag", tags=["rag"])
app.include_router(index_router, prefix="/index", tags=["index"])
app.include_router(retrive_router, prefix="/retrive", tags=["retrive"])
app.include_router(upsert_router, prefix="/upsert", tags=["upsert"])
app.include_router(evalution_router, prefix="/evalution", tags=["evalution"])


@app.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.get("/health")
def health() -> dict:
    logger.info("Health Check", event="health_check", endpoint="/health", status="success")
    return {
        "status": "ok",
    }


@app.get("/genrate_uuid")
def genrate_uuid():
    return uuid.uuid4()
