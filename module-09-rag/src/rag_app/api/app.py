import uuid

from fastapi import FastAPI

from rag_app.api.routes.index import router as index_router
from rag_app.api.routes.rag import router as rag_router
from rag_app.api.routes.retrive import router as retrive_router
from rag_app.api.routes.upsert import router as upsert_router
from rag_app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(rag_router, prefix="/rag", tags=["rag"])
app.include_router(index_router, prefix="/index", tags=["index"])
app.include_router(retrive_router, prefix="/retrive", tags=["retrive"])
app.include_router(upsert_router, prefix="/upsert", tags=["upsert"])


@app.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
    }


@app.get("/genrate_uuid")
def genrate_uuid():
    return uuid.uuid4()
