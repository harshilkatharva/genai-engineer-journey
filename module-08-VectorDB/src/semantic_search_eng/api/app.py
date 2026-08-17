from fastapi import FastAPI

from semantic_search_eng.api.routes.search import router as search_router
from semantic_search_eng.api.routes.index import router as index_router
from semantic_search_eng.config import get_settings

import uuid

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.include_router(index_router, prefix="/index", tags=["index"])
app.include_router(search_router, prefix="/search", tags=["search"])


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
