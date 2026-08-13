from fastapi import FastAPI

from semantic_search_eng.api.routes.search import (
    router as search_router,
)
from semantic_search_eng.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

# for this model only without prefix
app.include_router(search_router, tags=["search"])


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
