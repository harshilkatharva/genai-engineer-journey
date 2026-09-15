from fastapi import FastAPI

from ..core.config import get_settings
from ..schemas.rag import HealthResponse
from ..services.rag import RAGService
from .routes.rag import router as rag_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.state.rag_pipeline = RAGService(settings)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        pipeline = app.state.rag_pipeline
        return HealthResponse(status="ok", ready=pipeline.ready, documents=pipeline.document_count)

    app.include_router(rag_router)
    return app


app = create_app()
