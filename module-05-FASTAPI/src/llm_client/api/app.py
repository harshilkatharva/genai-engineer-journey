from fastapi import FastAPI

from llm_client.api.routes import health

app = FastAPI(title="AI Application API", version="0.141.1")

app.include_router(health.router, prefix="/health", tags=["health"])
