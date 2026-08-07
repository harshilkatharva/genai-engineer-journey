from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader
from llm_client.config import X_API_KEY
from llm_client.api.routes import health, chat

app = FastAPI(title="AI Application API", version="0.141.1")

api_key_header = APIKeyHeader(name="X_API_KEY")


async def verify_api_key(x_api_key: str = Depends(api_key_header)):
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(
    chat.router, prefix="/chat", tags=["chat"], dependencies=[Depends(verify_api_key)]
)
