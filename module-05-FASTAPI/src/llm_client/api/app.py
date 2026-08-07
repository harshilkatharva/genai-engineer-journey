from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from llm_client.api.exception_handler import register_exception_handler
from llm_client.api.limiter import limiter
from llm_client.api.routes import chat, health, prompts
from llm_client.config import X_API_KEY

app = FastAPI(title="AI Application API", version="0.141.1")

app.state.limiter = limiter
register_exception_handler(app)

api_key_header = APIKeyHeader(name="X_API_KEY")


async def verify_api_key(x_api_key: str = Security(api_key_header)):
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key to access")
    return x_api_key


app.include_router(health.router, prefix="/health", tags=["health"])


app.include_router(
    chat.router, prefix="/chat", tags=["chat"], dependencies=[Security(verify_api_key)]
)
app.include_router(
    prompts.router, prefix="/prompts", tags=["prompts"], dependencies=[Security(verify_api_key)]
)
