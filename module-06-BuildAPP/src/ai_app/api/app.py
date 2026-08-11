from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from ai_app.api.exception_handler import register_exception_handler
from ai_app.api.limiter import limiter
from ai_app.api.routes import chat, sentiment, summarization, start, report
from ai_app.core.config import X_API_KEY

app = FastAPI(title="AI Application API", version="0.141.1")

app.state.limiter = limiter
register_exception_handler(app)

api_key_header = APIKeyHeader(name="X_API_KEY")


async def verify_api_key(x_api_key: str = Security(api_key_header)):
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key to access")
    return x_api_key


app.include_router(
    start.router, prefix="/start", tags=["start"], dependencies=[Security(verify_api_key)]
)
app.include_router(
    chat.router, prefix="/chat", tags=["chat"], dependencies=[Security(verify_api_key)]
)
app.include_router(
    summarization.router,
    prefix="/summarization",
    tags=["summarization"],
    dependencies=[Security(verify_api_key)],
)
app.include_router(
    sentiment.router,
    prefix="/sentiment",
    tags=["sentiment"],
    dependencies=[Security(verify_api_key)],
)
app.include_router(
    report.router, prefix="/report", tags=["report"], dependencies=[Security(verify_api_key)]
)
