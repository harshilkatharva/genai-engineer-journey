from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ai_app.exceptions import (
    ConfigError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMContentFilterError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)


def register_exception_handler(app: FastAPI) -> None:
    app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler,
    )

    @app.exception_handler(LLMTimeoutError)
    async def timeout_handler(request: Request, exc: LLMTimeoutError):
        return JSONResponse(
            status_code=504,
            content={"detail": str(exc.user_message)},
        )

    @app.exception_handler(LLMRateLimitError)
    async def rate_limit_handler(request: Request, exc: LLMRateLimitError):
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc.user_message)},
        )

    @app.exception_handler(LLMAuthenticationError)
    async def auth_handler(request: Request, exc: LLMAuthenticationError):
        return JSONResponse(
            status_code=401,
            content={"detail": str(exc.user_message)},
        )

    @app.exception_handler(LLMConnectionError)
    async def connection_handler(request: Request, exc: LLMConnectionError):
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc.user_message)},
        )

    @app.exception_handler(LLMContentFilterError)
    async def content_filter_handler(request: Request, exc: LLMContentFilterError):
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc.user_message)},
        )

    @app.exception_handler(LLMInvalidResponseError)
    async def invalid_response_handler(request: Request, exc: LLMInvalidResponseError):
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc.user_message)},
        )

    @app.exception_handler(ConfigError)
    async def config_handler(request: Request, exc: ConfigError):
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc.message)},
        )

    @app.exception_handler(LLMError)
    async def llm_unknown_handler(request: Request, exc: LLMError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc.message)},
        )
