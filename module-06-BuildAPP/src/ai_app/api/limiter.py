from fastapi import Request
from slowapi import Limiter


def get_api_key(request: Request) -> str:
    return request.headers.get("X_API_KEY", "anonymus")


limiter = Limiter(
    key_func=get_api_key,
    default_limits=["300/days"],
    headers_enabled=True,
)
