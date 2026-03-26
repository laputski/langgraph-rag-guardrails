from __future__ import annotations

from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse


def get_user_identifier(request: Request) -> str:
    """Use X-User-ID header if present, else fall back to IP address."""
    return request.headers.get("X-User-ID") or get_remote_address(request)


limiter = Limiter(key_func=get_user_identifier)


def setup_rate_limiter(app: FastAPI, rate_limit_per_minute: int) -> None:
    """Attach the slowapi rate limiter to the FastAPI app."""
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "detail": f"Rate limit exceeded. Max {rate_limit_per_minute} requests/minute.",
            },
        )
