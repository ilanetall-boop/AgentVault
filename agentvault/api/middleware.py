"""API middleware — CORS, rate limiting, request logging, authentication."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# API key for optional authentication
# Set AGENTVAULT_API_KEY env var to enable auth, leave unset for local-only use
_API_KEY: str | None = os.environ.get("AGENTVAULT_API_KEY")


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication via X-API-Key header.

    Only active when AGENTVAULT_API_KEY environment variable is set.
    Health endpoint is always public.
    """

    async def dispatch(self, request: Request, call_next):
        # Health check is always public
        if request.url.path == "/health":
            return await call_next(request)

        # Skip auth if no API key configured (local mode)
        if _API_KEY is None:
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != _API_KEY:
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming requests with timing information."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting per client IP.

    Limits requests per minute per IP address.
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self.window_seconds
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


def setup_middleware(app: FastAPI) -> None:
    """Configure all middleware for the application."""
    # CORS: configurable origins, default to localhost only
    allowed_origins = os.environ.get(
        "AGENTVAULT_CORS_ORIGINS", "http://localhost:8420"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=200, window_seconds=60)
