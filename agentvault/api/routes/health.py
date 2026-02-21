"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Check if the API is healthy."""
    return {
        "status": "healthy",
        "service": "agentvault",
        "version": "0.1.0",
    }
