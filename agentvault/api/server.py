"""FastAPI application server for AgentVault."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentvault.api.dependencies import set_manager
from agentvault.api.middleware import setup_middleware
from agentvault.api.routes.agents import router as agents_router
from agentvault.api.routes.health import router as health_router
from agentvault.api.routes.memories import router as memories_router
from agentvault.api.routes.search import router as search_router
from agentvault.core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


def create_app(
    db_path: str | None = None,
    faiss_path: str | None = None,
    chroma_path: str | None = None,  # Backwards compatibility alias
    manager: MemoryManager | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: Path to SQLite database.
        faiss_path: Path to FAISS storage.
        chroma_path: Deprecated alias for faiss_path.
        manager: Optional pre-configured MemoryManager (for testing).

    Returns:
        Configured FastAPI application.
    """
    if manager is None:
        vector_path = faiss_path or chroma_path
        manager = MemoryManager(db_path=db_path, faiss_path=vector_path)
        set_manager(manager)
        should_manage_lifecycle = True
    else:
        set_manager(manager)
        should_manage_lifecycle = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan: initialize on startup, cleanup on shutdown."""
        if should_manage_lifecycle:
            await manager.initialize()
        logger.info("AgentVault API started")
        yield
        if should_manage_lifecycle:
            await manager.close()
        logger.info("AgentVault API stopped")

    app = FastAPI(
        title="AgentVault API",
        description="Persistent Memory for AI Agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    setup_middleware(app)

    app.include_router(health_router, tags=["Health"])
    app.include_router(memories_router, prefix="/api/v1", tags=["Memories"])
    app.include_router(search_router, prefix="/api/v1", tags=["Search"])
    app.include_router(agents_router, prefix="/api/v1", tags=["Agents"])

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    db_path: str | None = None,
    faiss_path: str | None = None,
    chroma_path: str | None = None,  # Backwards compatibility alias
) -> None:
    """Run the AgentVault API server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        db_path: Path to SQLite database.
        faiss_path: Path to FAISS storage.
        chroma_path: Deprecated alias for faiss_path.
    """
    import uvicorn

    vector_path = faiss_path or chroma_path
    app = create_app(db_path=db_path, faiss_path=vector_path)
    uvicorn.run(app, host=host, port=port)
