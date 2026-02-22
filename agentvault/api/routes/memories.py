"""CRUD routes for memories."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from agentvault.api.dependencies import get_manager
from agentvault.core.types import MemoryType

router = APIRouter()


class CreateMemoryRequest(BaseModel):
    """Request body for creating a memory."""

    agent_id: str
    content: str
    type: str | None = None
    importance: float | None = None
    tags: list[str] = []
    metadata: dict[str, Any] = {}
    source: str | None = None


class MemoryResponse(BaseModel):
    """Response body for a memory."""

    id: str
    agent_id: str
    type: str
    content: str
    importance: float
    tags: list[str]
    access_count: int
    created_at: str
    source: str | None = None


@router.post("/memories", status_code=201)
async def create_memory(request: CreateMemoryRequest) -> dict:
    """Create a new memory."""
    manager = get_manager()

    memory_type: MemoryType | None = None
    if request.type:
        memory_type = MemoryType(request.type)

    memory = await manager.remember(
        agent_id=request.agent_id,
        content=request.content,
        memory_type=memory_type,
        importance=request.importance,
        tags=request.tags if request.tags else None,
        metadata=request.metadata,
        source=request.source,
    )
    return {
        "id": memory.id,
        "agent_id": memory.agent_id,
        "type": memory.type.value,
        "content": memory.content,
        "importance": memory.importance,
        "tags": memory.tags,
        "access_count": memory.access_count,
        "created_at": memory.created_at.isoformat(),
        "source": memory.source,
    }


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: str) -> dict:
    """Retrieve a memory by ID."""
    manager = get_manager()
    memory = await manager.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "id": memory.id,
        "agent_id": memory.agent_id,
        "type": memory.type.value,
        "content": memory.content,
        "importance": memory.importance,
        "tags": memory.tags,
        "access_count": memory.access_count,
        "created_at": memory.created_at.isoformat(),
        "source": memory.source,
    }


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(memory_id: str) -> Response:
    """Delete a memory by ID."""
    manager = get_manager()
    memory = await manager.get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    await manager._store.delete(memory_id)
    return Response(status_code=204)
