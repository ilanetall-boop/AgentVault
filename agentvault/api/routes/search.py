"""Search and recall routes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from agentvault.api.dependencies import get_manager
from agentvault.core.types import MemoryType

router = APIRouter()


class SearchRequest(BaseModel):
    """Request body for searching memories."""

    agent_id: str
    query: str
    top_k: int = 10
    types: list[str] | None = None
    min_importance: float = 0.0
    tags: list[str] = []


@router.post("/search")
async def search_memories(request: SearchRequest) -> dict:
    """Search for relevant memories."""
    manager = get_manager()

    memory_types: list[MemoryType] | None = None
    if request.types:
        memory_types = [MemoryType(t) for t in request.types]

    result = await manager.recall(
        agent_id=request.agent_id,
        query=request.query,
        top_k=request.top_k,
        types=memory_types,
        min_importance=request.min_importance,
        tags=request.tags if request.tags else None,
    )

    return {
        "memories": [
            {
                "id": m.id,
                "agent_id": m.agent_id,
                "type": m.type.value,
                "content": m.content,
                "importance": m.importance,
                "tags": m.tags,
                "relevance_score": score,
                "created_at": m.created_at.isoformat(),
            }
            for m, score in zip(result.memories, result.relevance_scores, strict=False)
        ],
        "total_found": result.total_found,
        "query_time_ms": result.query_time_ms,
    }
