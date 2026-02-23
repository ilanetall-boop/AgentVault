"""Agent management routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentvault.api.dependencies import get_manager
from agentvault.core.types import MemoryType

router = APIRouter()


@router.get("/agents")
async def list_agents() -> dict:
    """List all agents that have stored memories."""
    manager = get_manager()
    agents = await manager.list_agents()
    return {"agents": agents}


@router.get("/agents/{agent_id}/memories")
async def list_agent_memories(
    agent_id: str,
    type: str | None = None,  # noqa: A002
    min_importance: float = 0.0,
    limit: int = 100,
) -> dict:
    """List all memories for an agent."""
    manager = get_manager()

    try:
        memory_type = MemoryType(type) if type else None
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid memory type: {type}")

    memories = await manager.get_memories(
        agent_id=agent_id,
        memory_type=memory_type,
        min_importance=min_importance,
        limit=limit,
    )

    return {
        "agent_id": agent_id,
        "memories": [
            {
                "id": m.id,
                "type": m.type.value,
                "content": m.content,
                "importance": m.importance,
                "tags": m.tags,
                "access_count": m.access_count,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ],
        "total": len(memories),
    }


@router.get("/agents/{agent_id}/stats")
async def agent_stats(agent_id: str) -> dict:
    """Get memory statistics for an agent."""
    manager = get_manager()
    total = await manager.count(agent_id)
    episodic = await manager.count(agent_id, MemoryType.EPISODIC)
    semantic = await manager.count(agent_id, MemoryType.SEMANTIC)
    procedural = await manager.count(agent_id, MemoryType.PROCEDURAL)

    return {
        "agent_id": agent_id,
        "total_memories": total,
        "by_type": {
            "episodic": episodic,
            "semantic": semantic,
            "procedural": procedural,
        },
    }


@router.post("/agents/{agent_id}/consolidate")
async def consolidate_agent(agent_id: str) -> dict:
    """Run consolidation for an agent's memories."""
    manager = get_manager()
    result = await manager.consolidate(agent_id)
    return {
        "agent_id": agent_id,
        "merged": result.merged,
        "promoted": result.promoted,
        "forgotten": result.forgotten,
        "reinforced": result.reinforced,
        "total_processed": result.total_processed,
        "duration_ms": result.duration_ms,
    }


@router.delete("/agents/{agent_id}/memories")
async def forget_agent_memories(
    agent_id: str,
    older_than: str | None = None,
    importance_below: float | None = None,
) -> dict:
    """Forget (delete) memories matching criteria."""
    manager = get_manager()
    count = await manager.forget(
        agent_id=agent_id,
        older_than=older_than,
        importance_below=importance_below,
    )
    return {
        "agent_id": agent_id,
        "forgotten": count,
    }
