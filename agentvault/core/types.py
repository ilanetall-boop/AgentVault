"""Core types and Pydantic models for AgentVault."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Types of memory supported by AgentVault."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Memory(BaseModel):
    """A single memory unit stored in AgentVault.

    Represents any piece of information an agent wants to persist,
    whether it's an event (episodic), a fact (semantic), or a skill (procedural).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    type: MemoryType
    content: str
    metadata: dict[str, Any] = {}
    embedding: list[float] | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    tags: list[str] = []
    source: str | None = None
    related_memories: list[str] = []


class EpisodicMemory(Memory):
    """Memory of a specific event or interaction.

    Automatically typed as episodic. Includes event-specific fields.
    """

    type: MemoryType = MemoryType.EPISODIC
    event: str | None = None
    context: dict[str, Any] = {}
    participants: list[str] = []


class SemanticMemory(Memory):
    """Memory of a fact, preference, or piece of knowledge.

    Automatically typed as semantic. Supports knowledge graph relations.
    """

    type: MemoryType = MemoryType.SEMANTIC
    subject: str | None = None
    predicate: str | None = None
    obj: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ProceduralMemory(Memory):
    """Memory of a skill, workflow, or learned procedure.

    Automatically typed as procedural. Stores steps and success metrics.
    """

    type: MemoryType = MemoryType.PROCEDURAL
    name: str | None = None
    steps: list[str] = []
    learned_from: str | None = None
    success_count: int = 0
    failure_count: int = 0


class MemoryQuery(BaseModel):
    """Query for searching and recalling memories."""

    query: str
    agent_id: str
    types: list[MemoryType] = [
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.PROCEDURAL,
    ]
    top_k: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    time_range: tuple[datetime, datetime] | None = None
    tags: list[str] = []
    include_shared: bool = True


class RecallResult(BaseModel):
    """Result of a memory recall/search operation."""

    memories: list[Memory]
    relevance_scores: list[float]
    total_found: int
    query_time_ms: float


class ConsolidationResult(BaseModel):
    """Result of a memory consolidation operation."""

    merged: int = 0
    promoted: int = 0
    forgotten: int = 0
    reinforced: int = 0
    total_processed: int = 0
    duration_ms: float = 0.0


class AgentPermission(StrEnum):
    """Permission levels for multi-agent memory access."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class SharedNamespace(BaseModel):
    """A shared memory namespace for multi-agent collaboration."""

    namespace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    owner_agent_id: str
    permissions: dict[str, AgentPermission] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
