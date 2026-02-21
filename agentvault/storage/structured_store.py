"""SQLite/SQLAlchemy-based structured store implementation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    and_,
    create_engine,
    delete,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from agentvault.core.types import Memory, MemoryQuery, MemoryType

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class MemoryRow(Base):
    """SQLAlchemy model for the memories table."""

    __tablename__ = "memories"

    id = Column(String, primary_key=True)
    agent_id = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, default="{}")
    importance = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    tags_json = Column(Text, default="[]")
    source = Column(String, nullable=True)
    related_memories_json = Column(Text, default="[]")
    extra_json = Column(Text, default="{}")


def _memory_to_row(memory: Memory) -> MemoryRow:
    """Convert a Memory pydantic model to a SQLAlchemy row."""
    extra: dict = {}
    if hasattr(memory, "event") and memory.event is not None:  # type: ignore[attr-defined]
        extra["event"] = memory.event  # type: ignore[attr-defined]
    if hasattr(memory, "context"):
        ctx = getattr(memory, "context", None)
        if ctx:
            extra["context"] = ctx
    if hasattr(memory, "steps"):
        steps = getattr(memory, "steps", None)
        if steps:
            extra["steps"] = steps
    if hasattr(memory, "name"):
        name = getattr(memory, "name", None)
        if name:
            extra["name"] = name
    if hasattr(memory, "subject"):
        subj = getattr(memory, "subject", None)
        if subj:
            extra["subject"] = subj
    if hasattr(memory, "predicate"):
        pred = getattr(memory, "predicate", None)
        if pred:
            extra["predicate"] = pred
    if hasattr(memory, "confidence"):
        conf = getattr(memory, "confidence", None)
        if conf is not None:
            extra["confidence"] = conf

    return MemoryRow(
        id=memory.id,
        agent_id=memory.agent_id,
        type=memory.type.value,
        content=memory.content,
        metadata_json=json.dumps(memory.metadata),
        importance=memory.importance,
        access_count=memory.access_count,
        created_at=memory.created_at,
        last_accessed=memory.last_accessed,
        expires_at=memory.expires_at,
        tags_json=json.dumps(memory.tags),
        source=memory.source,
        related_memories_json=json.dumps(memory.related_memories),
        extra_json=json.dumps(extra),
    )


def _row_to_memory(row: MemoryRow) -> Memory:
    """Convert a SQLAlchemy row to a Memory pydantic model."""
    return Memory(
        id=row.id,
        agent_id=row.agent_id,
        type=MemoryType(row.type),
        content=row.content,
        metadata=json.loads(row.metadata_json) if row.metadata_json else {},
        importance=row.importance,
        access_count=row.access_count,
        created_at=row.created_at,
        last_accessed=row.last_accessed,
        expires_at=row.expires_at,
        tags=json.loads(row.tags_json) if row.tags_json else [],
        source=row.source,
        related_memories=json.loads(row.related_memories_json)
        if row.related_memories_json
        else [],
    )


def _parse_duration(duration: str) -> timedelta:
    """Parse a duration string like '30d', '12h', '7d' into a timedelta."""
    unit = duration[-1]
    value = int(duration[:-1])
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "s":
        return timedelta(seconds=value)
    raise ValueError(f"Unknown duration unit: {unit}")


class SQLiteStructuredStore:
    """Structured store backed by SQLite via SQLAlchemy async.

    Provides CRUD operations and structured queries for memories.
    Uses SQLite for local/development and can be swapped for PostgreSQL.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or str(
            Path.home() / ".agentvault" / "memories.db"
        )
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite+aiosqlite:///{self._db_path}"
        self._engine = create_async_engine(url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("SQLite store initialized at %s", self._db_path)

    def _get_session(self) -> AsyncSession:
        """Get a new async session."""
        if self._session_factory is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._session_factory()

    async def save_memory(self, memory: Memory) -> None:
        """Save a memory to SQLite."""
        row = _memory_to_row(memory)
        async with self._get_session() as session:
            await session.merge(row)
            await session.commit()
        logger.debug("Saved memory %s", memory.id)

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        async with self._get_session() as session:
            result = await session.execute(
                select(MemoryRow).where(MemoryRow.id == memory_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _row_to_memory(row)

    async def get_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        min_importance: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Retrieve memories with filters."""
        async with self._get_session() as session:
            stmt = select(MemoryRow).where(
                and_(
                    MemoryRow.agent_id == agent_id,
                    MemoryRow.importance >= min_importance,
                )
            )
            if memory_type is not None:
                stmt = stmt.where(MemoryRow.type == memory_type.value)
            stmt = stmt.order_by(MemoryRow.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(stmt)
            rows = result.scalars().all()

            memories = [_row_to_memory(row) for row in rows]
            if tags:
                memories = [
                    m for m in memories if any(t in m.tags for t in tags)
                ]
            return memories

    async def update_memory(self, memory: Memory) -> None:
        """Update an existing memory."""
        await self.save_memory(memory)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        async with self._get_session() as session:
            await session.execute(
                delete(MemoryRow).where(MemoryRow.id == memory_id)
            )
            await session.commit()
        logger.debug("Deleted memory %s", memory_id)

    async def delete_memories(
        self,
        agent_id: str,
        older_than: Optional[str] = None,
        importance_below: Optional[float] = None,
    ) -> int:
        """Delete memories matching criteria. Returns count deleted."""
        async with self._get_session() as session:
            conditions = [MemoryRow.agent_id == agent_id]
            if older_than:
                cutoff = datetime.utcnow() - _parse_duration(older_than)
                conditions.append(MemoryRow.created_at < cutoff)
            if importance_below is not None:
                conditions.append(MemoryRow.importance < importance_below)
            stmt = delete(MemoryRow).where(and_(*conditions))
            result = await session.execute(stmt)
            await session.commit()
            count = result.rowcount
            logger.info("Deleted %d memories for agent %s", count, agent_id)
            return count

    async def count_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Count memories matching criteria."""
        async with self._get_session() as session:
            from sqlalchemy import func

            stmt = select(func.count()).select_from(MemoryRow).where(
                MemoryRow.agent_id == agent_id
            )
            if memory_type is not None:
                stmt = stmt.where(MemoryRow.type == memory_type.value)
            result = await session.execute(stmt)
            return result.scalar() or 0

    async def query_memories(self, query: MemoryQuery) -> list[Memory]:
        """Query memories using structured filters from a MemoryQuery."""
        async with self._get_session() as session:
            conditions = [MemoryRow.agent_id == query.agent_id]
            if query.types:
                type_values = [t.value for t in query.types]
                conditions.append(MemoryRow.type.in_(type_values))
            conditions.append(MemoryRow.importance >= query.min_importance)
            if query.time_range:
                start, end = query.time_range
                conditions.append(MemoryRow.created_at >= start)
                conditions.append(MemoryRow.created_at <= end)
            stmt = (
                select(MemoryRow)
                .where(and_(*conditions))
                .order_by(MemoryRow.importance.desc())
                .limit(query.top_k * 3)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            memories = [_row_to_memory(row) for row in rows]
            if query.tags:
                memories = [
                    m for m in memories if any(t in m.tags for t in query.tags)
                ]
            return memories

    async def get_all_memories(self, agent_id: str) -> list[Memory]:
        """Retrieve all memories for an agent."""
        async with self._get_session() as session:
            stmt = select(MemoryRow).where(MemoryRow.agent_id == agent_id)
            result = await session.execute(stmt)
            return [_row_to_memory(row) for row in result.scalars().all()]

    async def list_agents(self) -> list[dict]:
        """List all distinct agent IDs with memory counts."""
        async with self._get_session() as session:
            from sqlalchemy import func

            stmt = (
                select(
                    MemoryRow.agent_id,
                    func.count().label("memory_count"),
                )
                .group_by(MemoryRow.agent_id)
                .order_by(func.count().desc())
            )
            result = await session.execute(stmt)
            return [
                {"agent_id": row.agent_id, "memory_count": row.memory_count}
                for row in result.all()
            ]

    async def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            await self._engine.dispose()
        logger.info("SQLite store closed")
