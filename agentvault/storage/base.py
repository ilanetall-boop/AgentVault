"""Abstract base interfaces for storage backends."""

from __future__ import annotations

import abc
from typing import Optional

from agentvault.core.types import Memory, MemoryQuery, MemoryType


class BaseVectorStore(abc.ABC):
    """Abstract interface for vector similarity search backends."""

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store (create collections, etc.)."""

    @abc.abstractmethod
    async def add(self, memory_id: str, embedding: list[float], metadata: dict) -> None:
        """Add a vector embedding with metadata."""

    @abc.abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors. Returns list of (memory_id, score)."""

    @abc.abstractmethod
    async def delete(self, memory_id: str) -> None:
        """Delete a vector by memory ID."""

    @abc.abstractmethod
    async def update_metadata(self, memory_id: str, metadata: dict) -> None:
        """Update metadata for an existing vector."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the vector store connection."""


class BaseStructuredStore(abc.ABC):
    """Abstract interface for structured (relational) storage backends."""

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the store (create tables, etc.)."""

    @abc.abstractmethod
    async def save_memory(self, memory: Memory) -> None:
        """Save a memory to the store."""

    @abc.abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""

    @abc.abstractmethod
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

    @abc.abstractmethod
    async def update_memory(self, memory: Memory) -> None:
        """Update an existing memory."""

    @abc.abstractmethod
    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory by ID."""

    @abc.abstractmethod
    async def delete_memories(
        self,
        agent_id: str,
        older_than: Optional[str] = None,
        importance_below: Optional[float] = None,
    ) -> int:
        """Delete memories matching criteria. Returns count deleted."""

    @abc.abstractmethod
    async def count_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Count memories matching criteria."""

    @abc.abstractmethod
    async def query_memories(self, query: MemoryQuery) -> list[Memory]:
        """Query memories using structured filters from a MemoryQuery."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the store connection."""
