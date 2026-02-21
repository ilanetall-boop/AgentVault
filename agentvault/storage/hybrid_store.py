"""Hybrid store combining vector and structured storage."""

from __future__ import annotations

import logging
import time
from typing import Optional

from agentvault.core.types import Memory, MemoryQuery, MemoryType, RecallResult
from agentvault.storage.structured_store import SQLiteStructuredStore
from agentvault.storage.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class HybridStore:
    """Combines vector store (semantic search) with structured store (CRUD + filters).

    This is the main storage interface used by the Memory Manager.
    It routes operations to the appropriate backend and combines results
    for hybrid search (vector similarity + structured filters).
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        structured_store: Optional[SQLiteStructuredStore] = None,
        db_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        # Backwards compatibility alias
        chroma_path: Optional[str] = None,
    ) -> None:
        persist_path = faiss_path or chroma_path
        self.vector_store = vector_store or FAISSVectorStore(persist_directory=persist_path)
        self.structured_store = structured_store or SQLiteStructuredStore(db_path=db_path)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize both storage backends."""
        await self.structured_store.initialize()
        await self.vector_store.initialize()
        self._initialized = True
        logger.info("Hybrid store initialized")

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized:
            raise RuntimeError("Hybrid store not initialized. Call initialize() first.")

    async def store(self, memory: Memory) -> None:
        """Store a memory in both backends.

        Saves the full memory to structured store and the embedding
        to the vector store for similarity search.
        """
        self._ensure_initialized()
        await self.structured_store.save_memory(memory)
        if memory.embedding:
            metadata = {
                "agent_id": memory.agent_id,
                "type": memory.type.value,
                "importance": memory.importance,
                "tags": ",".join(memory.tags) if memory.tags else "",
            }
            await self.vector_store.add(memory.id, memory.embedding, metadata)
        logger.debug("Stored memory %s in hybrid store", memory.id)

    async def recall(
        self,
        query: MemoryQuery,
        query_embedding: Optional[list[float]] = None,
    ) -> RecallResult:
        """Recall memories using hybrid search (vector + structured).

        Combines vector similarity results with structured query filters
        for the most relevant results.
        """
        self._ensure_initialized()
        start_time = time.perf_counter()

        vector_results: dict[str, float] = {}
        if query_embedding is not None:
            filters: dict = {"agent_id": query.agent_id}
            if len(query.types) == 1:
                filters["type"] = query.types[0].value
            raw_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=query.top_k * 2,
                filters=filters,
            )
            vector_results = {mid: score for mid, score in raw_results}

        structured_memories = await self.structured_store.query_memories(query)

        scored_memories: list[tuple[Memory, float]] = []
        seen_ids: set[str] = set()

        for memory in structured_memories:
            if memory.id in seen_ids:
                continue
            seen_ids.add(memory.id)
            vector_score = vector_results.get(memory.id, 0.0)
            struct_score = memory.importance
            combined_score = 0.6 * vector_score + 0.4 * struct_score
            scored_memories.append((memory, combined_score))

        for memory_id, vector_score in vector_results.items():
            if memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            memory = await self.structured_store.get_memory(memory_id)
            if memory is None:
                continue
            if memory.importance < query.min_importance:
                continue
            if query.types and memory.type not in query.types:
                continue
            if query.tags and not any(t in memory.tags for t in query.tags):
                continue
            combined_score = 0.6 * vector_score + 0.4 * memory.importance
            scored_memories.append((memory, combined_score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_memories[: query.top_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        memories = [m for m, _ in top_results]
        scores = [s for _, s in top_results]

        return RecallResult(
            memories=memories,
            relevance_scores=scores,
            total_found=len(scored_memories),
            query_time_ms=round(elapsed_ms, 2),
        )

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        self._ensure_initialized()
        return await self.structured_store.get_memory(memory_id)

    async def get_memories(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[Memory]:
        """Retrieve memories with filters."""
        self._ensure_initialized()
        return await self.structured_store.get_memories(
            agent_id=agent_id,
            memory_type=memory_type,
            tags=tags,
            min_importance=min_importance,
            limit=limit,
        )

    async def update(self, memory: Memory) -> None:
        """Update a memory in both backends."""
        self._ensure_initialized()
        await self.structured_store.update_memory(memory)
        if memory.embedding:
            metadata = {
                "agent_id": memory.agent_id,
                "type": memory.type.value,
                "importance": memory.importance,
                "tags": ",".join(memory.tags) if memory.tags else "",
            }
            await self.vector_store.add(memory.id, memory.embedding, metadata)

    async def delete(self, memory_id: str) -> None:
        """Delete a memory from both backends."""
        self._ensure_initialized()
        await self.structured_store.delete_memory(memory_id)
        await self.vector_store.delete(memory_id)

    async def forget(
        self,
        agent_id: str,
        older_than: Optional[str] = None,
        importance_below: Optional[float] = None,
    ) -> int:
        """Delete memories matching forget criteria."""
        self._ensure_initialized()
        memories = await self.structured_store.get_memories(
            agent_id=agent_id, min_importance=0.0, limit=10000
        )
        to_delete = []
        for m in memories:
            if importance_below is not None and m.importance >= importance_below:
                continue
            to_delete.append(m.id)
        count = await self.structured_store.delete_memories(
            agent_id=agent_id,
            older_than=older_than,
            importance_below=importance_below,
        )
        for mid in to_delete:
            await self.vector_store.delete(mid)
        return count

    async def get_all_memories(self, agent_id: str) -> list[Memory]:
        """Retrieve all memories for an agent."""
        self._ensure_initialized()
        return await self.structured_store.get_all_memories(agent_id)

    async def count(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Count memories."""
        self._ensure_initialized()
        return await self.structured_store.count_memories(agent_id, memory_type)

    async def close(self) -> None:
        """Close both storage backends."""
        await self.vector_store.close()
        await self.structured_store.close()
        logger.info("Hybrid store closed")
