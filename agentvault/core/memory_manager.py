"""Memory Manager — the central orchestrator for AgentVault.
# TODO: Add support for memory priority queues in v0.2.0
# Performance optimization: cache frequently accessed memories
Routes memories to the correct type, calculates importance,
manages consolidation, and handles intelligent recall.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from agentvault.core.episodic import EpisodicMemoryManager
from agentvault.core.procedural import ProceduralMemoryManager
from agentvault.core.semantic import SemanticMemoryManager
from agentvault.core.types import (
    ConsolidationResult,
    Memory,
    MemoryQuery,
    MemoryType,
    RecallResult,
)
from agentvault.processing.consolidator import Consolidator
from agentvault.processing.extractor import Extractor
from agentvault.processing.indexer import Indexer
from agentvault.processing.ranker import Ranker
from agentvault.storage.hybrid_store import HybridStore

logger = logging.getLogger(__name__)

# Default number of memories before auto-consolidation triggers
DEFAULT_CONSOLIDATION_THRESHOLD = 500


class MemoryManager:
    """The central orchestrator that coordinates all memory operations.

    This is the 'brain' of AgentVault. It:
    - Routes memories to the correct type (episodic/semantic/procedural)
    - Calculates importance automatically via the Extractor
    - Generates embeddings via the Indexer
    - Manages hybrid recall (vector + structured)
    - Triggers consolidation when thresholds are reached
    """

    def __init__(
        self,
        store: Optional[HybridStore] = None,
        indexer: Optional[Indexer] = None,
        ranker: Optional[Ranker] = None,
        consolidator: Optional[Consolidator] = None,
        extractor: Optional[Extractor] = None,
        db_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        chroma_path: Optional[str] = None,  # Backwards compatibility alias
        consolidation_threshold: int = DEFAULT_CONSOLIDATION_THRESHOLD,
    ) -> None:
        self._indexer = indexer or Indexer()
        self._ranker = ranker or Ranker()
        self._extractor = extractor or Extractor()
        vector_path = faiss_path or chroma_path
        self._store = store or HybridStore(db_path=db_path, faiss_path=vector_path)
        self._consolidator = consolidator or Consolidator(indexer=self._indexer)

        self._episodic = EpisodicMemoryManager()
        self._semantic = SemanticMemoryManager()
        self._procedural = ProceduralMemoryManager()

        self._consolidation_threshold = consolidation_threshold
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components (store, indexer)."""
        await self._store.initialize()
        await self._indexer.initialize()
        self._initialized = True
        logger.info("MemoryManager initialized")

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized. Call initialize() first.")

    async def remember(
        self,
        agent_id: str,
        content: str,
        memory_type: Optional[MemoryType | str] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Memory:
        """Store a new memory. The primary write interface.

        Automatically:
        - Classifies memory type if not provided
        - Estimates importance if not provided
        - Extracts tags if not provided
        - Generates embedding
        - Stores in both vector and structured stores
        - Triggers consolidation if threshold is reached

        Args:
            agent_id: The agent creating this memory.
            content: The text content to remember.
            memory_type: Type of memory (auto-detected if None).
            importance: Importance score (auto-estimated if None).
            tags: Tags for categorization (auto-extracted if None).
            metadata: Additional metadata.
            source: What created this memory.

        Returns:
            The stored Memory object.
        """
        self._ensure_initialized()

        # Auto-extract info
        extraction = self._extractor.auto_extract(content)

        # Resolve memory type
        if memory_type is None:
            resolved_type = extraction["memory_type"]
        elif isinstance(memory_type, str):
            resolved_type = MemoryType(memory_type)
        else:
            resolved_type = memory_type

        # Resolve importance
        resolved_importance = importance if importance is not None else extraction["importance"]

        # Resolve tags
        resolved_tags = tags if tags is not None else extraction["tags"]

        # Create typed memory
        memory = self._create_typed_memory(
            agent_id=agent_id,
            content=content,
            memory_type=resolved_type,
            importance=resolved_importance,
            tags=resolved_tags,
            metadata=metadata or {},
            source=source,
            extraction=extraction,
        )

        # Generate embedding
        memory = await self._indexer.index_memory(memory)

        # Store
        await self._store.store(memory)

        # Check consolidation threshold
        count = await self._store.count(agent_id)
        if count > 0 and count % self._consolidation_threshold == 0:
            logger.info(
                "Consolidation threshold reached (%d memories), running consolidation",
                count,
            )
            await self.consolidate(agent_id)

        logger.debug(
            "Remembered: type=%s importance=%.2f tags=%s id=%s",
            resolved_type.value,
            resolved_importance,
            resolved_tags,
            memory.id,
        )
        return memory

    async def remember_episode(
        self,
        agent_id: str,
        event: str,
        context: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
    ) -> Memory:
        """Convenience method to store an episodic memory.

        Args:
            agent_id: The agent creating this memory.
            event: Description of what happened.
            context: Contextual information (file, action, etc.).
            importance: Importance score.
            tags: Tags for categorization.
            source: What created this memory.

        Returns:
            The stored Memory.
        """
        return await self.remember(
            agent_id=agent_id,
            content=event,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            tags=tags,
            metadata=context or {},
            source=source,
        )

    async def remember_fact(
        self,
        agent_id: str,
        content: str,
        importance: float = 0.6,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
    ) -> Memory:
        """Convenience method to store a semantic (fact) memory.

        Args:
            agent_id: The agent creating this memory.
            content: The fact to remember.
            importance: Importance score.
            tags: Tags for categorization.
            source: What created this memory.

        Returns:
            The stored Memory.
        """
        return await self.remember(
            agent_id=agent_id,
            content=content,
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            tags=tags,
            source=source,
        )

    async def remember_procedure(
        self,
        agent_id: str,
        name: str,
        steps: list[str],
        content: Optional[str] = None,
        learned_from: Optional[str] = None,
        importance: float = 0.6,
        tags: Optional[list[str]] = None,
    ) -> Memory:
        """Convenience method to store a procedural memory.

        Args:
            agent_id: The agent creating this memory.
            name: Name of the procedure.
            steps: Ordered list of steps.
            content: Optional text description.
            learned_from: Source conversation/interaction ID.
            importance: Importance score.
            tags: Tags for categorization.

        Returns:
            The stored Memory.
        """
        full_content = content or f"{name}: {' → '.join(steps)}"
        return await self.remember(
            agent_id=agent_id,
            content=full_content,
            memory_type=MemoryType.PROCEDURAL,
            importance=importance,
            tags=tags,
            metadata={"name": name, "steps": steps, "learned_from": learned_from},
            source=learned_from,
        )

    async def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        types: Optional[list[MemoryType]] = None,
        min_importance: float = 0.0,
        tags: Optional[list[str]] = None,
        include_shared: bool = True,
    ) -> RecallResult:
        """Recall relevant memories. The primary read interface.

        Combines vector similarity search with structured filters
        for the most relevant results. Reinforces accessed memories.

        Args:
            agent_id: The agent recalling memories.
            query: The search query text.
            top_k: Maximum number of results.
            types: Filter by memory types.
            min_importance: Minimum importance threshold.
            tags: Filter by tags.
            include_shared: Include shared multi-agent memories.

        Returns:
            RecallResult with ranked memories and scores.
        """
        self._ensure_initialized()

        query_embedding = await self._indexer.embed_text(query)

        memory_query = MemoryQuery(
            query=query,
            agent_id=agent_id,
            types=types or [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
            top_k=top_k,
            min_importance=min_importance,
            tags=tags or [],
            include_shared=include_shared,
        )

        result = await self._store.recall(memory_query, query_embedding)

        # Re-rank using the full ranker
        if result.memories:
            ranked = self._ranker.rank(
                memories=result.memories,
                similarities=result.relevance_scores,
                top_k=top_k,
            )
            result.memories = [m for m, _ in ranked]
            result.relevance_scores = [s for _, s in ranked]

        # Reinforce accessed memories
        for memory in result.memories:
            reinforced = self._consolidator.reinforce_single(memory)
            await self._store.update(reinforced)

        return result

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID.

        Args:
            memory_id: The memory's unique ID.

        Returns:
            The Memory if found, else None.
        """
        self._ensure_initialized()
        return await self._store.get_memory(memory_id)

    async def forget(
        self,
        agent_id: str,
        older_than: Optional[str] = None,
        importance_below: Optional[float] = None,
    ) -> int:
        """Forget (delete) memories matching criteria.

        Args:
            agent_id: The agent whose memories to forget.
            older_than: Duration string like '30d', '12h'.
            importance_below: Delete memories below this importance.

        Returns:
            Number of memories forgotten.
        """
        self._ensure_initialized()
        count = await self._store.forget(
            agent_id=agent_id,
            older_than=older_than,
            importance_below=importance_below,
        )
        logger.info("Forgot %d memories for agent %s", count, agent_id)
        return count

    async def consolidate(self, agent_id: str) -> ConsolidationResult:
        """Run memory consolidation for an agent.

        Merges similar memories, promotes frequent episodic to semantic,
        forgets old unimportant memories, and reinforces accessed ones.

        Args:
            agent_id: The agent whose memories to consolidate.

        Returns:
            ConsolidationResult with statistics.
        """
        self._ensure_initialized()
        memories = await self._store.get_all_memories(agent_id)
        if not memories:
            return ConsolidationResult()

        consolidated, result = await self._consolidator.consolidate(memories)

        # Update store with consolidated memories
        existing_ids = {m.id for m in memories}
        consolidated_ids = {m.id for m in consolidated}

        # Delete forgotten memories
        for mid in existing_ids - consolidated_ids:
            await self._store.delete(mid)

        # Update remaining memories
        for memory in consolidated:
            await self._store.update(memory)

        return result

    async def count(
        self,
        agent_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> int:
        """Count memories for an agent.

        Args:
            agent_id: The agent to count for.
            memory_type: Optional type filter.

        Returns:
            Number of memories.
        """
        self._ensure_initialized()
        return await self._store.count(agent_id, memory_type)

    async def close(self) -> None:
        """Close all resources."""
        await self._store.close()
        logger.info("MemoryManager closed")

    def _create_typed_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: MemoryType,
        importance: float,
        tags: list[str],
        metadata: dict[str, Any],
        source: Optional[str],
        extraction: dict[str, Any],
    ) -> Memory:
        """Create a memory of the appropriate type."""
        if memory_type == MemoryType.EPISODIC:
            return self._episodic.create_memory(
                agent_id=agent_id,
                content=content,
                importance=importance,
                tags=tags,
                source=source,
                context=metadata,
            )
        if memory_type == MemoryType.SEMANTIC:
            facts = extraction.get("facts", [])
            first_fact = facts[0] if facts else {}
            return self._semantic.create_memory(
                agent_id=agent_id,
                content=content,
                subject=first_fact.get("subject"),
                predicate=first_fact.get("predicate"),
                obj=first_fact.get("object"),
                importance=importance,
                tags=tags,
                source=source,
            )
        if memory_type == MemoryType.PROCEDURAL:
            steps = extraction.get("steps", metadata.get("steps", []))
            return self._procedural.create_memory(
                agent_id=agent_id,
                content=content,
                name=metadata.get("name"),
                steps=steps,
                learned_from=source,
                importance=importance,
                tags=tags,
                source=source,
            )
        return Memory(
            agent_id=agent_id,
            type=memory_type,
            content=content,
            importance=importance,
            tags=tags,
            metadata=metadata,
            source=source,
        )
