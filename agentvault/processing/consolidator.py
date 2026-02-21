"""Consolidator — merges, promotes, forgets, and reinforces memories.

This is the CORE INNOVATION of AgentVault. It simulates how the human
brain consolidates memories over time:
- Fusion: 2 similar memories → 1 enriched memory
- Promotion: frequent episodic → semantic (fact)
- Forgetting: unused + low importance → deleted
- Reinforcement: each access increases importance
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from agentvault.core.episodic import EpisodicMemoryManager
from agentvault.core.procedural import ProceduralMemoryManager
from agentvault.core.semantic import SemanticMemoryManager
from agentvault.core.types import (
    ConsolidationResult,
    Memory,
    MemoryType,
    SemanticMemory,
)
from agentvault.processing.indexer import Indexer

logger = logging.getLogger(__name__)


class Consolidator:
    """Consolidates memories by merging, promoting, forgetting, and reinforcing.

    This process is analogous to how the human brain processes memories
    during sleep — strengthening important memories and letting go of
    unimportant ones.
    """

    def __init__(
        self,
        indexer: Indexer,
        similarity_threshold: float = 0.85,
        promotion_access_threshold: int = 5,
        forget_max_age_days: int = 90,
        forget_importance_threshold: float = 0.2,
        reinforcement_boost: float = 0.05,
    ) -> None:
        """Initialize the consolidator.

        Args:
            indexer: The indexer for computing similarity.
            similarity_threshold: Minimum similarity to merge two memories.
            promotion_access_threshold: Accesses needed to promote episodic → semantic.
            forget_max_age_days: Max age before low-importance memories are forgotten.
            forget_importance_threshold: Importance below which memories may be forgotten.
            reinforcement_boost: How much importance increases per access.
        """
        self.indexer = indexer
        self.similarity_threshold = similarity_threshold
        self.promotion_access_threshold = promotion_access_threshold
        self.forget_max_age_days = forget_max_age_days
        self.forget_importance_threshold = forget_importance_threshold
        self.reinforcement_boost = reinforcement_boost

        self._episodic_mgr = EpisodicMemoryManager()
        self._semantic_mgr = SemanticMemoryManager()
        self._procedural_mgr = ProceduralMemoryManager()

    async def consolidate(
        self,
        memories: list[Memory],
    ) -> tuple[list[Memory], ConsolidationResult]:
        """Run full consolidation on a list of memories.

        Performs all consolidation operations in sequence:
        1. Reinforce accessed memories
        2. Merge similar memories
        3. Promote frequent episodic → semantic
        4. Forget old, unimportant memories

        Args:
            memories: The list of memories to consolidate.

        Returns:
            Tuple of (consolidated memories list, consolidation result stats).
        """
        start_time = time.perf_counter()
        result = ConsolidationResult(total_processed=len(memories))

        memories, reinforced = self._reinforce(memories)
        result.reinforced = reinforced

        memories, merged = await self._merge_similar(memories)
        result.merged = merged

        memories, promoted = self._promote(memories)
        result.promoted = promoted

        memories, forgotten = self._forget(memories)
        result.forgotten = forgotten

        result.duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Consolidation complete: merged=%d, promoted=%d, forgotten=%d, "
            "reinforced=%d, duration=%.2fms",
            result.merged,
            result.promoted,
            result.forgotten,
            result.reinforced,
            result.duration_ms,
        )
        return memories, result

    def _reinforce(self, memories: list[Memory]) -> tuple[list[Memory], int]:
        """Reinforce memories that have been accessed recently.

        Each access boosts importance slightly, simulating how
        frequently recalled memories become stronger.
        """
        reinforced_count = 0
        for memory in memories:
            if memory.access_count > 0:
                old_importance = memory.importance
                memory.importance = min(
                    1.0, memory.importance + self.reinforcement_boost * memory.access_count
                )
                if memory.importance != old_importance:
                    reinforced_count += 1
        return memories, reinforced_count

    async def _merge_similar(
        self, memories: list[Memory]
    ) -> tuple[list[Memory], int]:
        """Merge memories that are semantically very similar.

        Uses cosine similarity between embeddings to find near-duplicates
        and merges them into a single enriched memory.
        """
        if len(memories) < 2:
            return memories, 0

        merged_count = 0
        merged_ids: set[str] = set()
        result: list[Memory] = []

        for i, mem_a in enumerate(memories):
            if mem_a.id in merged_ids:
                continue
            if mem_a.embedding is None:
                result.append(mem_a)
                continue

            best_match: Memory | None = None
            best_similarity = 0.0

            for j in range(i + 1, len(memories)):
                mem_b = memories[j]
                if mem_b.id in merged_ids or mem_b.embedding is None:
                    continue
                if mem_a.type != mem_b.type:
                    continue

                similarity = self.indexer.cosine_similarity(
                    mem_a.embedding, mem_b.embedding
                )
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = mem_b

            if best_match is not None:
                merged = self._merge_two(mem_a, best_match)
                result.append(merged)
                merged_ids.add(best_match.id)
                merged_count += 1
                logger.debug(
                    "Merged memories %s and %s (similarity=%.3f)",
                    mem_a.id,
                    best_match.id,
                    best_similarity,
                )
            else:
                result.append(mem_a)

        return result, merged_count

    def _merge_two(self, mem_a: Memory, mem_b: Memory) -> Memory:
        """Merge two memories based on their type."""
        if mem_a.type == MemoryType.SEMANTIC:
            a_sem = SemanticMemory(**mem_a.model_dump())
            b_sem = SemanticMemory(**mem_b.model_dump())
            return self._semantic_mgr.merge_memories(a_sem, b_sem)
        if mem_a.type == MemoryType.PROCEDURAL:
            from agentvault.core.types import ProceduralMemory

            a_proc = ProceduralMemory(**mem_a.model_dump())
            b_proc = ProceduralMemory(**mem_b.model_dump())
            return self._procedural_mgr.merge_procedures(a_proc, b_proc)

        # For episodic, keep the more important one with combined metadata
        winner = mem_a if mem_a.importance >= mem_b.importance else mem_b
        winner.tags = list(set(mem_a.tags + mem_b.tags))
        winner.access_count = mem_a.access_count + mem_b.access_count
        winner.related_memories = list(
            set(mem_a.related_memories + mem_b.related_memories)
        )
        return winner

    def _promote(self, memories: list[Memory]) -> tuple[list[Memory], int]:
        """Promote frequently accessed episodic memories to semantic.

        When an event is recalled many times, it becomes general knowledge.
        """
        promoted_count = 0
        result: list[Memory] = []

        for memory in memories:
            if self._episodic_mgr.should_promote_to_semantic(
                memory, self.promotion_access_threshold
            ):
                promoted = Memory(
                    id=memory.id,
                    agent_id=memory.agent_id,
                    type=MemoryType.SEMANTIC,
                    content=memory.content,
                    metadata={**memory.metadata, "promoted_from": "episodic"},
                    embedding=memory.embedding,
                    importance=min(1.0, memory.importance + 0.1),
                    access_count=memory.access_count,
                    created_at=memory.created_at,
                    last_accessed=memory.last_accessed,
                    tags=memory.tags + ["promoted"],
                    source=memory.source,
                    related_memories=memory.related_memories,
                )
                result.append(promoted)
                promoted_count += 1
                logger.debug(
                    "Promoted episodic memory %s to semantic (accesses=%d)",
                    memory.id,
                    memory.access_count,
                )
            else:
                result.append(memory)

        return result, promoted_count

    def _forget(self, memories: list[Memory]) -> tuple[list[Memory], int]:
        """Forget old, unimportant, rarely accessed memories.

        Simulates natural memory decay — memories that are never recalled
        and have low importance eventually fade.
        """
        forgotten_count = 0
        result: list[Memory] = []
        cutoff = datetime.utcnow() - timedelta(days=self.forget_max_age_days)

        for memory in memories:
            should_forget = (
                memory.importance < self.forget_importance_threshold
                and memory.access_count == 0
                and memory.created_at < cutoff
            )
            if memory.expires_at and memory.expires_at < datetime.utcnow():
                should_forget = True

            if should_forget:
                forgotten_count += 1
                logger.debug(
                    "Forgetting memory %s (importance=%.2f, accesses=%d, age=%s)",
                    memory.id,
                    memory.importance,
                    memory.access_count,
                    datetime.utcnow() - memory.created_at,
                )
            else:
                result.append(memory)

        return result, forgotten_count

    def reinforce_single(self, memory: Memory) -> Memory:
        """Reinforce a single memory (called on each access).

        Args:
            memory: The memory being accessed.

        Returns:
            The memory with updated access count and importance.
        """
        memory.access_count += 1
        memory.last_accessed = datetime.utcnow()
        memory.importance = min(1.0, memory.importance + self.reinforcement_boost)
        return memory
