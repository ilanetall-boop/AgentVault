"""Semantic memory — stores facts, preferences, knowledge."""

from __future__ import annotations

import logging
from typing import Any

from agentvault.core.types import SemanticMemory

logger = logging.getLogger(__name__)


class SemanticMemoryManager:
    """Manages semantic memories — 'what I know'.

    Semantic memories capture facts, preferences, and extracted knowledge.
    They can be structured as subject-predicate-object triples for
    lightweight knowledge graph operations.
    """

    def create_memory(
        self,
        agent_id: str,
        content: str,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> SemanticMemory:
        """Create a new semantic memory.

        Args:
            agent_id: The agent creating this memory.
            content: The text content of the fact/knowledge.
            subject: Subject of the fact (e.g., 'user').
            predicate: Relationship (e.g., 'prefers').
            obj: Object of the fact (e.g., 'TypeScript').
            confidence: How confident we are in this fact (0-1).
            importance: Importance score (0-1).
            tags: Tags for categorization.
            source: What created this memory.

        Returns:
            A new SemanticMemory instance.
        """
        memory = SemanticMemory(
            agent_id=agent_id,
            content=content,
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=confidence,
            importance=importance,
            tags=tags or [],
            source=source,
        )
        logger.debug("Created semantic memory %s for agent %s", memory.id, agent_id)
        return memory

    def create_from_triple(
        self,
        agent_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> SemanticMemory:
        """Create a semantic memory from a knowledge triple.

        Automatically generates content from the triple.

        Args:
            agent_id: The agent creating this memory.
            subject: Subject entity.
            predicate: Relationship.
            obj: Object entity.
            confidence: Confidence in this fact.
            importance: Importance score.
            tags: Tags for categorization.

        Returns:
            A new SemanticMemory instance.
        """
        content = f"{subject} {predicate} {obj}"
        return self.create_memory(
            agent_id=agent_id,
            content=content,
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=confidence,
            importance=importance,
            tags=tags,
        )

    def merge_memories(
        self, existing: SemanticMemory, new: SemanticMemory
    ) -> SemanticMemory:
        """Merge two similar semantic memories into one.

        Keeps the most recent content, combines tags, and updates confidence.
        The confidence is boosted when the same fact is seen multiple times.

        Args:
            existing: The existing memory.
            new: The new memory with similar content.

        Returns:
            A merged SemanticMemory instance.
        """
        merged_tags = list(set(existing.tags + new.tags))
        merged_confidence = min(1.0, existing.confidence + 0.1)
        merged_importance = max(existing.importance, new.importance)

        merged = SemanticMemory(
            id=existing.id,
            agent_id=existing.agent_id,
            content=new.content if len(new.content) > len(existing.content) else existing.content,
            subject=new.subject or existing.subject,
            predicate=new.predicate or existing.predicate,
            obj=new.obj or existing.obj,
            confidence=merged_confidence,
            importance=merged_importance,
            tags=merged_tags,
            source=new.source or existing.source,
            access_count=existing.access_count + new.access_count,
            created_at=existing.created_at,
            related_memories=list(set(existing.related_memories + new.related_memories)),
        )
        logger.debug("Merged semantic memory %s", merged.id)
        return merged

    def is_contradictory(self, memory_a: SemanticMemory, memory_b: SemanticMemory) -> bool:
        """Check if two semantic memories contradict each other.

        Two memories contradict if they share the same subject and predicate
        but have different objects.

        Args:
            memory_a: First memory.
            memory_b: Second memory.

        Returns:
            True if the memories are contradictory.
        """
        if not (memory_a.subject and memory_a.predicate and memory_a.obj):
            return False
        if not (memory_b.subject and memory_b.predicate and memory_b.obj):
            return False
        return (
            memory_a.subject == memory_b.subject
            and memory_a.predicate == memory_b.predicate
            and memory_a.obj != memory_b.obj
        )

    def resolve_contradiction(
        self, memory_a: SemanticMemory, memory_b: SemanticMemory
    ) -> SemanticMemory:
        """Resolve a contradiction by keeping the more confident/recent one.

        Args:
            memory_a: First memory.
            memory_b: Second memory.

        Returns:
            The winning memory with updated metadata.
        """
        if memory_b.confidence > memory_a.confidence or memory_b.created_at > memory_a.created_at:
            winner = memory_b
        else:
            winner = memory_a
        logger.info(
            "Resolved contradiction: kept memory %s (confidence=%.2f)",
            winner.id,
            winner.confidence,
        )
        return winner

    def extract_knowledge(self, memory: SemanticMemory) -> dict[str, Any]:
        """Extract structured knowledge from a semantic memory.

        Returns:
            Dict with subject, predicate, object, confidence, and content.
        """
        return {
            "subject": memory.subject,
            "predicate": memory.predicate,
            "object": memory.obj,
            "confidence": memory.confidence,
            "content": memory.content,
            "tags": memory.tags,
        }
