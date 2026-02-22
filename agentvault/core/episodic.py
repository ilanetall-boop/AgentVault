"""Episodic memory — stores events, interactions, conversations."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from agentvault.core.types import EpisodicMemory, Memory, MemoryType

logger = logging.getLogger(__name__)


class EpisodicMemoryManager:
    """Manages episodic memories — 'what happened'.

    Episodic memories capture events, interactions, and conversations.
    They are timestamped, contextual, and can be queried by time range.
    """

    def create_memory(
        self,
        agent_id: str,
        content: str,
        event: str | None = None,
        context: dict[str, Any] | None = None,
        importance: float = 0.5,
        tags: list[str] | None = None,
        source: str | None = None,
        participants: list[str] | None = None,
    ) -> EpisodicMemory:
        """Create a new episodic memory.

        Args:
            agent_id: The agent creating this memory.
            content: The text content of the memory.
            event: Short description of the event.
            context: Additional context dict (file, action, etc.).
            importance: Importance score between 0 and 1.
            tags: Tags for categorization.
            source: What created this memory.
            participants: Other agents/users involved.

        Returns:
            A new EpisodicMemory instance.
        """
        memory = EpisodicMemory(
            agent_id=agent_id,
            content=content,
            event=event or content[:100],
            context=context or {},
            importance=importance,
            tags=tags or [],
            source=source,
            participants=participants or [],
        )
        logger.debug("Created episodic memory %s for agent %s", memory.id, agent_id)
        return memory

    def should_promote_to_semantic(self, memory: Memory, access_threshold: int = 5) -> bool:
        """Check if an episodic memory should be promoted to semantic.

        An episodic memory that is accessed frequently becomes a 'fact'
        (semantic memory). This simulates how human brains consolidate
        repeated experiences into general knowledge.

        Args:
            memory: The memory to evaluate.
            access_threshold: Number of accesses to trigger promotion.

        Returns:
            True if the memory should be promoted.
        """
        if memory.type != MemoryType.EPISODIC:
            return False
        return memory.access_count >= access_threshold and memory.importance >= 0.6

    def extract_key_info(self, memory: EpisodicMemory) -> dict[str, Any]:
        """Extract key information from an episodic memory for indexing.

        Returns:
            Dict with extracted event, context, participants, and timestamp.
        """
        return {
            "event": memory.event,
            "context": memory.context,
            "participants": memory.participants,
            "timestamp": memory.created_at.isoformat(),
            "content_preview": memory.content[:200],
        }

    def calculate_recency_score(self, memory: Memory, decay_factor: float = 0.01) -> float:
        """Calculate a recency-weighted score for temporal relevance.

        More recent memories get higher scores, decaying over time.

        Args:
            memory: The memory to score.
            decay_factor: How fast the score decays (higher = faster decay).

        Returns:
            A score between 0 and 1.
        """
        age_hours = (datetime.utcnow() - memory.created_at).total_seconds() / 3600
        recency = 1.0 / (1.0 + decay_factor * age_hours)
        return recency * memory.importance
