"""Procedural memory — stores skills, workflows, learned procedures."""

from __future__ import annotations

import logging
from typing import Any

from agentvault.core.types import ProceduralMemory

logger = logging.getLogger(__name__)


class ProceduralMemoryManager:
    """Manages procedural memories — 'how to do things'.

    Procedural memories capture skills, workflows, and step-by-step
    procedures learned from experience. They include success/failure
    tracking to learn from mistakes.
    """

    def create_memory(
        self,
        agent_id: str,
        content: str,
        name: str | None = None,
        steps: list[str] | None = None,
        learned_from: str | None = None,
        importance: float = 0.5,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> ProceduralMemory:
        """Create a new procedural memory.

        Args:
            agent_id: The agent creating this memory.
            content: Text description of the procedure.
            name: Short name for the procedure.
            steps: Ordered list of steps.
            learned_from: Source conversation/interaction ID.
            importance: Importance score (0-1).
            tags: Tags for categorization.
            source: What created this memory.

        Returns:
            A new ProceduralMemory instance.
        """
        memory = ProceduralMemory(
            agent_id=agent_id,
            content=content,
            name=name or content[:50],
            steps=steps or [],
            learned_from=learned_from,
            importance=importance,
            tags=tags or [],
            source=source,
        )
        logger.debug("Created procedural memory %s for agent %s", memory.id, agent_id)
        return memory

    def record_success(self, memory: ProceduralMemory) -> ProceduralMemory:
        """Record a successful execution of this procedure.

        Increases success count and boosts importance.

        Args:
            memory: The procedural memory that succeeded.

        Returns:
            Updated ProceduralMemory.
        """
        memory.success_count += 1
        memory.importance = min(1.0, memory.importance + 0.05)
        logger.debug(
            "Recorded success for procedure %s (total: %d)",
            memory.id,
            memory.success_count,
        )
        return memory

    def record_failure(self, memory: ProceduralMemory) -> ProceduralMemory:
        """Record a failed execution of this procedure.

        Increases failure count and may decrease importance.

        Args:
            memory: The procedural memory that failed.

        Returns:
            Updated ProceduralMemory.
        """
        memory.failure_count += 1
        if memory.failure_count > memory.success_count:
            memory.importance = max(0.0, memory.importance - 0.1)
        logger.debug(
            "Recorded failure for procedure %s (total: %d)",
            memory.id,
            memory.failure_count,
        )
        return memory

    def get_success_rate(self, memory: ProceduralMemory) -> float:
        """Calculate the success rate of a procedure.

        Args:
            memory: The procedural memory to evaluate.

        Returns:
            Success rate between 0 and 1, or 0 if never executed.
        """
        total = memory.success_count + memory.failure_count
        if total == 0:
            return 0.0
        return memory.success_count / total

    def merge_procedures(
        self,
        existing: ProceduralMemory,
        new: ProceduralMemory,
    ) -> ProceduralMemory:
        """Merge two similar procedures into one refined version.

        Keeps the most complete steps list and combines metrics.

        Args:
            existing: The existing procedure.
            new: The new procedure with similar intent.

        Returns:
            A merged ProceduralMemory.
        """
        merged_steps = new.steps if len(new.steps) > len(existing.steps) else existing.steps
        merged_tags = list(set(existing.tags + new.tags))

        merged = ProceduralMemory(
            id=existing.id,
            agent_id=existing.agent_id,
            content=new.content if len(new.content) > len(existing.content) else existing.content,
            name=new.name or existing.name,
            steps=merged_steps,
            learned_from=new.learned_from or existing.learned_from,
            importance=max(existing.importance, new.importance),
            tags=merged_tags,
            source=new.source or existing.source,
            success_count=existing.success_count + new.success_count,
            failure_count=existing.failure_count + new.failure_count,
            access_count=existing.access_count + new.access_count,
            created_at=existing.created_at,
            related_memories=list(set(existing.related_memories + new.related_memories)),
        )
        logger.debug("Merged procedural memory %s", merged.id)
        return merged

    def extract_procedure_info(self, memory: ProceduralMemory) -> dict[str, Any]:
        """Extract structured information from a procedural memory.

        Returns:
            Dict with name, steps, success rate, and metadata.
        """
        return {
            "name": memory.name,
            "steps": memory.steps,
            "success_rate": self.get_success_rate(memory),
            "success_count": memory.success_count,
            "failure_count": memory.failure_count,
            "learned_from": memory.learned_from,
            "content": memory.content,
        }
