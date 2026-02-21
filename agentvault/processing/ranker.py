"""Ranker — scores and ranks memories by relevance."""

from __future__ import annotations

import logging
import math
from datetime import datetime

from agentvault.core.types import Memory

logger = logging.getLogger(__name__)


class Ranker:
    """Scores and ranks memories for recall relevance.

    Combines multiple signals (vector similarity, recency, importance,
    access frequency) into a final relevance score.
    """

    def __init__(
        self,
        similarity_weight: float = 0.4,
        recency_weight: float = 0.2,
        importance_weight: float = 0.25,
        frequency_weight: float = 0.15,
        recency_decay: float = 0.01,
    ) -> None:
        """Initialize the ranker with scoring weights.

        Args:
            similarity_weight: Weight for vector similarity score.
            recency_weight: Weight for recency score.
            importance_weight: Weight for importance score.
            frequency_weight: Weight for access frequency score.
            recency_decay: Decay rate for recency scoring (per hour).
        """
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.frequency_weight = frequency_weight
        self.recency_decay = recency_decay

    def score(
        self,
        memory: Memory,
        similarity: float = 0.0,
        now: datetime | None = None,
    ) -> float:
        """Calculate the combined relevance score for a memory.

        Args:
            memory: The memory to score.
            similarity: Vector similarity score (0-1).
            now: Current time (defaults to utcnow).

        Returns:
            Combined relevance score between 0 and 1.
        """
        now = now or datetime.utcnow()

        recency_score = self._recency_score(memory, now)
        frequency_score = self._frequency_score(memory)

        combined = (
            self.similarity_weight * similarity
            + self.recency_weight * recency_score
            + self.importance_weight * memory.importance
            + self.frequency_weight * frequency_score
        )
        return min(1.0, max(0.0, combined))

    def rank(
        self,
        memories: list[Memory],
        similarities: list[float],
        top_k: int = 10,
    ) -> list[tuple[Memory, float]]:
        """Rank a list of memories by combined relevance.

        Args:
            memories: List of candidate memories.
            similarities: Corresponding similarity scores.
            top_k: Number of top results to return.

        Returns:
            Sorted list of (memory, score) tuples, highest first.
        """
        if len(memories) != len(similarities):
            raise ValueError("memories and similarities must have same length")

        scored = [
            (memory, self.score(memory, sim))
            for memory, sim in zip(memories, similarities)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _recency_score(self, memory: Memory, now: datetime) -> float:
        """Calculate recency score with exponential decay.

        More recent memories score higher.
        """
        age_hours = (now - memory.last_accessed).total_seconds() / 3600
        return math.exp(-self.recency_decay * age_hours)

    def _frequency_score(self, memory: Memory, max_accesses: int = 50) -> float:
        """Calculate frequency score based on access count.

        Normalized logarithmic scale — diminishing returns for very
        frequently accessed memories.
        """
        if memory.access_count == 0:
            return 0.0
        return min(1.0, math.log(1 + memory.access_count) / math.log(1 + max_accesses))

    def adjust_weights(
        self,
        similarity_weight: float | None = None,
        recency_weight: float | None = None,
        importance_weight: float | None = None,
        frequency_weight: float | None = None,
    ) -> None:
        """Adjust scoring weights dynamically.

        Weights are automatically normalized to sum to 1.0.
        """
        if similarity_weight is not None:
            self.similarity_weight = similarity_weight
        if recency_weight is not None:
            self.recency_weight = recency_weight
        if importance_weight is not None:
            self.importance_weight = importance_weight
        if frequency_weight is not None:
            self.frequency_weight = frequency_weight

        total = (
            self.similarity_weight
            + self.recency_weight
            + self.importance_weight
            + self.frequency_weight
        )
        if total > 0:
            self.similarity_weight /= total
            self.recency_weight /= total
            self.importance_weight /= total
            self.frequency_weight /= total
        logger.debug(
            "Ranker weights adjusted: sim=%.2f rec=%.2f imp=%.2f freq=%.2f",
            self.similarity_weight,
            self.recency_weight,
            self.importance_weight,
            self.frequency_weight,
        )
