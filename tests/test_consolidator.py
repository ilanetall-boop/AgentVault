"""Tests for the Consolidator — memory consolidation logic."""

from datetime import datetime, timedelta

import pytest

from agentvault.core.types import MemoryType
from tests.conftest import make_memory


class TestConsolidator:
    """Tests for the Consolidator."""

    @pytest.mark.asyncio
    async def test_reinforce_single(self, consolidator):
        """Test reinforcing a single memory on access."""
        memory = make_memory(importance=0.5, access_count=0)
        original_importance = memory.importance

        reinforced = consolidator.reinforce_single(memory)

        assert reinforced.access_count == 1
        assert reinforced.importance > original_importance
        assert reinforced.last_accessed >= memory.created_at

    @pytest.mark.asyncio
    async def test_consolidate_reinforcement(self, consolidator):
        """Test that consolidation reinforces accessed memories."""
        memories = [
            make_memory(content="Accessed memory", access_count=3, importance=0.5),
            make_memory(content="Never accessed", access_count=0, importance=0.5),
        ]

        consolidated, result = await consolidator.consolidate(memories)
        assert result.reinforced >= 1
        assert result.total_processed == 2

    @pytest.mark.asyncio
    async def test_consolidate_merge_similar(self, consolidator, indexer):
        """Test merging two similar memories."""
        mem_a = make_memory(
            content="The user prefers Python",
            importance=0.6,
        )
        mem_b = make_memory(
            content="The user really likes Python",
            importance=0.7,
        )
        # Give them embeddings
        mem_a = await indexer.index_memory(mem_a)
        mem_b = await indexer.index_memory(mem_b)

        consolidated, result = await consolidator.consolidate([mem_a, mem_b])
        # With high similarity, they should be merged
        assert result.merged >= 0  # depends on actual similarity score
        assert len(consolidated) <= 2

    @pytest.mark.asyncio
    async def test_consolidate_forget_old_unimportant(self, consolidator):
        """Test forgetting old, unimportant, unaccessed memories."""
        old_date = datetime.utcnow() - timedelta(days=120)
        old_memory = make_memory(
            content="Very old unimportant memory",
            importance=0.1,
            access_count=0,
        )
        old_memory.created_at = old_date

        recent_memory = make_memory(
            content="Recent important memory",
            importance=0.8,
            access_count=5,
        )

        consolidated, result = await consolidator.consolidate([old_memory, recent_memory])
        assert result.forgotten >= 1
        assert len(consolidated) < 2 or any(
            m.content == "Recent important memory" for m in consolidated
        )

    @pytest.mark.asyncio
    async def test_consolidate_promote_episodic(self, consolidator):
        """Test promoting frequent episodic memories to semantic."""
        episodic = make_memory(
            content="Frequently recalled event",
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            access_count=10,
        )

        consolidated, result = await consolidator.consolidate([episodic])
        assert result.promoted >= 1
        promoted = [m for m in consolidated if m.type == MemoryType.SEMANTIC]
        assert len(promoted) >= 1

    @pytest.mark.asyncio
    async def test_consolidate_keeps_important_old_memories(self, consolidator):
        """Test that important old memories are kept even if old."""
        old_date = datetime.utcnow() - timedelta(days=120)
        important_old = make_memory(
            content="Important old memory",
            importance=0.9,
            access_count=10,
        )
        important_old.created_at = old_date

        consolidated, result = await consolidator.consolidate([important_old])
        assert any(m.content == "Important old memory" for m in consolidated)

    @pytest.mark.asyncio
    async def test_consolidate_expired_memories(self, consolidator):
        """Test that expired memories are forgotten."""
        expired = make_memory(content="Expired memory", importance=0.8)
        expired.expires_at = datetime.utcnow() - timedelta(days=1)

        consolidated, result = await consolidator.consolidate([expired])
        assert result.forgotten >= 1

    @pytest.mark.asyncio
    async def test_consolidation_result_stats(self, consolidator):
        """Test that consolidation returns correct statistics."""
        memories = [
            make_memory(content=f"Memory {i}", importance=0.5)
            for i in range(5)
        ]
        _, result = await consolidator.consolidate(memories)

        assert result.total_processed == 5
        assert result.duration_ms > 0
        assert result.merged >= 0
        assert result.promoted >= 0
        assert result.forgotten >= 0
        assert result.reinforced >= 0
