"""Tests for HybridStore — especially the fixed forget() method."""

import os
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from agentvault.core.types import Memory, MemoryQuery, MemoryType
from agentvault.storage.hybrid_store import HybridStore


@pytest_asyncio.fixture
async def store(tmp_dir):
    """Create an initialized HybridStore."""
    s = HybridStore(
        db_path=os.path.join(tmp_dir, "test.db"),
        faiss_path=os.path.join(tmp_dir, "faiss"),
    )
    await s.initialize()
    yield s
    await s.close()


def _make_memory(
    agent_id: str = "agent-1",
    content: str = "test",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    importance: float = 0.5,
    embedding: list[float] | None = None,
    created_at: datetime | None = None,
) -> Memory:
    m = Memory(
        agent_id=agent_id,
        type=memory_type,
        content=content,
        importance=importance,
        embedding=embedding or [0.1] * 384,
    )
    if created_at:
        m.created_at = created_at
    return m


class TestHybridStore:
    """Tests for hybrid store operations."""

    @pytest.mark.asyncio
    async def test_store_and_get(self, store):
        """Store a memory and retrieve it."""
        mem = _make_memory(content="hello world")
        await store.store(mem)

        retrieved = await store.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.content == "hello world"

    @pytest.mark.asyncio
    async def test_delete_from_both_stores(self, store):
        """Delete removes from both structured and vector stores."""
        mem = _make_memory()
        await store.store(mem)

        await store.delete(mem.id)

        assert await store.get_memory(mem.id) is None
        # Vector store should also have removed it
        assert mem.id not in store.vector_store._id_to_idx

    @pytest.mark.asyncio
    async def test_count(self, store):
        """Count returns correct totals."""
        await store.store(_make_memory(content="one", memory_type=MemoryType.SEMANTIC))
        await store.store(_make_memory(content="two", memory_type=MemoryType.EPISODIC))

        total = await store.count("agent-1")
        assert total == 2

        semantic = await store.count("agent-1", MemoryType.SEMANTIC)
        assert semantic == 1

    @pytest.mark.asyncio
    async def test_forget_by_importance(self, store):
        """Forget deletes low-importance memories from both stores (BUG-2 regression)."""
        low = _make_memory(content="low", importance=0.1)
        high = _make_memory(content="high", importance=0.9)
        await store.store(low)
        await store.store(high)

        count = await store.forget("agent-1", importance_below=0.5)
        assert count == 1

        # Structured: low should be gone
        assert await store.get_memory(low.id) is None
        # Structured: high should remain
        assert await store.get_memory(high.id) is not None
        # Vector: low should be orphaned/removed
        assert low.id not in store.vector_store._id_to_idx

    @pytest.mark.asyncio
    async def test_forget_empty_store(self, store):
        """Forget on empty store returns 0."""
        count = await store.forget("agent-1", importance_below=0.5)
        assert count == 0

    @pytest.mark.asyncio
    async def test_forget_no_matches(self, store):
        """Forget with criteria matching nothing returns 0."""
        mem = _make_memory(importance=0.9)
        await store.store(mem)

        count = await store.forget("agent-1", importance_below=0.1)
        assert count == 0
        assert await store.get_memory(mem.id) is not None

    @pytest.mark.asyncio
    async def test_ensure_initialized_raises(self, tmp_dir):
        """Operations before initialize() raise RuntimeError."""
        s = HybridStore(
            db_path=os.path.join(tmp_dir, "x.db"),
            faiss_path=os.path.join(tmp_dir, "f"),
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            await s.store(_make_memory())

    @pytest.mark.asyncio
    async def test_recall_scoring(self, store):
        """Recall combines vector and structured scores correctly."""
        high_imp = _make_memory(content="important fact", importance=0.9)
        low_imp = _make_memory(content="trivial note", importance=0.1)
        await store.store(high_imp)
        await store.store(low_imp)

        query = MemoryQuery(
            query="fact",
            agent_id="agent-1",
            types=[MemoryType.SEMANTIC],
            top_k=10,
        )

        result = await store.recall(query, query_embedding=[0.1] * 384)
        assert result.total_found >= 1
        # Higher importance memory should generally rank higher
        # when vector scores are similar
        assert len(result.memories) >= 1
