"""Tests for the FAISS vector store — covers recently fixed bugs."""

import os

import numpy as np
import pytest
import pytest_asyncio

from agentvault.storage.vector_store import FAISSVectorStore


@pytest_asyncio.fixture
async def store(tmp_dir):
    """Create a fresh FAISS store in a temp dir."""
    path = os.path.join(tmp_dir, "faiss")
    s = FAISSVectorStore(persist_directory=path, dimension=4)
    await s.initialize()
    yield s
    await s.close()


def _vec(values: list[float]) -> list[float]:
    """Helper: normalize a 4-d vector."""
    arr = np.array(values, dtype=np.float32)
    arr /= np.linalg.norm(arr)
    return arr.tolist()


class TestFAISSVectorStore:
    """Unit tests for FAISSVectorStore."""

    @pytest.mark.asyncio
    async def test_add_and_search(self, store):
        """Basic add + search returns the stored vector."""
        vec = _vec([1, 0, 0, 0])
        await store.add("mem-1", vec, {"agent_id": "a"})

        results = await store.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "mem-1"
        assert results[0][1] >= 0.99  # cosine ~1.0

    @pytest.mark.asyncio
    async def test_search_empty_index(self, store):
        """Search on empty index returns empty list."""
        results = await store.search(_vec([1, 0, 0, 0]), top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_orphans_vector(self, store):
        """Deleted vector should not appear in search results."""
        vec = _vec([1, 0, 0, 0])
        await store.add("mem-1", vec, {"agent_id": "a"})
        await store.delete("mem-1")

        results = await store.search(vec, top_k=5)
        assert all(mid != "mem-1" for mid, _ in results)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_safe(self, store):
        """Deleting a non-existent ID should not raise."""
        await store.delete("does-not-exist")

    @pytest.mark.asyncio
    async def test_replace_vector_on_readd(self, store):
        """Re-adding same ID replaces the vector (BUG-1 regression test)."""
        old_vec = _vec([1, 0, 0, 0])
        new_vec = _vec([0, 1, 0, 0])

        await store.add("mem-1", old_vec, {"agent_id": "a"})
        await store.add("mem-1", new_vec, {"agent_id": "a"})

        # Search with new vector should find mem-1 with high score
        results_new = await store.search(new_vec, top_k=5)
        assert len(results_new) >= 1
        assert results_new[0][0] == "mem-1"
        assert results_new[0][1] >= 0.99

        # Only one mapping should exist for mem-1
        assert store._id_to_idx["mem-1"] == 1  # second index position

    @pytest.mark.asyncio
    async def test_filter_by_metadata(self, store):
        """Filtering search results by metadata works."""
        await store.add("mem-a", _vec([1, 0, 0, 0]), {"agent_id": "alice"})
        await store.add("mem-b", _vec([1, 0, 0, 0]), {"agent_id": "bob"})

        results = await store.search(
            _vec([1, 0, 0, 0]),
            top_k=5,
            filters={"agent_id": "alice"},
        )
        ids = [mid for mid, _ in results]
        assert "mem-a" in ids
        assert "mem-b" not in ids

    @pytest.mark.asyncio
    async def test_filter_with_list_value(self, store):
        """Filter with a list value uses 'in' semantics."""
        await store.add("m1", _vec([1, 0, 0, 0]), {"type": "semantic"})
        await store.add("m2", _vec([1, 0, 0, 0]), {"type": "episodic"})
        await store.add("m3", _vec([1, 0, 0, 0]), {"type": "procedural"})

        results = await store.search(
            _vec([1, 0, 0, 0]),
            top_k=5,
            filters={"type": ["semantic", "episodic"]},
        )
        ids = [mid for mid, _ in results]
        assert "m1" in ids
        assert "m2" in ids
        assert "m3" not in ids

    @pytest.mark.asyncio
    async def test_update_metadata(self, store):
        """update_metadata updates without re-adding vector."""
        await store.add("mem-1", _vec([1, 0, 0, 0]), {"agent_id": "a", "importance": 0.5})
        await store.update_metadata("mem-1", {"agent_id": "a", "importance": 0.9})

        # Verify updated metadata by filtering
        results = await store.search(
            _vec([1, 0, 0, 0]),
            top_k=1,
        )
        assert results[0][0] == "mem-1"
        assert store._metadata["mem-1"]["importance"] == 0.9

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_dir):
        """Index persists to disk and reloads correctly."""
        path = os.path.join(tmp_dir, "faiss_persist")
        vec = _vec([1, 0, 0, 0])

        # Write
        s1 = FAISSVectorStore(persist_directory=path, dimension=4)
        await s1.initialize()
        await s1.add("mem-1", vec, {"agent_id": "a"})
        await s1.close()

        # Read
        s2 = FAISSVectorStore(persist_directory=path, dimension=4)
        await s2.initialize()
        results = await s2.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "mem-1"
        await s2.close()

    @pytest.mark.asyncio
    async def test_ensure_index_raises_uninitialized(self, tmp_dir):
        """_ensure_index raises RuntimeError before initialize()."""
        s = FAISSVectorStore(persist_directory=os.path.join(tmp_dir, "x"))
        with pytest.raises(RuntimeError, match="not initialized"):
            s._ensure_index()

    @pytest.mark.asyncio
    async def test_sanitize_metadata(self):
        """Metadata sanitization converts unsupported types to string."""
        result = FAISSVectorStore._sanitize_metadata({
            "name": "test",
            "count": 42,
            "active": True,
            "tags": ["a", "b"],
            "nested": {"key": "val"},
        })
        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["active"] is True
        assert result["tags"] == "a,b"
        assert isinstance(result["nested"], str)
