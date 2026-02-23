"""Tests for the Indexer — embedding, text preparation, cosine similarity."""

import pytest
import pytest_asyncio

from agentvault.core.types import Memory, MemoryType
from agentvault.processing.indexer import Indexer


class TestCosineSimlarity:
    """Tests for the static cosine_similarity method."""

    def test_identical_vectors(self):
        assert Indexer.cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert Indexer.cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert Indexer.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert Indexer.cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_arbitrary_vectors(self):
        sim = Indexer.cosine_similarity([1, 2, 3], [1, 2, 3])
        assert sim == pytest.approx(1.0)


class TestPrepareText:
    """Tests for _prepare_text()."""

    def test_content_only(self, indexer):
        m = Memory(agent_id="a", type=MemoryType.SEMANTIC, content="hello")
        text = indexer._prepare_text(m)
        assert "hello" in text

    def test_includes_tags(self, indexer):
        m = Memory(agent_id="a", type=MemoryType.SEMANTIC, content="hello", tags=["python", "ai"])
        text = indexer._prepare_text(m)
        assert "python" in text
        assert "ai" in text

    def test_includes_short_metadata(self, indexer):
        m = Memory(
            agent_id="a",
            type=MemoryType.SEMANTIC,
            content="hello",
            metadata={"file": "auth.py"},
        )
        text = indexer._prepare_text(m)
        assert "auth.py" in text

    def test_excludes_long_metadata(self, indexer):
        long_val = "x" * 250
        m = Memory(
            agent_id="a",
            type=MemoryType.SEMANTIC,
            content="hello",
            metadata={"description": long_val},
        )
        text = indexer._prepare_text(m)
        assert long_val not in text


class TestEmbedding:
    """Tests for embed_text and index_memory (requires sentence-transformers)."""

    @pytest.mark.asyncio
    async def test_embed_text_returns_floats(self, indexer):
        emb = await indexer.embed_text("hello world")
        assert isinstance(emb, list)
        assert len(emb) > 0
        assert all(isinstance(v, float) for v in emb)

    @pytest.mark.asyncio
    async def test_same_text_same_embedding(self, indexer):
        e1 = await indexer.embed_text("deterministic test")
        e2 = await indexer.embed_text("deterministic test")
        # Should be identical (deterministic model)
        assert e1 == e2

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, indexer):
        e1 = await indexer.embed_text("python programming")
        e2 = await indexer.embed_text("cooking recipes")
        assert e1 != e2

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self, indexer):
        results = await indexer.embed_texts(["one", "two", "three"])
        assert len(results) == 3
        assert all(len(e) > 0 for e in results)

    @pytest.mark.asyncio
    async def test_index_memory_attaches_embedding(self, indexer):
        m = Memory(agent_id="a", type=MemoryType.SEMANTIC, content="test content")
        assert m.embedding is None

        indexed = await indexer.index_memory(m)
        assert indexed.embedding is not None
        assert len(indexed.embedding) > 0
