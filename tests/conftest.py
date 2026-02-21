"""Shared test fixtures for AgentVault tests."""

from __future__ import annotations

import os
import tempfile

import pytest
import pytest_asyncio

from agentvault.core.memory_manager import MemoryManager
from agentvault.core.types import Memory, MemoryType
from agentvault.processing.consolidator import Consolidator
from agentvault.processing.extractor import Extractor
from agentvault.processing.indexer import Indexer
from agentvault.processing.ranker import Ranker
from agentvault.storage.hybrid_store import HybridStore
from agentvault.storage.structured_store import SQLiteStructuredStore
from agentvault.storage.vector_store import FAISSVectorStore


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def extractor():
    """Create an Extractor instance."""
    return Extractor()


@pytest.fixture
def ranker():
    """Create a Ranker instance."""
    return Ranker()


@pytest_asyncio.fixture
async def structured_store(tmp_dir):
    """Create and initialize a SQLite structured store."""
    db_path = os.path.join(tmp_dir, "test.db")
    store = SQLiteStructuredStore(db_path=db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def vector_store(tmp_dir):
    """Create and initialize a FAISS vector store."""
    faiss_path = os.path.join(tmp_dir, "faiss")
    store = FAISSVectorStore(persist_directory=faiss_path)
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def hybrid_store(tmp_dir):
    """Create and initialize a hybrid store."""
    db_path = os.path.join(tmp_dir, "test.db")
    faiss_path = os.path.join(tmp_dir, "faiss")
    store = HybridStore(db_path=db_path, faiss_path=faiss_path)
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def indexer():
    """Create and initialize an Indexer."""
    idx = Indexer()
    await idx.initialize()
    yield idx


@pytest_asyncio.fixture
async def consolidator(indexer):
    """Create a Consolidator with initialized indexer."""
    return Consolidator(indexer=indexer)


@pytest_asyncio.fixture
async def memory_manager(tmp_dir):
    """Create and initialize a MemoryManager."""
    db_path = os.path.join(tmp_dir, "test.db")
    faiss_path = os.path.join(tmp_dir, "faiss")
    manager = MemoryManager(db_path=db_path, faiss_path=faiss_path)
    await manager.initialize()
    yield manager
    await manager.close()


def make_memory(
    agent_id: str = "test-agent",
    content: str = "Test memory content",
    memory_type: MemoryType = MemoryType.SEMANTIC,
    importance: float = 0.5,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    access_count: int = 0,
) -> Memory:
    """Helper to create a Memory instance for testing."""
    return Memory(
        agent_id=agent_id,
        type=memory_type,
        content=content,
        importance=importance,
        tags=tags or [],
        embedding=embedding,
        access_count=access_count,
    )
