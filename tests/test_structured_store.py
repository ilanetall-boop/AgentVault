"""Tests for the SQLite structured store — duration parsing, CRUD, filtering."""

import os
from datetime import timedelta

import pytest
import pytest_asyncio

from agentvault.core.types import Memory, MemoryType
from agentvault.storage.structured_store import SQLiteStructuredStore, _parse_duration


class TestParseDuration:
    """Tests for _parse_duration()."""

    def test_days(self):
        assert _parse_duration("30d") == timedelta(days=30)

    def test_hours(self):
        assert _parse_duration("12h") == timedelta(hours=12)

    def test_minutes(self):
        assert _parse_duration("45m") == timedelta(minutes=45)

    def test_seconds(self):
        assert _parse_duration("30s") == timedelta(seconds=30)

    def test_zero(self):
        assert _parse_duration("0d") == timedelta(days=0)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown duration unit"):
            _parse_duration("10x")


class TestSQLiteStructuredStore:
    """Tests for SQLite CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, structured_store):
        m = Memory(agent_id="a", type=MemoryType.SEMANTIC, content="hello")
        await structured_store.save_memory(m)

        retrieved = await structured_store.get_memory(m.id)
        assert retrieved is not None
        assert retrieved.content == "hello"
        assert retrieved.type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, structured_store):
        assert await structured_store.get_memory("no-such-id") is None

    @pytest.mark.asyncio
    async def test_upsert_on_duplicate(self, structured_store):
        """save_memory with same ID should update (merge)."""
        m = Memory(agent_id="a", type=MemoryType.SEMANTIC, content="v1")
        await structured_store.save_memory(m)

        m.content = "v2"
        await structured_store.save_memory(m)

        retrieved = await structured_store.get_memory(m.id)
        assert retrieved.content == "v2"

    @pytest.mark.asyncio
    async def test_delete_memory(self, structured_store):
        m = Memory(agent_id="a", type=MemoryType.SEMANTIC, content="to delete")
        await structured_store.save_memory(m)
        await structured_store.delete_memory(m.id)
        assert await structured_store.get_memory(m.id) is None

    @pytest.mark.asyncio
    async def test_count_memories(self, structured_store):
        await structured_store.save_memory(
            Memory(agent_id="a", type=MemoryType.SEMANTIC, content="one")
        )
        await structured_store.save_memory(
            Memory(agent_id="a", type=MemoryType.EPISODIC, content="two")
        )
        assert await structured_store.count_memories("a") == 2
        assert await structured_store.count_memories("a", MemoryType.SEMANTIC) == 1
        assert await structured_store.count_memories("b") == 0

    @pytest.mark.asyncio
    async def test_get_memories_with_type_filter(self, structured_store):
        await structured_store.save_memory(
            Memory(agent_id="a", type=MemoryType.SEMANTIC, content="fact")
        )
        await structured_store.save_memory(
            Memory(agent_id="a", type=MemoryType.EPISODIC, content="event")
        )

        semantic = await structured_store.get_memories("a", memory_type=MemoryType.SEMANTIC)
        assert len(semantic) == 1
        assert semantic[0].content == "fact"

    @pytest.mark.asyncio
    async def test_get_memories_importance_filter(self, structured_store):
        await structured_store.save_memory(
            Memory(agent_id="a", type=MemoryType.SEMANTIC, content="low", importance=0.1)
        )
        await structured_store.save_memory(
            Memory(agent_id="a", type=MemoryType.SEMANTIC, content="high", importance=0.9)
        )

        high_only = await structured_store.get_memories("a", min_importance=0.5)
        assert len(high_only) == 1
        assert high_only[0].content == "high"

    @pytest.mark.asyncio
    async def test_list_agents(self, structured_store):
        await structured_store.save_memory(
            Memory(agent_id="alice", type=MemoryType.SEMANTIC, content="one")
        )
        await structured_store.save_memory(
            Memory(agent_id="alice", type=MemoryType.SEMANTIC, content="two")
        )
        await structured_store.save_memory(
            Memory(agent_id="bob", type=MemoryType.SEMANTIC, content="three")
        )

        agents = await structured_store.list_agents()
        agent_map = {a["agent_id"]: a["memory_count"] for a in agents}
        assert agent_map["alice"] == 2
        assert agent_map["bob"] == 1

    @pytest.mark.asyncio
    async def test_delete_memories_by_importance(self, structured_store):
        await structured_store.save_memory(
            Memory(agent_id="a", type=MemoryType.SEMANTIC, content="low", importance=0.1)
        )
        await structured_store.save_memory(
            Memory(agent_id="a", type=MemoryType.SEMANTIC, content="high", importance=0.9)
        )

        count = await structured_store.delete_memories("a", importance_below=0.5)
        assert count == 1
        remaining = await structured_store.get_memories("a")
        assert len(remaining) == 1
        assert remaining[0].content == "high"
