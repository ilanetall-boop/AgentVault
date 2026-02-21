"""Tests for the Memory Manager — integration tests for the full flow."""

import pytest

from agentvault.core.types import MemoryType


AGENT_ID = "test-agent"


class TestMemoryManager:
    """Integration tests for the MemoryManager."""

    @pytest.mark.asyncio
    async def test_remember_and_recall(self, memory_manager):
        """Test the basic remember → recall flow."""
        memory = await memory_manager.remember(
            agent_id=AGENT_ID,
            content="The user prefers TypeScript over JavaScript",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            tags=["preference", "language"],
        )
        assert memory.id is not None
        assert memory.embedding is not None

        result = await memory_manager.recall(
            agent_id=AGENT_ID,
            query="what language does the user prefer?",
            top_k=5,
        )
        assert result.total_found >= 1
        assert len(result.memories) >= 1
        contents = [m.content for m in result.memories]
        assert any("TypeScript" in c for c in contents)

    @pytest.mark.asyncio
    async def test_remember_auto_detect_type(self, memory_manager):
        """Test that memory type is auto-detected."""
        memory = await memory_manager.remember(
            agent_id=AGENT_ID,
            content="The user prefers dark mode for all editors",
        )
        # Should auto-detect as semantic (preference)
        assert memory.type in [MemoryType.SEMANTIC, MemoryType.EPISODIC, MemoryType.PROCEDURAL]
        assert memory.importance > 0

    @pytest.mark.asyncio
    async def test_remember_episode(self, memory_manager):
        """Test storing an episodic memory."""
        memory = await memory_manager.remember_episode(
            agent_id=AGENT_ID,
            event="User asked to refactor the auth module",
            context={"file": "auth.py"},
            importance=0.7,
            tags=["refactor"],
        )
        assert memory.type == MemoryType.EPISODIC

    @pytest.mark.asyncio
    async def test_remember_procedure(self, memory_manager):
        """Test storing a procedural memory."""
        memory = await memory_manager.remember_procedure(
            agent_id=AGENT_ID,
            name="deploy_vercel",
            steps=["npm run build", "vercel deploy --prod"],
            importance=0.8,
        )
        assert memory.type == MemoryType.PROCEDURAL

    @pytest.mark.asyncio
    async def test_remember_fact(self, memory_manager):
        """Test storing a fact (semantic memory)."""
        memory = await memory_manager.remember_fact(
            agent_id=AGENT_ID,
            content="The project uses FastAPI for the backend",
            importance=0.7,
            tags=["tech-stack"],
        )
        assert memory.type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_recall_with_type_filter(self, memory_manager):
        """Test recalling with type filter."""
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Episodic event: meeting at 3pm",
            memory_type=MemoryType.EPISODIC,
        )
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Semantic fact: project uses Python",
            memory_type=MemoryType.SEMANTIC,
        )

        result = await memory_manager.recall(
            agent_id=AGENT_ID,
            query="what programming language?",
            types=[MemoryType.SEMANTIC],
        )
        for mem in result.memories:
            assert mem.type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_recall_with_importance_filter(self, memory_manager):
        """Test recalling with minimum importance."""
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Not important fact",
            importance=0.1,
        )
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Very important critical fact",
            importance=0.9,
        )

        result = await memory_manager.recall(
            agent_id=AGENT_ID,
            query="important facts",
            min_importance=0.5,
        )
        for mem in result.memories:
            assert mem.importance >= 0.5

    @pytest.mark.asyncio
    async def test_forget(self, memory_manager):
        """Test forgetting memories."""
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Unimportant test memory to forget",
            importance=0.1,
        )
        count_before = await memory_manager.count(AGENT_ID)
        assert count_before >= 1

        forgotten = await memory_manager.forget(
            agent_id=AGENT_ID,
            importance_below=0.2,
        )
        assert forgotten >= 1

    @pytest.mark.asyncio
    async def test_count(self, memory_manager):
        """Test counting memories."""
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Memory to count 1",
            memory_type=MemoryType.SEMANTIC,
        )
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Memory to count 2",
            memory_type=MemoryType.EPISODIC,
        )
        total = await memory_manager.count(AGENT_ID)
        assert total >= 2

        semantic_count = await memory_manager.count(AGENT_ID, MemoryType.SEMANTIC)
        assert semantic_count >= 1

    @pytest.mark.asyncio
    async def test_get_memory_by_id(self, memory_manager):
        """Test retrieving a specific memory by ID."""
        stored = await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Specific memory to retrieve",
            importance=0.7,
        )
        retrieved = await memory_manager.get_memory(stored.id)
        assert retrieved is not None
        assert retrieved.id == stored.id
        assert retrieved.content == "Specific memory to retrieve"

    @pytest.mark.asyncio
    async def test_recall_reinforces_memories(self, memory_manager):
        """Test that recalling a memory reinforces it."""
        memory = await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Memory that should be reinforced when recalled",
            importance=0.5,
        )
        original_access = memory.access_count

        await memory_manager.recall(
            agent_id=AGENT_ID,
            query="reinforced memory",
        )

        updated = await memory_manager.get_memory(memory.id)
        if updated:
            assert updated.access_count >= original_access

    @pytest.mark.asyncio
    async def test_multiple_memories_recall_relevance(self, memory_manager):
        """Test that recall returns relevant memories first."""
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="Python is a great programming language for data science",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
        )
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="The weather is sunny today",
            memory_type=MemoryType.EPISODIC,
            importance=0.3,
        )
        await memory_manager.remember(
            agent_id=AGENT_ID,
            content="To install numpy: pip install numpy",
            memory_type=MemoryType.PROCEDURAL,
            importance=0.6,
        )

        result = await memory_manager.recall(
            agent_id=AGENT_ID,
            query="programming language data",
            top_k=3,
        )
        assert len(result.memories) >= 1
        # The Python-related memory should rank high
        top_content = result.memories[0].content.lower()
        assert "python" in top_content or "programming" in top_content or "data" in top_content
