"""Tests for semantic memory."""

import pytest

from agentvault.core.semantic import SemanticMemoryManager
from agentvault.core.types import MemoryType, SemanticMemory


class TestSemanticMemoryManager:
    """Tests for the SemanticMemoryManager."""

    def setup_method(self):
        self.manager = SemanticMemoryManager()

    def test_create_memory(self):
        """Test creating a semantic memory."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="The user prefers TypeScript over JavaScript",
            subject="user",
            predicate="prefers",
            obj="TypeScript",
            confidence=0.9,
            importance=0.8,
            tags=["preference", "language"],
        )
        assert isinstance(memory, SemanticMemory)
        assert memory.type == MemoryType.SEMANTIC
        assert memory.subject == "user"
        assert memory.predicate == "prefers"
        assert memory.obj == "TypeScript"
        assert memory.confidence == 0.9

    def test_create_from_triple(self):
        """Test creating from a knowledge triple."""
        memory = self.manager.create_from_triple(
            agent_id="agent-1",
            subject="user",
            predicate="prefers",
            obj="dark mode",
        )
        assert memory.content == "user prefers dark mode"
        assert memory.subject == "user"
        assert memory.predicate == "prefers"
        assert memory.obj == "dark mode"

    def test_merge_memories(self):
        """Test merging two similar semantic memories."""
        mem_a = SemanticMemory(
            agent_id="agent-1",
            content="User likes Python",
            subject="user",
            predicate="likes",
            obj="Python",
            confidence=0.7,
            importance=0.5,
            tags=["preference"],
            access_count=3,
        )
        mem_b = SemanticMemory(
            agent_id="agent-1",
            content="The user really likes Python for scripting",
            subject="user",
            predicate="likes",
            obj="Python",
            confidence=0.8,
            importance=0.6,
            tags=["language"],
            access_count=2,
        )
        merged = self.manager.merge_memories(mem_a, mem_b)
        assert merged.id == mem_a.id  # keeps original ID
        assert merged.confidence == pytest.approx(0.8)  # boosted
        assert merged.importance == pytest.approx(0.6)  # max of both
        assert "preference" in merged.tags
        assert "language" in merged.tags
        assert merged.access_count == 5  # combined

    def test_is_contradictory(self):
        """Test contradiction detection."""
        mem_a = SemanticMemory(
            agent_id="agent-1",
            content="User prefers Python",
            subject="user",
            predicate="prefers",
            obj="Python",
        )
        mem_b = SemanticMemory(
            agent_id="agent-1",
            content="User prefers JavaScript",
            subject="user",
            predicate="prefers",
            obj="JavaScript",
        )
        assert self.manager.is_contradictory(mem_a, mem_b)

    def test_not_contradictory_different_subject(self):
        """Test that different subjects are not contradictory."""
        mem_a = SemanticMemory(
            agent_id="agent-1",
            content="User prefers Python",
            subject="user",
            predicate="prefers",
            obj="Python",
        )
        mem_b = SemanticMemory(
            agent_id="agent-1",
            content="Team prefers JavaScript",
            subject="team",
            predicate="prefers",
            obj="JavaScript",
        )
        assert not self.manager.is_contradictory(mem_a, mem_b)

    def test_resolve_contradiction_by_confidence(self):
        """Test contradiction resolution favoring higher confidence."""
        mem_a = SemanticMemory(
            agent_id="agent-1",
            content="User prefers Python",
            subject="user",
            predicate="prefers",
            obj="Python",
            confidence=0.5,
        )
        mem_b = SemanticMemory(
            agent_id="agent-1",
            content="User prefers TypeScript",
            subject="user",
            predicate="prefers",
            obj="TypeScript",
            confidence=0.9,
        )
        winner = self.manager.resolve_contradiction(mem_a, mem_b)
        assert winner.obj == "TypeScript"

    def test_extract_knowledge(self):
        """Test knowledge extraction."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="User uses VS Code",
            subject="user",
            predicate="uses",
            obj="VS Code",
            confidence=0.95,
            tags=["tools"],
        )
        knowledge = self.manager.extract_knowledge(memory)
        assert knowledge["subject"] == "user"
        assert knowledge["predicate"] == "uses"
        assert knowledge["object"] == "VS Code"
        assert knowledge["confidence"] == 0.95
