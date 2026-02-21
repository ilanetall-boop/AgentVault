"""Tests for episodic memory."""

from datetime import datetime, timedelta

import pytest

from agentvault.core.episodic import EpisodicMemoryManager
from agentvault.core.types import EpisodicMemory, MemoryType


class TestEpisodicMemoryManager:
    """Tests for the EpisodicMemoryManager."""

    def setup_method(self):
        self.manager = EpisodicMemoryManager()

    def test_create_memory(self):
        """Test creating an episodic memory."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="User asked to refactor auth module",
            event="refactor_request",
            context={"file": "auth.py", "action": "refactor"},
            importance=0.7,
            tags=["refactor", "auth"],
        )
        assert isinstance(memory, EpisodicMemory)
        assert memory.type == MemoryType.EPISODIC
        assert memory.agent_id == "agent-1"
        assert memory.event == "refactor_request"
        assert memory.context == {"file": "auth.py", "action": "refactor"}
        assert memory.importance == 0.7
        assert "refactor" in memory.tags

    def test_create_memory_defaults(self):
        """Test creating with minimal params."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="Something happened",
        )
        assert memory.type == MemoryType.EPISODIC
        assert memory.event == "Something happened"
        assert memory.importance == 0.5
        assert memory.tags == []

    def test_should_promote_to_semantic(self):
        """Test promotion detection."""
        memory = EpisodicMemory(
            agent_id="agent-1",
            content="Frequently accessed event",
            access_count=6,
            importance=0.7,
        )
        assert self.manager.should_promote_to_semantic(memory, access_threshold=5)

    def test_should_not_promote_low_access(self):
        """Test that low-access memories are not promoted."""
        memory = EpisodicMemory(
            agent_id="agent-1",
            content="Rarely accessed event",
            access_count=2,
            importance=0.7,
        )
        assert not self.manager.should_promote_to_semantic(memory)

    def test_should_not_promote_low_importance(self):
        """Test that low-importance memories are not promoted."""
        memory = EpisodicMemory(
            agent_id="agent-1",
            content="Low importance event",
            access_count=10,
            importance=0.3,
        )
        assert not self.manager.should_promote_to_semantic(memory)

    def test_extract_key_info(self):
        """Test key information extraction."""
        memory = self.manager.create_memory(
            agent_id="agent-1",
            content="User asked to deploy the app",
            event="deploy_request",
            context={"env": "production"},
            participants=["user", "agent-1"],
        )
        info = self.manager.extract_key_info(memory)
        assert info["event"] == "deploy_request"
        assert info["context"] == {"env": "production"}
        assert info["participants"] == ["user", "agent-1"]

    def test_recency_score_decreases_over_time(self):
        """Test that recency score decreases for older memories."""
        recent = EpisodicMemory(
            agent_id="agent-1",
            content="Recent event",
            importance=0.8,
            created_at=datetime.utcnow(),
        )
        old = EpisodicMemory(
            agent_id="agent-1",
            content="Old event",
            importance=0.8,
            created_at=datetime.utcnow() - timedelta(days=30),
        )
        recent_score = self.manager.calculate_recency_score(recent)
        old_score = self.manager.calculate_recency_score(old)
        assert recent_score > old_score
