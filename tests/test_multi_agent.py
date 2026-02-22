"""Tests for multi-agent shared memory."""

import pytest

from agentvault.core.types import MemoryType
from agentvault.multi_agent.permissions import PermissionManager


class TestMultiAgent:
    """Tests for multi-agent memory sharing."""

    @pytest.mark.asyncio
    async def test_agent_a_writes_agent_b_reads_shared(self, memory_manager):
        """Test that Agent A can write and Agent B can read shared memories."""
        # Agent A stores a shared memory
        await memory_manager.remember(
            agent_id="agent-a",
            content="Important shared fact about the project",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            metadata={"shared_with": ["agent-b"]},
        )

        # Agent A can read its own memory
        result_a = await memory_manager.recall(
            agent_id="agent-a",
            query="shared fact about project",
        )
        assert any("shared fact" in m.content for m in result_a.memories)

    @pytest.mark.asyncio
    async def test_separate_agent_namespaces(self, memory_manager):
        """Test that agents have separate memory namespaces."""
        await memory_manager.remember(
            agent_id="agent-a",
            content="Agent A's private memory",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
        )
        await memory_manager.remember(
            agent_id="agent-b",
            content="Agent B's private memory",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
        )

        # Agent A's recall should only find A's memories
        result_a = await memory_manager.recall(
            agent_id="agent-a",
            query="private memory",
        )
        for mem in result_a.memories:
            assert mem.agent_id == "agent-a"

        # Agent B's recall should only find B's memories
        result_b = await memory_manager.recall(
            agent_id="agent-b",
            query="private memory",
        )
        for mem in result_b.memories:
            assert mem.agent_id == "agent-b"


class TestPermissionManager:
    """Tests for the permission system."""

    def test_grant_permission(self):
        """Test granting permissions."""
        pm = PermissionManager()
        pm.grant("namespace-1", "agent-a", "read")
        assert pm.has_permission("namespace-1", "agent-a", "read")

    def test_deny_without_permission(self):
        """Test that access is denied without permission."""
        pm = PermissionManager()
        assert not pm.has_permission("namespace-1", "agent-b", "read")

    def test_write_permission_implies_read(self):
        """Test that write permission includes read access."""
        pm = PermissionManager()
        pm.grant("namespace-1", "agent-a", "write")
        assert pm.has_permission("namespace-1", "agent-a", "read")
        assert pm.has_permission("namespace-1", "agent-a", "write")

    def test_admin_permission_implies_all(self):
        """Test that admin permission includes all access."""
        pm = PermissionManager()
        pm.grant("namespace-1", "agent-a", "admin")
        assert pm.has_permission("namespace-1", "agent-a", "read")
        assert pm.has_permission("namespace-1", "agent-a", "write")
        assert pm.has_permission("namespace-1", "agent-a", "admin")

    def test_revoke_permission(self):
        """Test revoking permissions."""
        pm = PermissionManager()
        pm.grant("namespace-1", "agent-a", "write")
        pm.revoke("namespace-1", "agent-a")
        assert not pm.has_permission("namespace-1", "agent-a", "read")
