"""Shared memory management for multi-agent collaboration."""

from __future__ import annotations

import logging
from typing import Any

from agentvault.core.memory_manager import MemoryManager
from agentvault.core.types import (
    AgentPermission,
    RecallResult,
    SharedNamespace,
)
from agentvault.multi_agent.permissions import PermissionManager

logger = logging.getLogger(__name__)


class SharedMemoryManager:
    """Manages shared memory namespaces for multi-agent collaboration.

    Each agent has its own private memory space. Agents can create
    shared namespaces to collaborate. Memories in shared namespaces
    are visible to all agents with appropriate permissions.

    Features:
    - Namespace-based isolation
    - Permission control (read/write/admin)
    - Cross-agent memory sharing
    - Event notifications for memory changes
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._manager = memory_manager
        self._permissions = PermissionManager()
        self._namespaces: dict[str, SharedNamespace] = {}
        self._listeners: dict[str, list[callable]] = {}

    def create_namespace(
        self,
        name: str,
        owner_agent_id: str,
        member_agents: dict[str, str] | None = None,
    ) -> SharedNamespace:
        """Create a new shared namespace.

        The owner automatically gets admin permission. Other agents
        can be added with specified permission levels.

        Args:
            name: Human-readable name for the namespace.
            owner_agent_id: Agent ID of the namespace owner.
            member_agents: Dict of {agent_id: permission} to add.

        Returns:
            The created SharedNamespace.
        """
        namespace = SharedNamespace(
            name=name,
            owner_agent_id=owner_agent_id,
        )

        self._namespaces[namespace.namespace_id] = namespace
        self._permissions.grant(namespace.namespace_id, owner_agent_id, AgentPermission.ADMIN)

        if member_agents:
            for agent_id, permission in member_agents.items():
                self._permissions.grant(namespace.namespace_id, agent_id, permission)
                namespace.permissions[agent_id] = AgentPermission(permission)

        logger.info(
            "Created namespace '%s' (id=%s) owned by %s",
            name,
            namespace.namespace_id,
            owner_agent_id,
        )
        return namespace

    async def share_memory(
        self,
        memory_id: str,
        namespace_id: str,
        agent_id: str,
    ) -> None:
        """Share a memory into a namespace.

        The agent must have write permission to the namespace.

        Args:
            memory_id: ID of the memory to share.
            namespace_id: ID of the target namespace.
            agent_id: ID of the agent sharing the memory.

        Raises:
            PermissionError: If the agent lacks write permission.
            ValueError: If the memory or namespace doesn't exist.
        """
        if namespace_id not in self._namespaces:
            raise ValueError(f"Namespace {namespace_id} not found")

        if not self._permissions.has_permission(namespace_id, agent_id, AgentPermission.WRITE):
            raise PermissionError(
                f"Agent {agent_id} does not have write permission "
                f"for namespace {namespace_id}"
            )

        memory = await self._manager.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} not found")

        # Tag the memory with the namespace
        if "shared_namespaces" not in memory.metadata:
            memory.metadata["shared_namespaces"] = []
        if namespace_id not in memory.metadata["shared_namespaces"]:
            memory.metadata["shared_namespaces"].append(namespace_id)
        await self._manager._store.update(memory)

        # Notify listeners
        await self._notify(namespace_id, "memory_shared", {
            "memory_id": memory_id,
            "agent_id": agent_id,
        })

        logger.info(
            "Agent %s shared memory %s to namespace %s",
            agent_id,
            memory_id,
            namespace_id,
        )

    async def recall_shared(
        self,
        namespace_id: str,
        agent_id: str,
        query: str,
        top_k: int = 10,
    ) -> RecallResult:
        """Recall memories from a shared namespace.

        The agent must have read permission to the namespace.

        Args:
            namespace_id: ID of the namespace to search.
            agent_id: ID of the agent making the query.
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            RecallResult with matching shared memories.

        Raises:
            PermissionError: If the agent lacks read permission.
        """
        if not self._permissions.has_permission(namespace_id, agent_id, AgentPermission.READ):
            raise PermissionError(
                f"Agent {agent_id} does not have read permission "
                f"for namespace {namespace_id}"
            )

        # Get the namespace owner to search their memories
        namespace = self._namespaces.get(namespace_id)
        if namespace is None:
            raise ValueError(f"Namespace {namespace_id} not found")

        result = await self._manager.recall(
            agent_id=namespace.owner_agent_id,
            query=query,
            top_k=top_k,
        )

        # Filter to only shared memories in this namespace
        shared_memories = []
        shared_scores = []
        for mem, score in zip(result.memories, result.relevance_scores, strict=False):
            ns_list = mem.metadata.get("shared_namespaces", [])
            if namespace_id in ns_list:
                shared_memories.append(mem)
                shared_scores.append(score)

        return RecallResult(
            memories=shared_memories,
            relevance_scores=shared_scores,
            total_found=len(shared_memories),
            query_time_ms=result.query_time_ms,
        )

    def add_agent_to_namespace(
        self,
        namespace_id: str,
        agent_id: str,
        permission: str | AgentPermission,
        granted_by: str,
    ) -> None:
        """Add an agent to a shared namespace.

        Only admin agents can add other agents.

        Args:
            namespace_id: The namespace.
            agent_id: The agent to add.
            permission: Permission level to grant.
            granted_by: The admin agent granting access.

        Raises:
            PermissionError: If granted_by lacks admin permission.
        """
        if not self._permissions.has_permission(namespace_id, granted_by, AgentPermission.ADMIN):
            raise PermissionError(
                f"Agent {granted_by} does not have admin permission"
            )
        self._permissions.grant(namespace_id, agent_id, permission)
        logger.info(
            "Agent %s added agent %s to namespace %s with %s permission",
            granted_by,
            agent_id,
            namespace_id,
            permission,
        )

    def remove_agent_from_namespace(
        self,
        namespace_id: str,
        agent_id: str,
        removed_by: str,
    ) -> None:
        """Remove an agent from a shared namespace.

        Only admin agents can remove other agents.

        Args:
            namespace_id: The namespace.
            agent_id: The agent to remove.
            removed_by: The admin agent revoking access.

        Raises:
            PermissionError: If removed_by lacks admin permission.
        """
        if not self._permissions.has_permission(namespace_id, removed_by, AgentPermission.ADMIN):
            raise PermissionError(
                f"Agent {removed_by} does not have admin permission"
            )
        self._permissions.revoke(namespace_id, agent_id)

    def on_memory_change(self, namespace_id: str, callback: callable) -> None:
        """Register a listener for memory changes in a namespace.

        Args:
            namespace_id: The namespace to listen to.
            callback: Async callable(event_type, data) to invoke.
        """
        if namespace_id not in self._listeners:
            self._listeners[namespace_id] = []
        self._listeners[namespace_id].append(callback)

    async def _notify(
        self,
        namespace_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Notify all listeners of a memory change event."""
        for callback in self._listeners.get(namespace_id, []):
            try:
                await callback(event_type, data)
            except Exception:
                logger.warning(
                    "Listener error for namespace %s", namespace_id, exc_info=True
                )

    def get_namespace(self, namespace_id: str) -> SharedNamespace | None:
        """Get a namespace by ID."""
        return self._namespaces.get(namespace_id)

    def list_namespaces(self, agent_id: str) -> list[SharedNamespace]:
        """List all namespaces accessible to an agent."""
        ns_ids = self._permissions.list_agent_namespaces(agent_id)
        return [self._namespaces[ns_id] for ns_id in ns_ids if ns_id in self._namespaces]
