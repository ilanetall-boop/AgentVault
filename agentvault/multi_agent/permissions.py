"""Permission management for multi-agent memory access."""

from __future__ import annotations

import logging
from typing import Optional

from agentvault.core.types import AgentPermission

logger = logging.getLogger(__name__)

# Permission hierarchy: admin > write > read
_PERMISSION_HIERARCHY = {
    AgentPermission.READ: 0,
    AgentPermission.WRITE: 1,
    AgentPermission.ADMIN: 2,
}


class PermissionManager:
    """Manages per-agent permissions for shared memory namespaces.

    Supports a simple permission hierarchy:
    - READ: can read memories from the namespace
    - WRITE: can read and write memories (implies READ)
    - ADMIN: full access including delete and manage permissions (implies WRITE)
    """

    def __init__(self) -> None:
        # {namespace_id: {agent_id: permission_level}}
        self._permissions: dict[str, dict[str, AgentPermission]] = {}

    def grant(
        self,
        namespace_id: str,
        agent_id: str,
        permission: str | AgentPermission,
    ) -> None:
        """Grant a permission to an agent for a namespace.

        Args:
            namespace_id: The namespace to grant access to.
            agent_id: The agent receiving the permission.
            permission: The permission level ('read', 'write', 'admin').
        """
        if isinstance(permission, str):
            permission = AgentPermission(permission)

        if namespace_id not in self._permissions:
            self._permissions[namespace_id] = {}
        self._permissions[namespace_id][agent_id] = permission
        logger.info(
            "Granted %s permission to agent %s for namespace %s",
            permission.value,
            agent_id,
            namespace_id,
        )

    def revoke(self, namespace_id: str, agent_id: str) -> None:
        """Revoke all permissions for an agent in a namespace.

        Args:
            namespace_id: The namespace to revoke access from.
            agent_id: The agent losing the permission.
        """
        if namespace_id in self._permissions:
            self._permissions[namespace_id].pop(agent_id, None)
            logger.info(
                "Revoked permissions for agent %s in namespace %s",
                agent_id,
                namespace_id,
            )

    def has_permission(
        self,
        namespace_id: str,
        agent_id: str,
        required: str | AgentPermission,
    ) -> bool:
        """Check if an agent has the required permission level.

        Higher permissions imply lower ones (admin implies write implies read).

        Args:
            namespace_id: The namespace to check.
            agent_id: The agent to check.
            required: The minimum required permission level.

        Returns:
            True if the agent has sufficient permission.
        """
        if isinstance(required, str):
            required = AgentPermission(required)

        ns_perms = self._permissions.get(namespace_id, {})
        agent_perm = ns_perms.get(agent_id)
        if agent_perm is None:
            return False

        return _PERMISSION_HIERARCHY[agent_perm] >= _PERMISSION_HIERARCHY[required]

    def get_permission(
        self,
        namespace_id: str,
        agent_id: str,
    ) -> Optional[AgentPermission]:
        """Get the current permission level for an agent.

        Args:
            namespace_id: The namespace.
            agent_id: The agent.

        Returns:
            The permission level, or None if no permission.
        """
        return self._permissions.get(namespace_id, {}).get(agent_id)

    def list_permissions(self, namespace_id: str) -> dict[str, AgentPermission]:
        """List all permissions for a namespace.

        Args:
            namespace_id: The namespace.

        Returns:
            Dict mapping agent_id to permission level.
        """
        return dict(self._permissions.get(namespace_id, {}))

    def list_agent_namespaces(self, agent_id: str) -> list[str]:
        """List all namespaces an agent has access to.

        Args:
            agent_id: The agent.

        Returns:
            List of namespace IDs.
        """
        namespaces = []
        for ns_id, perms in self._permissions.items():
            if agent_id in perms:
                namespaces.append(ns_id)
        return namespaces
