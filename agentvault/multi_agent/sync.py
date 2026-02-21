"""Synchronization utilities for multi-agent memory operations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MemoryEvent:
    """Represents a memory change event for synchronization."""

    def __init__(
        self,
        event_type: str,
        namespace_id: str,
        agent_id: str,
        memory_id: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        self.event_type = event_type
        self.namespace_id = namespace_id
        self.agent_id = agent_id
        self.memory_id = memory_id
        self.data = data or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event to a dictionary."""
        return {
            "event_type": self.event_type,
            "namespace_id": self.namespace_id,
            "agent_id": self.agent_id,
            "memory_id": self.memory_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class SyncManager:
    """Manages event-driven synchronization between agents.

    When an agent writes to a shared namespace, all other agents
    with access are notified via async event queues.

    Conflict resolution uses last-write-wins with full history.
    """

    def __init__(self) -> None:
        # {agent_id: asyncio.Queue of MemoryEvents}
        self._queues: dict[str, asyncio.Queue[MemoryEvent]] = {}
        # History of all events for conflict resolution
        self._history: list[MemoryEvent] = []

    def register_agent(self, agent_id: str) -> None:
        """Register an agent for event notifications.

        Args:
            agent_id: The agent to register.
        """
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
            logger.info("Registered agent %s for sync", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from event notifications.

        Args:
            agent_id: The agent to unregister.
        """
        self._queues.pop(agent_id, None)
        logger.info("Unregistered agent %s from sync", agent_id)

    async def publish(
        self,
        event: MemoryEvent,
        target_agents: Optional[list[str]] = None,
    ) -> None:
        """Publish a memory event to subscribed agents.

        Args:
            event: The event to publish.
            target_agents: Specific agents to notify (all if None).
        """
        self._history.append(event)

        targets = target_agents or list(self._queues.keys())
        for agent_id in targets:
            if agent_id == event.agent_id:
                continue  # Don't notify the sender
            queue = self._queues.get(agent_id)
            if queue is not None:
                await queue.put(event)
                logger.debug(
                    "Published %s event to agent %s",
                    event.event_type,
                    agent_id,
                )

    async def poll(
        self,
        agent_id: str,
        timeout: float = 0.1,
    ) -> Optional[MemoryEvent]:
        """Poll for new events for an agent (non-blocking).

        Args:
            agent_id: The agent polling for events.
            timeout: How long to wait for an event (seconds).

        Returns:
            A MemoryEvent if available, else None.
        """
        queue = self._queues.get(agent_id)
        if queue is None:
            return None
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def drain(self, agent_id: str) -> list[MemoryEvent]:
        """Drain all pending events for an agent.

        Args:
            agent_id: The agent to drain events for.

        Returns:
            List of all pending events.
        """
        events: list[MemoryEvent] = []
        queue = self._queues.get(agent_id)
        if queue is None:
            return events
        while not queue.empty():
            try:
                event = queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break
        return events

    def get_history(
        self,
        namespace_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Get event history for conflict resolution.

        Args:
            namespace_id: Filter by namespace.
            memory_id: Filter by memory ID.
            limit: Maximum events to return.

        Returns:
            List of historical events, most recent first.
        """
        filtered = self._history
        if namespace_id:
            filtered = [e for e in filtered if e.namespace_id == namespace_id]
        if memory_id:
            filtered = [e for e in filtered if e.memory_id == memory_id]
        return list(reversed(filtered[-limit:]))

    def resolve_conflict(
        self,
        memory_id: str,
        namespace_id: str,
    ) -> Optional[MemoryEvent]:
        """Resolve a conflict using last-write-wins strategy.

        Args:
            memory_id: The contested memory ID.
            namespace_id: The namespace.

        Returns:
            The winning event (most recent write).
        """
        writes = [
            e
            for e in self._history
            if e.memory_id == memory_id
            and e.namespace_id == namespace_id
            and e.event_type in ("memory_created", "memory_updated", "memory_shared")
        ]
        if not writes:
            return None
        return max(writes, key=lambda e: e.timestamp)
