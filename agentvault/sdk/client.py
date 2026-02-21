"""AgentVault SDK client — the simple API that developers use.

Usage:
    from agentvault import AgentVault

    vault = AgentVault(agent_id="my-agent")
    vault.remember("The user prefers short answers", type="semantic", importance=0.8)
    memories = vault.recall("how does the user like responses?")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from agentvault.core.memory_manager import MemoryManager
from agentvault.core.types import (
    ConsolidationResult,
    Memory,
    MemoryType,
    RecallResult,
)

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an existing event loop — create a task
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class AgentVault:
    """The main SDK client for AgentVault.

    Provides a simple, sync-first API for storing and recalling
    memories. All operations also have async counterparts prefixed
    with 'a' (e.g., aremember, arecall).

    Example:
        vault = AgentVault(agent_id="my-agent")
        vault.remember("User prefers TypeScript", type="semantic")
        results = vault.recall("what language does user prefer?")
    """

    def __init__(
        self,
        agent_id: str = "default",
        db_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        chroma_path: Optional[str] = None,  # Backwards compatibility alias
        auto_initialize: bool = True,
    ) -> None:
        """Initialize AgentVault.

        Args:
            agent_id: Unique identifier for this agent.
            db_path: Path to SQLite database (default: ~/.agentvault/memories.db).
            faiss_path: Path to FAISS storage (default: ~/.agentvault/faiss).
            chroma_path: Deprecated alias for faiss_path.
            auto_initialize: Whether to auto-initialize on first operation.
        """
        self.agent_id = agent_id
        vector_path = faiss_path or chroma_path
        self._manager = MemoryManager(db_path=db_path, faiss_path=vector_path)
        self._initialized = False
        self._auto_initialize = auto_initialize

    async def _ensure_initialized(self) -> None:
        """Ensure the manager is initialized."""
        if not self._initialized:
            if self._auto_initialize:
                await self._manager.initialize()
                self._initialized = True
            else:
                raise RuntimeError(
                    "AgentVault not initialized. Call initialize() first or set auto_initialize=True."
                )

    # --- Initialization ---

    def initialize(self) -> None:
        """Initialize AgentVault (sync)."""
        _run_async(self.ainitialize())

    async def ainitialize(self) -> None:
        """Initialize AgentVault (async)."""
        await self._manager.initialize()
        self._initialized = True
        logger.info("AgentVault initialized for agent '%s'", self.agent_id)

    # --- Remember (write) ---

    def remember(
        self,
        content: str,
        type: Optional[str | MemoryType] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Memory:
        """Store a new memory (sync).

        Args:
            content: The text content to remember.
            type: Memory type ('episodic', 'semantic', 'procedural') or auto-detect.
            importance: Importance score 0-1 (auto-estimated if None).
            tags: Tags for categorization (auto-extracted if None).
            metadata: Additional metadata.
            source: What created this memory.

        Returns:
            The stored Memory object.
        """
        return _run_async(
            self.aremember(content, type, importance, tags, metadata, source)
        )

    async def aremember(
        self,
        content: str,
        type: Optional[str | MemoryType] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Memory:
        """Store a new memory (async)."""
        await self._ensure_initialized()

        memory_type: Optional[MemoryType] = None
        if isinstance(type, str):
            memory_type = MemoryType(type)
        elif isinstance(type, MemoryType):
            memory_type = type

        return await self._manager.remember(
            agent_id=self.agent_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            metadata=metadata,
            source=source,
        )

    def remember_episode(
        self,
        event: str,
        context: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
    ) -> Memory:
        """Store an episodic memory (sync).

        Args:
            event: Description of the event.
            context: Contextual information.
            importance: Importance score.
            tags: Tags for categorization.
            source: What created this memory.

        Returns:
            The stored Memory.
        """
        return _run_async(
            self.aremember_episode(event, context, importance, tags, source)
        )

    async def aremember_episode(
        self,
        event: str,
        context: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
    ) -> Memory:
        """Store an episodic memory (async)."""
        await self._ensure_initialized()
        return await self._manager.remember_episode(
            agent_id=self.agent_id,
            event=event,
            context=context,
            importance=importance,
            tags=tags,
            source=source,
        )

    def remember_procedure(
        self,
        name: str,
        steps: list[str],
        content: Optional[str] = None,
        learned_from: Optional[str] = None,
        importance: float = 0.6,
        tags: Optional[list[str]] = None,
    ) -> Memory:
        """Store a procedural memory (sync).

        Args:
            name: Name of the procedure.
            steps: Ordered list of steps.
            content: Optional text description.
            learned_from: Source conversation/interaction.
            importance: Importance score.
            tags: Tags for categorization.

        Returns:
            The stored Memory.
        """
        return _run_async(
            self.aremember_procedure(name, steps, content, learned_from, importance, tags)
        )

    async def aremember_procedure(
        self,
        name: str,
        steps: list[str],
        content: Optional[str] = None,
        learned_from: Optional[str] = None,
        importance: float = 0.6,
        tags: Optional[list[str]] = None,
    ) -> Memory:
        """Store a procedural memory (async)."""
        await self._ensure_initialized()
        return await self._manager.remember_procedure(
            agent_id=self.agent_id,
            name=name,
            steps=steps,
            content=content,
            learned_from=learned_from,
            importance=importance,
            tags=tags,
        )

    # --- Recall (read) ---

    def recall(
        self,
        query: str,
        top_k: int = 10,
        types: Optional[list[str]] = None,
        min_importance: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> RecallResult:
        """Recall relevant memories (sync).

        Args:
            query: The search query.
            top_k: Maximum number of results.
            types: Filter by memory types.
            min_importance: Minimum importance threshold.
            tags: Filter by tags.

        Returns:
            RecallResult with ranked memories.
        """
        return _run_async(
            self.arecall(query, top_k, types, min_importance, tags)
        )

    async def arecall(
        self,
        query: str,
        top_k: int = 10,
        types: Optional[list[str]] = None,
        min_importance: float = 0.0,
        tags: Optional[list[str]] = None,
    ) -> RecallResult:
        """Recall relevant memories (async)."""
        await self._ensure_initialized()

        memory_types: Optional[list[MemoryType]] = None
        if types:
            memory_types = [MemoryType(t) for t in types]

        return await self._manager.recall(
            agent_id=self.agent_id,
            query=query,
            top_k=top_k,
            types=memory_types,
            min_importance=min_importance,
            tags=tags,
        )

    # --- Forget (delete) ---

    def forget(
        self,
        older_than: Optional[str] = None,
        importance_below: Optional[float] = None,
    ) -> int:
        """Forget memories matching criteria (sync).

        Args:
            older_than: Duration string like '30d', '12h'.
            importance_below: Delete memories below this importance.

        Returns:
            Number of memories forgotten.
        """
        return _run_async(
            self.aforget(older_than, importance_below)
        )

    async def aforget(
        self,
        older_than: Optional[str] = None,
        importance_below: Optional[float] = None,
    ) -> int:
        """Forget memories matching criteria (async)."""
        await self._ensure_initialized()
        return await self._manager.forget(
            agent_id=self.agent_id,
            older_than=older_than,
            importance_below=importance_below,
        )

    # --- Share (multi-agent) ---

    def share(
        self,
        memory_id: str,
        with_agents: list[str],
    ) -> None:
        """Share a memory with other agents (sync).

        Args:
            memory_id: ID of the memory to share.
            with_agents: List of agent IDs to share with.
        """
        _run_async(self.ashare(memory_id, with_agents))

    async def ashare(
        self,
        memory_id: str,
        with_agents: list[str],
    ) -> None:
        """Share a memory with other agents (async)."""
        await self._ensure_initialized()
        memory = await self._manager.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} not found")
        memory.metadata["shared_with"] = with_agents
        await self._manager._store.update(memory)
        logger.info("Shared memory %s with agents %s", memory_id, with_agents)

    # --- Consolidation ---

    def consolidate(self) -> ConsolidationResult:
        """Run memory consolidation (sync).

        Returns:
            ConsolidationResult with statistics.
        """
        return _run_async(self.aconsolidate())

    async def aconsolidate(self) -> ConsolidationResult:
        """Run memory consolidation (async)."""
        await self._ensure_initialized()
        return await self._manager.consolidate(self.agent_id)

    # --- Utilities ---

    def count(self, memory_type: Optional[str] = None) -> int:
        """Count memories (sync)."""
        return _run_async(self.acount(memory_type))

    async def acount(self, memory_type: Optional[str] = None) -> int:
        """Count memories (async)."""
        await self._ensure_initialized()
        mt = MemoryType(memory_type) if memory_type else None
        return await self._manager.count(self.agent_id, mt)

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID (sync)."""
        return _run_async(self.aget(memory_id))

    async def aget(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID (async)."""
        await self._ensure_initialized()
        return await self._manager.get_memory(memory_id)

    def close(self) -> None:
        """Close all resources (sync)."""
        _run_async(self.aclose())

    async def aclose(self) -> None:
        """Close all resources (async)."""
        await self._manager.close()
        self._initialized = False

    # --- Context manager ---

    def __enter__(self) -> AgentVault:
        self.initialize()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> AgentVault:
        await self.ainitialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
