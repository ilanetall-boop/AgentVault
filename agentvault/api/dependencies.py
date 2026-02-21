"""Shared dependencies for API routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agentvault.core.memory_manager import MemoryManager

# Global memory manager instance
_manager: Optional["MemoryManager"] = None


def set_manager(manager: "MemoryManager") -> None:
    """Set the global MemoryManager instance."""
    global _manager
    _manager = manager


def get_manager() -> "MemoryManager":
    """Get the global MemoryManager instance."""
    if _manager is None:
        raise RuntimeError("MemoryManager not initialized")
    return _manager
