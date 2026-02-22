"""Decorators for easy memory integration.

Provides @remember and @recall decorators for automatic
memory extraction from function calls and conversations.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def auto_remember(
    vault: Any,
    memory_type: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
) -> Callable:
    """Decorator that automatically stores function input/output as memories.

    Wraps a function so that its arguments and return value are
    automatically remembered by the vault.

    Args:
        vault: An AgentVault instance.
        memory_type: Type of memory to create.
        importance: Importance score.
        tags: Tags for the memory.

    Returns:
        Decorated function.

    Usage:
        @auto_remember(vault, memory_type="episodic")
        async def handle_message(message: str) -> str:
            return process(message)
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await func(*args, **kwargs)
                content = _build_memory_content(func, args, kwargs, result)
                try:
                    await vault.aremember(
                        content=content,
                        type=memory_type,
                        importance=importance,
                        tags=tags,
                        source=f"auto_remember:{func.__name__}",
                    )
                except Exception:
                    logger.warning(
                        "Failed to auto-remember for %s", func.__name__, exc_info=True
                    )
                return result

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = func(*args, **kwargs)
                content = _build_memory_content(func, args, kwargs, result)
                try:
                    vault.remember(
                        content=content,
                        type=memory_type,
                        importance=importance,
                        tags=tags,
                        source=f"auto_remember:{func.__name__}",
                    )
                except Exception:
                    logger.warning(
                        "Failed to auto-remember for %s", func.__name__, exc_info=True
                    )
                return result

            return sync_wrapper

    return decorator


def with_recall(
    vault: Any,
    query_param: str = "query",
    top_k: int = 5,
    inject_as: str = "memories",
) -> Callable:
    """Decorator that injects recalled memories into function kwargs.

    Before calling the function, recalls relevant memories based on
    a specified parameter and injects them as an additional kwarg.

    Args:
        vault: An AgentVault instance.
        query_param: Name of the parameter to use as recall query.
        top_k: Number of memories to recall.
        inject_as: Name of the kwarg to inject memories into.

    Returns:
        Decorated function.

    Usage:
        @with_recall(vault, query_param="message", inject_as="context")
        async def respond(message: str, context: list = None) -> str:
            # context will contain recalled memories
            return generate_response(message, context)
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                query = _extract_query(func, args, kwargs, query_param)
                if query:
                    try:
                        result = await vault.arecall(query, top_k=top_k)
                        kwargs[inject_as] = [m.content for m in result.memories]
                    except Exception:
                        logger.warning(
                            "Failed to recall for %s", func.__name__, exc_info=True
                        )
                        kwargs[inject_as] = []
                return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                query = _extract_query(func, args, kwargs, query_param)
                if query:
                    try:
                        result = vault.recall(query, top_k=top_k)
                        kwargs[inject_as] = [m.content for m in result.memories]
                    except Exception:
                        logger.warning(
                            "Failed to recall for %s", func.__name__, exc_info=True
                        )
                        kwargs[inject_as] = []
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def _build_memory_content(
    func: Callable,
    args: tuple,
    kwargs: dict,
    result: Any,
) -> str:
    """Build a memory content string from function call details."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    parts = [f"Function '{func.__name__}' called"]

    input_parts = []
    for i, arg in enumerate(args):
        param_name = params[i] if i < len(params) else f"arg{i}"
        if isinstance(arg, str) and len(arg) < 500:
            input_parts.append(f"{param_name}={arg!r}")
    for key, val in kwargs.items():
        if isinstance(val, str) and len(val) < 500:
            input_parts.append(f"{key}={val!r}")

    if input_parts:
        parts.append(f"with inputs: {', '.join(input_parts)}")

    if result is not None:
        result_str = str(result)
        if len(result_str) < 500:
            parts.append(f"returned: {result_str}")

    return " | ".join(parts)


def _extract_query(
    func: Callable,
    args: tuple,
    kwargs: dict,
    query_param: str,
) -> str | None:
    """Extract the query value from function arguments."""
    if query_param in kwargs:
        val = kwargs[query_param]
        return str(val) if val is not None else None

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    try:
        idx = params.index(query_param)
        if idx < len(args):
            return str(args[idx])
    except ValueError:
        pass

    return None
