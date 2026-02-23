"""Tests for API middleware — auth, rate limiting, logging."""

import os
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agentvault.api.server import create_app
from agentvault.core.memory_manager import MemoryManager


@pytest_asyncio.fixture
async def app_with_key(tmp_dir):
    """App with API key auth enabled."""
    manager = MemoryManager(
        db_path=f"{tmp_dir}/test.db",
        faiss_path=f"{tmp_dir}/faiss",
    )
    await manager.initialize()

    with patch.dict(os.environ, {"AGENTVAULT_API_KEY": "test-secret-key"}):
        # Need to re-import middleware to pick up env var
        import agentvault.api.middleware as mw
        original_key = mw._API_KEY
        mw._API_KEY = "test-secret-key"
        try:
            app = create_app(manager=manager)
            yield app
        finally:
            mw._API_KEY = original_key
            await manager.close()


@pytest_asyncio.fixture
async def app_no_key(tmp_dir):
    """App without API key (local mode)."""
    manager = MemoryManager(
        db_path=f"{tmp_dir}/test.db",
        faiss_path=f"{tmp_dir}/faiss",
    )
    await manager.initialize()

    import agentvault.api.middleware as mw
    original_key = mw._API_KEY
    mw._API_KEY = None
    try:
        app = create_app(manager=manager)
        yield app
    finally:
        mw._API_KEY = original_key
        await manager.close()


@pytest.mark.asyncio
class TestApiKeyMiddleware:
    """Tests for API key authentication."""

    async def test_health_always_public(self, app_with_key):
        """Health endpoint is accessible without API key."""
        transport = ASGITransport(app=app_with_key)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200

    async def test_valid_key_allows_request(self, app_with_key):
        """Valid API key grants access."""
        transport = ASGITransport(app=app_with_key)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/agents",
                headers={"X-API-Key": "test-secret-key"},
            )
            assert resp.status_code == 200

    async def test_invalid_key_returns_401(self, app_with_key):
        """Invalid API key returns 401."""
        transport = ASGITransport(app=app_with_key)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/agents",
                headers={"X-API-Key": "wrong-key"},
            )
            assert resp.status_code == 401

    async def test_missing_key_returns_401(self, app_with_key):
        """Missing API key header returns 401 when auth is required."""
        transport = ASGITransport(app=app_with_key)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/agents")
            assert resp.status_code == 401

    async def test_no_key_mode_allows_all(self, app_no_key):
        """Without API key configured, all requests are allowed."""
        transport = ASGITransport(app=app_no_key)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/agents")
            assert resp.status_code == 200


@pytest.mark.asyncio
class TestRequestLogging:
    """Tests for request logging middleware."""

    async def test_response_time_header(self, app_no_key):
        """Response includes X-Response-Time-Ms header."""
        transport = ASGITransport(app=app_no_key)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert "X-Response-Time-Ms" in resp.headers
            time_ms = float(resp.headers["X-Response-Time-Ms"])
            assert time_ms >= 0


@pytest.mark.asyncio
class TestStructuredStore:
    """Tests for structured store via API (duration parsing, etc)."""

    async def test_parse_duration_via_forget(self, app_no_key):
        """Duration parsing works through the forget API endpoint."""
        transport = ASGITransport(app=app_no_key)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a memory first
            await client.post(
                "/api/v1/memories",
                json={
                    "agent_id": "test-agent",
                    "content": "test memory",
                    "type": "semantic",
                },
            )
            # Forget with duration filter (should not error)
            resp = await client.delete(
                "/api/v1/agents/test-agent/memories",
                params={"older_than": "30d"},
            )
            assert resp.status_code == 200
