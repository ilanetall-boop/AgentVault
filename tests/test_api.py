"""Tests for the FastAPI REST API."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agentvault.api.server import create_app
from agentvault.core.memory_manager import MemoryManager


@pytest_asyncio.fixture
async def app(tmp_dir):
    """Create and initialize a test FastAPI app."""
    # Create and initialize manager
    manager = MemoryManager(
        db_path=f"{tmp_dir}/test.db",
        faiss_path=f"{tmp_dir}/faiss"
    )
    await manager.initialize()

    # Create the app with pre-initialized manager
    app = create_app(manager=manager)

    yield app

    # Cleanup
    await manager.close()


@pytest.mark.asyncio
class TestAPI:
    """Tests for the REST API endpoints."""

    async def test_health_check(self, app):
        """Test the health check endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    async def test_create_memory(self, app):
        """Test creating a memory via API."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/memories",
                json={
                    "agent_id": "test-agent",
                    "content": "API test memory",
                    "type": "semantic",
                    "importance": 0.7,
                    "tags": ["test"],
                },
            )
            assert response.status_code == 201
            data = response.json()
            assert data["content"] == "API test memory"
            assert data["id"] is not None

    async def test_get_memory(self, app):
        """Test retrieving a memory by ID."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create first
            create_resp = await client.post(
                "/api/v1/memories",
                json={
                    "agent_id": "test-agent",
                    "content": "Memory to retrieve",
                    "type": "semantic",
                },
            )
            memory_id = create_resp.json()["id"]

            # Retrieve
            response = await client.get(f"/api/v1/memories/{memory_id}")
            assert response.status_code == 200
            assert response.json()["content"] == "Memory to retrieve"

    async def test_search_memories(self, app):
        """Test searching memories."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create some memories
            await client.post(
                "/api/v1/memories",
                json={
                    "agent_id": "test-agent",
                    "content": "Python is great for AI",
                    "type": "semantic",
                    "importance": 0.8,
                },
            )

            # Search
            response = await client.post(
                "/api/v1/search",
                json={
                    "agent_id": "test-agent",
                    "query": "programming language for AI",
                    "top_k": 5,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "memories" in data
            assert "query_time_ms" in data

    async def test_delete_memory(self, app):
        """Test deleting a memory."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create
            create_resp = await client.post(
                "/api/v1/memories",
                json={
                    "agent_id": "test-agent",
                    "content": "Memory to delete",
                    "type": "semantic",
                },
            )
            memory_id = create_resp.json()["id"]

            # Delete
            response = await client.delete(f"/api/v1/memories/{memory_id}")
            assert response.status_code == 204

            # Verify deleted
            response = await client.get(f"/api/v1/memories/{memory_id}")
            assert response.status_code == 404

    async def test_get_nonexistent_memory(self, app):
        """Test getting a non-existent memory returns 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/memories/nonexistent-id")
            assert response.status_code == 404

    async def test_list_agent_memories(self, app):
        """Test listing memories for an agent."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create memories
            for i in range(3):
                await client.post(
                    "/api/v1/memories",
                    json={
                        "agent_id": "test-agent",
                        "content": f"List test memory {i}",
                        "type": "semantic",
                    },
                )

            response = await client.get("/api/v1/agents/test-agent/memories")
            assert response.status_code == 200
            data = response.json()
            assert len(data["memories"]) >= 3
