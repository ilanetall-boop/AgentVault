"""Multi-agent team example — 3 agents sharing memory.

Demonstrates how multiple agents can collaborate through
shared memory namespaces with permission control.
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from agentvault import AgentVault
from agentvault.core.memory_manager import MemoryManager
from agentvault.multi_agent.shared_memory import SharedMemoryManager


def cleanup_demo_data():
    """Remove existing demo data to avoid duplicates."""
    demo_dir = Path.home() / ".agentvault"
    if demo_dir.exists():
        for pattern in ["*.db", "faiss/*.index", "faiss/*.json"]:
            for f in demo_dir.glob(pattern):
                f.unlink(missing_ok=True)


async def main():
    # Clean up previous demo data
    cleanup_demo_data()
    print("Cleaned up previous demo data.\n")

    # Create a shared memory manager
    manager = MemoryManager()
    await manager.initialize()
    shared = SharedMemoryManager(manager)

    # Create 3 agents
    researcher = AgentVault(agent_id="researcher")
    coder = AgentVault(agent_id="coder")
    reviewer = AgentVault(agent_id="reviewer")

    await researcher.ainitialize()
    await coder.ainitialize()
    await reviewer.ainitialize()

    # Create a shared namespace for the team
    namespace = shared.create_namespace(
        name="project-alpha",
        owner_agent_id="researcher",
        member_agents={
            "coder": "write",
            "reviewer": "read",
        },
    )
    print(f"Created namespace: {namespace.name} (id={namespace.namespace_id})")

    # Researcher discovers facts and shares them
    memory = await researcher.aremember(
        "The API must use OAuth 2.0 for authentication",
        type="semantic",
        importance=0.9,
        tags=["requirements", "auth"],
    )
    await shared.share_memory(memory.id, namespace.namespace_id, "researcher")
    print(f"Researcher shared: {memory.content}")

    memory2 = await researcher.aremember(
        "The database should use PostgreSQL in production",
        type="semantic",
        importance=0.8,
        tags=["requirements", "database"],
    )
    await shared.share_memory(memory2.id, namespace.namespace_id, "researcher")
    print(f"Researcher shared: {memory2.content}")

    # Coder can write to the shared namespace
    coder_memory = await coder.aremember(
        "Implemented OAuth 2.0 flow using authlib library",
        type="episodic",
        importance=0.7,
        tags=["implementation", "auth"],
    )
    print(f"Coder remembered: {coder_memory.content}")

    # Reviewer can read from the shared namespace
    shared_results = await shared.recall_shared(
        namespace_id=namespace.namespace_id,
        agent_id="reviewer",
        query="authentication requirements",
        top_k=5,
    )
    print(f"\nReviewer queried shared memories about auth:")
    for mem, score in zip(shared_results.memories, shared_results.relevance_scores):
        print(f"  [{score:.3f}] {mem.content}")

    # Reviewer cannot write (read-only permission)
    try:
        await shared.share_memory(coder_memory.id, namespace.namespace_id, "reviewer")
    except PermissionError as e:
        print(f"\nPermission denied (expected): {e}")

    # Clean up
    await researcher.aclose()
    await coder.aclose()
    await reviewer.aclose()
    await manager.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
