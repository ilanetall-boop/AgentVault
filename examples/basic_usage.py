"""Basic usage example for AgentVault.

Shows the simplest way to store and recall memories.
"""

import asyncio
import shutil
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from agentvault import AgentVault


def cleanup_demo_data():
    """Remove existing demo data to avoid duplicates."""
    demo_dir = Path.home() / ".agentvault"
    if demo_dir.exists():
        # Remove FAISS and SQLite files
        for pattern in ["*.db", "faiss/*.index", "faiss/*.json"]:
            for f in demo_dir.glob(pattern):
                f.unlink(missing_ok=True)
        # Remove faiss directory if empty
        faiss_dir = demo_dir / "faiss"
        if faiss_dir.exists() and not any(faiss_dir.iterdir()):
            faiss_dir.rmdir()


async def main():
    # Clean up previous demo data
    cleanup_demo_data()
    print("Cleaned up previous demo data.\n")

    # Initialize with default settings (SQLite + local embeddings)
    async with AgentVault(agent_id="my-agent") as vault:
        # Store different types of memories
        await vault.aremember(
            "The user prefers short, concise responses",
            type="semantic",
            importance=0.8,
            tags=["preference", "style"],
        )

        await vault.aremember_episode(
            event="User asked to refactor the authentication module",
            context={"file": "auth.py", "action": "refactor"},
            importance=0.7,
        )

        await vault.aremember_procedure(
            name="deploy_to_vercel",
            steps=[
                "npm run build",
                "npx vercel deploy --prod",
                "Verify deployment at https://app.vercel.com",
            ],
            importance=0.8,
            tags=["deploy", "vercel"],
        )

        # Recall relevant memories
        result = await vault.arecall(
            "How does the user like their responses?",
            top_k=5,
        )
        print(f"--- Recall: 'How does the user like responses?' ---")
        print(f"Found {result.total_found} memories in {result.query_time_ms:.1f}ms\n")
        for memory, score in zip(result.memories, result.relevance_scores):
            print(f"  [{score:.3f}] [{memory.type.value}] {memory.content}")

        # Recall procedures
        result = await vault.arecall(
            "How to deploy?",
            types=["procedural"],
            top_k=3,
        )
        print(f"\n--- Recall: 'How to deploy?' ---")
        for memory, score in zip(result.memories, result.relevance_scores):
            print(f"  [{score:.3f}] {memory.content}")

        # Count memories
        total = await vault.acount()
        print(f"\nTotal memories stored: {total}")

        # Forget old, unimportant memories
        forgotten = await vault.aforget(importance_below=0.1)
        print(f"Forgotten {forgotten} low-importance memories")


if __name__ == "__main__":
    asyncio.run(main())
