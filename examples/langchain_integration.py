"""LangChain integration example.

Shows how to use AgentVault as a memory backend for LangChain agents.
Requires: pip install langchain langchain-openai
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from agentvault import AgentVault


def cleanup_demo_data():
    """Remove existing demo data to avoid duplicates."""
    demo_dir = Path.home() / ".agentvault"
    if demo_dir.exists():
        for pattern in ["*.db", "faiss/*.index", "faiss/*.json"]:
            for f in demo_dir.glob(pattern):
                f.unlink(missing_ok=True)


class AgentVaultLangChainMemory:
    """LangChain-compatible memory wrapper for AgentVault.

    Implements the interface expected by LangChain's ConversationChain
    and Agent classes for persistent memory.

    Usage with LangChain:
        from langchain.chains import ConversationChain
        from langchain_openai import ChatOpenAI

        vault_memory = AgentVaultLangChainMemory(agent_id="langchain-agent")

        chain = ConversationChain(
            llm=ChatOpenAI(),
            memory=vault_memory,
        )
    """

    memory_key: str = "history"

    def __init__(self, agent_id: str = "langchain-agent", top_k: int = 5):
        self.vault = AgentVault(agent_id=agent_id)
        self.top_k = top_k
        self.vault.initialize()

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load relevant memories based on the current input."""
        query = inputs.get("input", inputs.get("question", ""))
        if not query:
            return {self.memory_key: ""}

        result = self.vault.recall(query, top_k=self.top_k)
        memories_text = "\n".join(
            f"- {m.content}" for m in result.memories
        )
        return {self.memory_key: memories_text}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save the interaction as an episodic memory."""
        user_input = inputs.get("input", inputs.get("question", ""))
        ai_output = outputs.get("output", outputs.get("response", ""))

        if user_input and ai_output:
            self.vault.remember_episode(
                event=f"User: {user_input}\nAssistant: {ai_output}",
                context={"source": "langchain"},
                importance=0.5,
                tags=["conversation", "langchain"],
            )

    def clear(self) -> None:
        """Clear all memories."""
        self.vault.forget(importance_below=1.1)  # Delete all


async def demo():
    """Demonstrate the LangChain integration."""
    # Clean up previous demo data
    cleanup_demo_data()
    print("Cleaned up previous demo data.\n")

    memory = AgentVaultLangChainMemory(agent_id="langchain-demo")

    # Simulate a conversation
    memory.save_context(
        {"input": "What's the best way to deploy a Python app?"},
        {"output": "Use Docker containers with a CI/CD pipeline for production deployments."},
    )
    memory.save_context(
        {"input": "I prefer using Vercel for my projects"},
        {"output": "Noted! Vercel is great for frontend and serverless deployments."},
    )

    # Later, recall relevant context
    context = memory.load_memory_variables({"input": "How should I deploy?"})
    print("Recalled context for 'How should I deploy?':")
    print(context["history"])

    memory.vault.close()


if __name__ == "__main__":
    asyncio.run(demo())
