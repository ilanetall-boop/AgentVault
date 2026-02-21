"""CrewAI integration example.

Shows how to use AgentVault as persistent memory for CrewAI agents.
Requires: pip install crewai
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from agentvault import AgentVault
from agentvault.sdk.decorators import auto_remember, with_recall


def cleanup_demo_data():
    """Remove existing demo data to avoid duplicates."""
    demo_dir = Path.home() / ".agentvault"
    if demo_dir.exists():
        for pattern in ["*.db", "faiss/*.index", "faiss/*.json"]:
            for f in demo_dir.glob(pattern):
                f.unlink(missing_ok=True)


class CrewAIMemoryTool:
    """CrewAI-compatible tool that provides memory capabilities.

    Can be used as a tool in CrewAI agent definitions to give
    agents the ability to remember and recall information.

    Usage with CrewAI:
        from crewai import Agent, Task, Crew

        memory_tool = CrewAIMemoryTool(agent_id="researcher")

        agent = Agent(
            role="Research Analyst",
            goal="Research and remember key findings",
            tools=[memory_tool.remember_tool, memory_tool.recall_tool],
        )
    """

    def __init__(self, agent_id: str = "crewai-agent"):
        self.vault = AgentVault(agent_id=agent_id)
        self.vault.initialize()

    def remember(self, content: str, importance: float = 0.6) -> str:
        """Tool: Store a piece of information in long-term memory.

        Args:
            content: The information to remember.
            importance: How important this is (0-1).

        Returns:
            Confirmation message.
        """
        memory = self.vault.remember(
            content=content,
            importance=importance,
            tags=["crewai"],
            source="crewai-tool",
        )
        return f"Remembered: '{content}' (id={memory.id})"

    def recall(self, query: str, top_k: int = 5) -> str:
        """Tool: Search long-term memory for relevant information.

        Args:
            query: What to search for.
            top_k: Maximum results.

        Returns:
            Formatted string of recalled memories.
        """
        result = self.vault.recall(query, top_k=top_k)
        if not result.memories:
            return "No relevant memories found."

        lines = [f"Found {result.total_found} memories:"]
        for mem, score in zip(result.memories, result.relevance_scores):
            lines.append(f"  [{score:.2f}] {mem.content}")
        return "\n".join(lines)

    def close(self):
        """Clean up resources."""
        self.vault.close()


async def demo():
    """Demonstrate the CrewAI integration."""
    # Clean up previous demo data
    cleanup_demo_data()
    print("Cleaned up previous demo data.\n")

    tool = CrewAIMemoryTool(agent_id="crewai-demo")

    # Simulate a CrewAI agent workflow
    print("=== CrewAI Agent: Research Phase ===\n")

    # Agent remembers findings
    print(tool.remember("React 19 introduces Server Components as stable", importance=0.8))
    print(tool.remember("Next.js 15 uses Turbopack by default", importance=0.7))
    print(tool.remember("Vercel supports edge functions for low-latency APIs", importance=0.6))

    # Later, agent recalls relevant info
    print("\n=== CrewAI Agent: Writing Phase ===\n")
    print("Query: 'frontend framework features'")
    print(tool.recall("frontend framework features"))

    print("\nQuery: 'deployment and hosting'")
    print(tool.recall("deployment and hosting"))

    tool.close()


# Example with decorators
async def demo_with_decorators():
    """Demonstrate using decorators for automatic memory."""
    vault = AgentVault(agent_id="crewai-decorator-demo")
    vault.initialize()

    @auto_remember(vault, memory_type="episodic", tags=["task"])
    def execute_task(task_description: str) -> str:
        return f"Completed: {task_description}"

    @with_recall(vault, query_param="question", inject_as="context")
    def answer_question(question: str, context: list = None) -> str:
        context_str = "\n".join(context) if context else "No context"
        return f"Based on context:\n{context_str}\nAnswer to: {question}"

    # Execute tasks — automatically remembered
    execute_task("Analyzed competitor pricing data")
    execute_task("Created marketing strategy document")

    # Answer questions — automatically recalls context
    answer = answer_question("What tasks have been completed?")
    print(answer)

    vault.close()


if __name__ == "__main__":
    asyncio.run(demo())
