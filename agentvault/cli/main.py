"""CLI interface for AgentVault using Typer."""

from __future__ import annotations

import asyncio
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="agentvault",
    help="AgentVault — Persistent Memory for AI Agents",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    db_path: str | None = typer.Option(None, help="SQLite database path"),
    faiss_path: str | None = typer.Option(None, help="FAISS storage path"),
) -> None:
    """Start the AgentVault API server."""
    console.print(f"[bold green]Starting AgentVault API on {host}:{port}[/bold green]")
    from agentvault.api.server import run_server

    run_server(host=host, port=port, db_path=db_path, faiss_path=faiss_path)


@app.command()
def stats(
    agent_id: str = typer.Argument(help="Agent ID to show stats for"),
    db_path: str | None = typer.Option(None, help="SQLite database path"),
    faiss_path: str | None = typer.Option(None, help="FAISS storage path"),
) -> None:
    """Show memory statistics for an agent."""

    async def _stats():
        from agentvault.core.memory_manager import MemoryManager
        from agentvault.core.types import MemoryType

        manager = MemoryManager(db_path=db_path, faiss_path=faiss_path)
        await manager.initialize()

        total = await manager.count(agent_id)
        episodic = await manager.count(agent_id, MemoryType.EPISODIC)
        semantic = await manager.count(agent_id, MemoryType.SEMANTIC)
        procedural = await manager.count(agent_id, MemoryType.PROCEDURAL)

        table = Table(title=f"Memory Stats for '{agent_id}'")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="magenta", justify="right")

        table.add_row("Episodic", str(episodic))
        table.add_row("Semantic", str(semantic))
        table.add_row("Procedural", str(procedural))
        table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")

        console.print(table)
        await manager.close()

    asyncio.run(_stats())


@app.command()
def recall(
    agent_id: str = typer.Argument(help="Agent ID"),
    query: str = typer.Argument(help="Search query"),
    top_k: int = typer.Option(10, help="Maximum results"),
    db_path: str | None = typer.Option(None, help="SQLite database path"),
    faiss_path: str | None = typer.Option(None, help="FAISS storage path"),
) -> None:
    """Search memories for an agent."""

    async def _recall():
        from agentvault.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=db_path, faiss_path=faiss_path)
        await manager.initialize()

        result = await manager.recall(agent_id=agent_id, query=query, top_k=top_k)

        if not result.memories:
            console.print("[yellow]No memories found.[/yellow]")
        else:
            table = Table(title=f"Recall results ({result.query_time_ms:.1f}ms)")
            table.add_column("Score", style="green", justify="right")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="white")
            table.add_column("Importance", style="magenta", justify="right")

            for memory, score in zip(result.memories, result.relevance_scores, strict=False):
                table.add_row(
                    f"{score:.3f}",
                    memory.type.value,
                    memory.content[:80],
                    f"{memory.importance:.2f}",
                )

            console.print(table)
            console.print(f"Total found: {result.total_found}")

        await manager.close()

    asyncio.run(_recall())


@app.command()
def consolidate(
    agent_id: str = typer.Argument(help="Agent ID"),
    db_path: str | None = typer.Option(None, help="SQLite database path"),
    faiss_path: str | None = typer.Option(None, help="FAISS storage path"),
) -> None:
    """Run memory consolidation for an agent."""

    async def _consolidate():
        from agentvault.core.memory_manager import MemoryManager

        manager = MemoryManager(db_path=db_path, faiss_path=faiss_path)
        await manager.initialize()

        result = await manager.consolidate(agent_id)

        table = Table(title=f"Consolidation Results for '{agent_id}'")
        table.add_column("Operation", style="cyan")
        table.add_column("Count", style="magenta", justify="right")

        table.add_row("Merged", str(result.merged))
        table.add_row("Promoted", str(result.promoted))
        table.add_row("Forgotten", str(result.forgotten))
        table.add_row("Reinforced", str(result.reinforced))
        table.add_row("[bold]Total Processed[/bold]", f"[bold]{result.total_processed}[/bold]")
        table.add_row("Duration", f"{result.duration_ms:.1f}ms")

        console.print(table)
        await manager.close()

    asyncio.run(_consolidate())


@app.command()
def version() -> None:
    """Show the AgentVault version."""
    from agentvault import __version__

    console.print(f"AgentVault v{__version__}")


if __name__ == "__main__":
    app()
