"""Performance benchmarks for AgentVault.

Measures:
- Write time (1 memory)
- Recall time (top 10 on 10K memories)
- Recall time (top 10 on 100K memories)
- Consolidation time (1000 memories)
- Memory usage
"""

from __future__ import annotations

import asyncio
import os
import resource
import sys
import tempfile
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvault.core.memory_manager import MemoryManager
from agentvault.core.types import MemoryType


AGENT_ID = "bench-agent"


async def benchmark_write(manager: MemoryManager, count: int = 100) -> float:
    """Benchmark write performance. Returns avg time per write in ms."""
    start = time.perf_counter()
    for i in range(count):
        await manager.remember(
            agent_id=AGENT_ID,
            content=f"Benchmark memory {i}: testing write performance with varied content #{i}",
            memory_type=MemoryType.SEMANTIC,
            importance=0.5,
            tags=["benchmark"],
        )
    elapsed = (time.perf_counter() - start) * 1000
    avg = elapsed / count
    return avg


async def benchmark_recall(manager: MemoryManager, top_k: int = 10) -> float:
    """Benchmark recall performance. Returns time in ms."""
    start = time.perf_counter()
    result = await manager.recall(
        agent_id=AGENT_ID,
        query="testing write performance benchmark",
        top_k=top_k,
    )
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed


async def benchmark_consolidation(manager: MemoryManager) -> float:
    """Benchmark consolidation. Returns time in ms."""
    start = time.perf_counter()
    result = await manager.consolidate(AGENT_ID)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Convert KB to MB (Linux)
    except Exception:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


async def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("AgentVault Performance Benchmarks")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench.db")
        faiss_path = os.path.join(tmpdir, "faiss")
        manager = MemoryManager(db_path=db_path, faiss_path=faiss_path)
        await manager.initialize()

        # Benchmark: Single write
        print("\n[1/5] Write performance (single memory)...")
        avg_write = await benchmark_write(manager, count=50)
        target_write = 10
        status = "PASS" if avg_write < target_write else "FAIL"
        print(f"  Avg write time: {avg_write:.2f}ms (target: <{target_write}ms) [{status}]")

        # Benchmark: Recall on current set
        print("\n[2/5] Recall performance (top 10 on current set)...")
        recall_time = await benchmark_recall(manager, top_k=10)
        target_recall = 50
        status = "PASS" if recall_time < target_recall else "FAIL"
        print(f"  Recall time: {recall_time:.2f}ms (target: <{target_recall}ms) [{status}]")

        # Benchmark: Scale to more memories
        print("\n[3/5] Scaling writes (500 more memories)...")
        avg_write_scaled = await benchmark_write(manager, count=500)
        print(f"  Avg write time at scale: {avg_write_scaled:.2f}ms")

        # Benchmark: Recall at scale
        print("\n[4/5] Recall at scale (top 10 on ~550 memories)...")
        recall_scaled = await benchmark_recall(manager, top_k=10)
        print(f"  Recall time: {recall_scaled:.2f}ms")

        # Benchmark: Consolidation
        print("\n[5/5] Consolidation performance...")
        consol_time = await benchmark_consolidation(manager)
        target_consol = 5000
        status = "PASS" if consol_time < target_consol else "FAIL"
        print(f"  Consolidation time: {consol_time:.2f}ms (target: <{target_consol}ms) [{status}]")

        # Memory usage
        try:
            mem_mb = get_memory_usage_mb()
            print(f"\n  Memory usage: {mem_mb:.1f}MB")
        except Exception:
            print("\n  Memory usage: (unable to measure)")

        await manager.close()

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
