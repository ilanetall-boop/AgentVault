# Quick Start Guide

## Installation

```bash
pip install agentvault
```

### Optional extras

```bash
# OpenAI embeddings
pip install agentvault[openai]

# PostgreSQL support
pip install agentvault[postgres]

# Qdrant vector store
pip install agentvault[qdrant]

# Everything
pip install agentvault[all]
```

## Basic Usage

### 1. Initialize

```python
from agentvault import AgentVault

# Zero config — uses SQLite + local embeddings
vault = AgentVault(agent_id="my-agent")
```

### 2. Store Memories

```python
# Auto-detect type and importance
vault.remember("The user prefers TypeScript")

# Explicit type and importance
vault.remember(
    "React 19 has server components",
    type="semantic",
    importance=0.8,
    tags=["react", "frontend"],
)

# Store an event
vault.remember_episode(
    event="Deployed v2.0 to production",
    context={"version": "2.0", "env": "production"},
)

# Store a procedure
vault.remember_procedure(
    name="deploy",
    steps=["npm run build", "docker push", "kubectl apply"],
)
```

### 3. Recall Memories

```python
# Simple recall
result = vault.recall("what does the user prefer?")
for memory in result.memories:
    print(memory.content)

# Filtered recall
result = vault.recall(
    "deployment steps",
    types=["procedural"],
    min_importance=0.5,
    top_k=3,
)

# Access metadata
print(f"Found {result.total_found} memories in {result.query_time_ms}ms")
```

### 4. Forget Memories

```python
# Forget old, unimportant memories
vault.forget(older_than="30d", importance_below=0.2)
```

### 5. Consolidate

```python
result = vault.consolidate()
print(f"Merged: {result.merged}")
print(f"Promoted: {result.promoted}")
print(f"Forgotten: {result.forgotten}")
```

## Async Usage

All operations have async counterparts prefixed with `a`:

```python
import asyncio

async def main():
    async with AgentVault(agent_id="my-agent") as vault:
        await vault.aremember("async memory")
        result = await vault.arecall("search")
        await vault.aforget(importance_below=0.1)

asyncio.run(main())
```

## Context Manager

```python
# Sync
with AgentVault(agent_id="my-agent") as vault:
    vault.remember("auto-cleanup on exit")

# Async
async with AgentVault(agent_id="my-agent") as vault:
    await vault.aremember("auto-cleanup on exit")
```

## Next Steps

- See [Architecture](architecture.md) for system design details
- See [API Reference](api_reference.md) for full API documentation
- Check out the `examples/` directory for integration examples
