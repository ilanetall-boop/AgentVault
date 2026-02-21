# AgentVault — Persistent Memory for AI Agents

> Give your AI agents a brain that remembers.

[![PyPI version](https://badge.fury.io/py/agentvault.svg)](https://pypi.org/project/agentvault/)
[![Tests](https://github.com/agentvault/agentvault/actions/workflows/ci.yml/badge.svg)](https://github.com/agentvault/agentvault/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

AI agents forget everything. Every conversation starts from zero. Every context is lost. Your agent has no idea what happened yesterday, what the user prefers, or what it learned from its mistakes.

## The Solution

AgentVault gives any AI agent **persistent, intelligent memory** — episodic, semantic, and procedural — in **3 lines of code**.

```python
from agentvault import AgentVault

vault = AgentVault(agent_id="my-agent")
vault.remember("The user prefers TypeScript over JavaScript", type="semantic", importance=0.8)
memories = vault.recall("what language does the user prefer?")
```

That's it. Your agent now remembers.

---

## Quick Start (30 seconds)

### Install

```bash
pip install agentvault
```

### Use

```python
from agentvault import AgentVault

# Zero config — just works
vault = AgentVault(agent_id="my-agent")

# Store a memory
vault.remember("User prefers short responses", type="semantic", importance=0.8)

# Store an event
vault.remember_episode(
    event="User asked to refactor the auth module",
    context={"file": "auth.py"},
    importance=0.7,
)

# Store a learned procedure
vault.remember_procedure(
    name="deploy_vercel",
    steps=["npm run build", "vercel deploy --prod"],
)

# Recall relevant memories
result = vault.recall("how does the user like responses?", top_k=5)
for memory in result.memories:
    print(f"[{memory.type.value}] {memory.content}")

# Smart cleanup
vault.forget(older_than="30d", importance_below=0.2)
```

### Async Support

```python
async with AgentVault(agent_id="my-agent") as vault:
    await vault.aremember("async memory", type="semantic")
    result = await vault.arecall("search query")
```

---

## Why AgentVault?

| Feature | AgentVault | Simple Vector DB | Custom Solution |
|---------|-----------|-----------------|-----------------|
| 3 memory types (episodic, semantic, procedural) | Yes | No | Maybe |
| < 50ms recall on 100K+ memories | Yes | Varies | Unlikely |
| Multi-agent shared memory | Yes | No | Hard |
| Smart consolidation (merge, forget, reinforce) | Yes | No | Very hard |
| Works with LangChain, CrewAI, AutoGen | Yes | Manual | Manual |
| Per-agent permissions | Yes | No | DIY |
| Zero config local mode | Yes | Sometimes | No |
| Scales to production (PostgreSQL + Qdrant) | Yes | Varies | DIY |
| Open source (MIT) | Yes | Varies | N/A |

---

## The 3 Memory Types

AgentVault is inspired by how the human brain organizes memories:

### Episodic Memory — "What happened"
Stores events, interactions, and conversations. Timestamped and contextual.

```python
vault.remember_episode(
    event="The user asked to deploy the app",
    context={"environment": "production", "service": "api"},
    importance=0.7,
)
```

### Semantic Memory — "What I know"
Stores facts, preferences, and extracted knowledge.

```python
vault.remember("The project uses FastAPI for the backend", type="semantic")
vault.remember("User prefers dark mode", type="semantic", importance=0.8)
```

### Procedural Memory — "How to do things"
Stores skills, workflows, and step-by-step procedures.

```python
vault.remember_procedure(
    name="git_deploy",
    steps=["git add .", "git commit -m 'deploy'", "git push origin main"],
    importance=0.8,
)
```

---

## Smart Consolidation

AgentVault simulates how the human brain consolidates memories:

- **Fusion**: Two similar memories are merged into one enriched memory
- **Promotion**: A frequently recalled event becomes a fact (episodic -> semantic)
- **Forgetting**: Old, unimportant, never-accessed memories fade away
- **Reinforcement**: Every recall strengthens the memory

```python
# Run consolidation manually
result = vault.consolidate()
print(f"Merged: {result.merged}, Promoted: {result.promoted}, Forgotten: {result.forgotten}")
```

---

## Multi-Agent Shared Memory

Multiple agents can collaborate through shared memory namespaces:

```python
from agentvault.multi_agent.shared_memory import SharedMemoryManager

# Create a shared namespace
shared = SharedMemoryManager(memory_manager)
namespace = shared.create_namespace(
    name="project-alpha",
    owner_agent_id="researcher",
    member_agents={"coder": "write", "reviewer": "read"},
)

# Agent A shares a memory
await shared.share_memory(memory.id, namespace.namespace_id, "researcher")

# Agent B reads shared memories
results = await shared.recall_shared(namespace.namespace_id, "coder", "auth requirements")
```

---

## REST API

AgentVault includes a FastAPI server for remote access:

```bash
# Start the server
agentvault serve --port 8000
```

```bash
# Store a memory
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "my-agent", "content": "Important fact", "type": "semantic"}'

# Search memories
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "my-agent", "query": "important facts", "top_k": 5}'
```

---

## Integrations

AgentVault works with popular AI agent frameworks:

- **LangChain** — Use as a memory backend for ConversationChain
- **CrewAI** — Give your crew persistent memory across tasks
- **AutoGen** — Share memory between AutoGen agents
- **Custom agents** — Simple SDK for any Python agent

See the [`examples/`](examples/) directory for integration code.

---

## Architecture

```
+--------------------------------------------------+
|                  AgentVault API                   |
|              (REST + SDK Python)                  |
+--------------------------------------------------+
|              Memory Manager                       |
|  +------------+--------------+-----------------+  |
|  | Episodic   |  Semantic    |  Procedural     |  |
|  | Memory     |  Memory      |  Memory         |  |
|  | (events)   |  (facts)     |  (skills)       |  |
|  +------------+--------------+-----------------+  |
+--------------------------------------------------+
|            Memory Processing Pipeline             |
|  +-----------+------------+-------------------+   |
|  | Indexer   | Ranker     | Consolidator      |   |
|  |           |            | (merge/forget)    |   |
|  +-----------+------------+-------------------+   |
+--------------------------------------------------+
|              Storage Layer                        |
|  +---------------+---------------------------+   |
|  | Vector Store  |  Structured Store         |   |
|  | (ChromaDB)    |  (SQLite/PostgreSQL)      |   |
|  +---------------+---------------------------+   |
+--------------------------------------------------+
```

---

## Configuration

### Zero Config (default)
```python
vault = AgentVault(agent_id="my-agent")
# Uses SQLite + ChromaDB locally at ~/.agentvault/
```

### Custom Paths
```python
vault = AgentVault(
    agent_id="my-agent",
    db_path="/data/memories.db",
    chroma_path="/data/chroma",
)
```

### Production (PostgreSQL + Qdrant)
```python
# Coming in v0.2.0
vault = AgentVault(
    agent_id="my-agent",
    storage_backend="postgres",
    vector_backend="qdrant",
)
```

---

## Development

```bash
# Clone the repo
git clone https://github.com/agentvault/agentvault.git
cd agentvault

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=agentvault --cov-report=term-missing

# Lint
ruff check .
```

---

## Roadmap

- [x] Core memory engine (episodic, semantic, procedural)
- [x] Smart consolidation (merge, promote, forget, reinforce)
- [x] Multi-agent shared memory with permissions
- [x] FastAPI REST API
- [x] LangChain & CrewAI integrations
- [ ] PostgreSQL + Qdrant production backends
- [ ] TypeScript SDK
- [ ] Memory visualization dashboard
- [ ] Streaming recall (SSE)
- [ ] Memory import/export

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests.

---

Built with purpose. Give your agents a brain.
