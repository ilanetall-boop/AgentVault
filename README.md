# AgentVault — Persistent Memory for AI Agents

**Give your AI agents a brain that remembers.**

AgentVault is an open-source memory engine that gives any AI agent persistent, intelligent memory — episodic, semantic, and procedural — in 3 lines of code.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

AI agents forget everything. Every conversation starts from zero. Your agent doesn't remember your architecture, your preferences, your past bugs, or your decisions.

## The Solution

```python
from agentvault import AgentVault

vault = AgentVault(agent_id="my-agent")

# Store a memory
vault.remember("User prefers TypeScript over JavaScript", type="semantic", importance=0.8)

# Recall relevant memories
memories = vault.recall("What language does the user prefer?", top_k=5)
# -> [0.72] User prefers TypeScript over JavaScript
```

That's it. Your agent now remembers.

---

## Features

- **3 Memory Types** — Like the human brain:
  - **Episodic**: Events and interactions (*"User asked to refactor auth module on March 15"*)
  - **Semantic**: Facts and knowledge (*"User prefers short responses"*)
  - **Procedural**: Skills and how-tos (*"To deploy: npm run build -> vercel deploy"*)

- **Fast** — < 50ms recall on thousands of memories (FAISS-powered vector search)

- **Multi-Agent** — Agents can share memories with permissions and namespaces

- **Smart Consolidation** — Automatically merges duplicates, forgets irrelevant memories, reinforces important ones

- **Zero Config** — Works out of the box with SQLite + FAISS locally. No external services needed.

- **REST API** — Full FastAPI server for production deployments

---

## Quick Start

### Install

```bash
pip install -e .
```

### Basic Usage

```python
import asyncio
from agentvault import AgentVault

async def main():
    vault = AgentVault(agent_id="my-agent")
    await vault.initialize()

    # Remember facts
    await vault.remember("The project uses FastAPI and PostgreSQL", type="semantic")

    # Remember events
    await vault.remember_episode(
        event="Fixed Unicode encoding bug on Windows",
        context={"file": "utils.py", "solution": "sys.stdout.reconfigure(encoding='utf-8')"}
    )

    # Remember procedures
    await vault.remember_procedure(
        name="run_tests",
        steps=["cd project/", "pytest tests/ -v"],
        learned_from="debugging session"
    )

    # Recall
    results = await vault.recall("How to run tests?")
    for memory, score in zip(results.memories, results.relevance_scores):
        print(f"  [{score:.2f}] {memory.content}")

asyncio.run(main())
```

### Multi-Agent Shared Memory

```python
from agentvault import AgentVault

researcher = AgentVault(agent_id="researcher")
coder = AgentVault(agent_id="coder")

# Researcher shares a finding
await researcher.remember("API must use OAuth 2.0", type="semantic")
await researcher.share("API must use OAuth 2.0", with_agents=["coder"])

# Coder can now recall shared memories
results = await coder.recall("authentication requirements", include_shared=True)
# -> [0.71] API must use OAuth 2.0
```

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
|  | FAISS         |  SQLite / PostgreSQL       |   |
|  | (vectors)     |  (structured data)         |   |
|  +---------------+---------------------------+   |
+--------------------------------------------------+
```

---

## Performance

| Metric | Result | Target |
|--------|--------|--------|
| Write (single memory) | ~30ms | < 50ms |
| Recall (top 10, 500+ memories) | ~100ms | < 200ms |
| Consolidation (500 memories) | ~1.7s | < 5s |

*Benchmarked on Windows with OneDrive sync. Faster on Linux/Mac.*

---

## API Server

```bash
# Start the API server
python -m agentvault.cli.main serve --port 8420

# Health check
curl http://localhost:8420/health
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/memories` | Store a memory |
| POST | `/api/v1/search` | Recall memories |
| GET | `/api/v1/memories/{id}` | Get a memory |
| DELETE | `/api/v1/memories/{id}` | Delete a memory |
| GET | `/api/v1/agents` | List all agents |
| POST | `/api/v1/consolidate` | Run consolidation |

---

## Integrations

- **[CodeVault VS Code Extension](https://github.com/ilanetall-boop/codevault-vscode)** — Persistent memory for your coding agents
- **LangChain** — See `examples/langchain_integration.py`
- **CrewAI** — See `examples/crewai_integration.py`

---

## Project Structure

```
agentvault/
├── core/           # Memory engine (types, manager, episodic, semantic, procedural)
├── processing/     # Indexer, ranker, consolidator, extractor
├── storage/        # FAISS vector store + SQLite structured store
├── api/            # FastAPI REST server
├── sdk/            # Python client + decorators
├── multi_agent/    # Shared memory, permissions, sync
└── cli/            # Command-line interface
```

---

## Tests

```bash
pytest tests/ -v
# 56 tests passed
```

---

## Roadmap

- [ ] PyPI package (`pip install agentvault`)
- [ ] PostgreSQL + Qdrant for production scale
- [ ] TypeScript SDK
- [ ] Dashboard web for memory visualization
- [ ] Claude Code native integration
- [ ] LangChain / CrewAI official plugins

---

## License

MIT — Use it, fork it, build on it.

---

**Built with purpose for the AI agent ecosystem.**

*If this project helps you, consider giving it a star.*
