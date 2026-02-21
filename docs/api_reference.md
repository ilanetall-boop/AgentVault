# API Reference

## SDK Client — `AgentVault`

### Constructor

```python
AgentVault(
    agent_id: str = "default",
    db_path: str | None = None,        # SQLite path (default: ~/.agentvault/memories.db)
    chroma_path: str | None = None,     # ChromaDB path (default: ~/.agentvault/chroma)
    auto_initialize: bool = True,       # Auto-init on first operation
)
```

### Memory Operations

#### `remember(content, type?, importance?, tags?, metadata?, source?) -> Memory`
Store a new memory. Auto-detects type and importance if not provided.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | required | Text content to remember |
| `type` | `str \| None` | `None` | `"episodic"`, `"semantic"`, or `"procedural"` (auto-detected) |
| `importance` | `float \| None` | `None` | Score 0-1 (auto-estimated) |
| `tags` | `list[str] \| None` | `None` | Categorization tags (auto-extracted) |
| `metadata` | `dict \| None` | `None` | Additional metadata |
| `source` | `str \| None` | `None` | What created this memory |

#### `remember_episode(event, context?, importance?, tags?, source?) -> Memory`
Store an episodic memory.

#### `remember_procedure(name, steps, content?, learned_from?, importance?, tags?) -> Memory`
Store a procedural memory.

#### `recall(query, top_k?, types?, min_importance?, tags?) -> RecallResult`
Recall relevant memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Search query text |
| `top_k` | `int` | `10` | Maximum results |
| `types` | `list[str] \| None` | `None` | Filter by memory types |
| `min_importance` | `float` | `0.0` | Minimum importance threshold |
| `tags` | `list[str] \| None` | `None` | Filter by tags |

#### `forget(older_than?, importance_below?) -> int`
Delete memories matching criteria. Returns count deleted.

#### `consolidate() -> ConsolidationResult`
Run memory consolidation (merge, promote, forget, reinforce).

#### `count(memory_type?) -> int`
Count memories, optionally filtered by type.

#### `get(memory_id) -> Memory | None`
Retrieve a specific memory by ID.

#### `share(memory_id, with_agents) -> None`
Share a memory with other agents.

### Async Variants

All methods have async counterparts prefixed with `a`:
- `aremember()`, `arecall()`, `aforget()`, `aconsolidate()`, `acount()`, `aget()`

---

## Data Models

### `Memory`

```python
class Memory:
    id: str                           # UUID
    agent_id: str                     # Owner agent
    type: MemoryType                  # episodic | semantic | procedural
    content: str                      # Text content
    metadata: dict[str, Any]          # Additional metadata
    embedding: list[float] | None     # Vector embedding
    importance: float                 # 0-1 score
    access_count: int                 # Times recalled
    created_at: datetime              # Creation time
    last_accessed: datetime           # Last recall time
    expires_at: datetime | None       # Optional expiration
    tags: list[str]                   # Categorization tags
    source: str | None                # Creation source
    related_memories: list[str]       # Related memory IDs
```

### `RecallResult`

```python
class RecallResult:
    memories: list[Memory]            # Ranked memories
    relevance_scores: list[float]     # Corresponding scores
    total_found: int                  # Total matches before limit
    query_time_ms: float              # Query duration
```

### `ConsolidationResult`

```python
class ConsolidationResult:
    merged: int                       # Memories merged
    promoted: int                     # Episodic promoted to semantic
    forgotten: int                    # Memories deleted
    reinforced: int                   # Memories strengthened
    total_processed: int              # Total memories processed
    duration_ms: float                # Processing time
```

---

## REST API Endpoints

### Health
- `GET /health` — Health check

### Memories
- `POST /api/v1/memories` — Create a memory
- `GET /api/v1/memories/{id}` — Get a memory by ID
- `DELETE /api/v1/memories/{id}` — Delete a memory

### Search
- `POST /api/v1/search` — Search memories

### Agents
- `GET /api/v1/agents/{id}/memories` — List agent memories
- `GET /api/v1/agents/{id}/stats` — Memory statistics
- `POST /api/v1/agents/{id}/consolidate` — Run consolidation
- `DELETE /api/v1/agents/{id}/memories` — Forget memories
