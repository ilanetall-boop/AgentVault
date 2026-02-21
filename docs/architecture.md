# AgentVault Architecture

## Overview

AgentVault is a layered system designed for modularity and extensibility. Each layer has a clear responsibility and communicates through well-defined interfaces.

## Layer Diagram

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
|  | Extractor |            | (merge/forget)    |   |
|  +-----------+------------+-------------------+   |
+--------------------------------------------------+
|              Storage Layer                        |
|  +---------------+---------------------------+   |
|  | Vector Store  |  Structured Store         |   |
|  | (ChromaDB)    |  (SQLite/PostgreSQL)      |   |
|  +---------------+---------------------------+   |
+--------------------------------------------------+
```

## Components

### 1. SDK Client (`sdk/client.py`)
The public-facing API. Provides sync and async interfaces for all operations. Wraps the Memory Manager with a simple, developer-friendly API.

### 2. Memory Manager (`core/memory_manager.py`)
The central orchestrator. Routes memories to the correct type handler, coordinates indexing, and triggers consolidation.

### 3. Memory Types (`core/episodic.py`, `semantic.py`, `procedural.py`)
Type-specific logic for creating, validating, and managing each memory type:
- **Episodic**: Timestamped events with context
- **Semantic**: Facts with subject-predicate-object triples
- **Procedural**: Step-by-step procedures with success tracking

### 4. Processing Pipeline

- **Indexer** (`processing/indexer.py`): Generates vector embeddings using sentence-transformers (local) or OpenAI (optional).
- **Ranker** (`processing/ranker.py`): Combines similarity, recency, importance, and frequency into a final relevance score.
- **Consolidator** (`processing/consolidator.py`): Merges similar memories, promotes frequent episodic to semantic, forgets old unimportant memories, and reinforces accessed memories.
- **Extractor** (`processing/extractor.py`): Extracts structured information (type, importance, tags, facts, steps) from raw text.

### 5. Storage Layer

- **Vector Store** (`storage/vector_store.py`): ChromaDB for semantic similarity search.
- **Structured Store** (`storage/structured_store.py`): SQLite/PostgreSQL for CRUD, filters, and metadata.
- **Hybrid Store** (`storage/hybrid_store.py`): Combines both for hybrid search (vector similarity + structured filters).

### 6. Multi-Agent (`multi_agent/`)
- **Shared Memory**: Namespace-based memory sharing between agents
- **Permissions**: Read/write/admin access control
- **Sync**: Event-driven notifications for memory changes

## Data Flow

### Write (Remember)
```
Content → Extractor (auto-detect type, importance, tags)
       → Indexer (generate embedding)
       → Hybrid Store (save to SQLite + ChromaDB)
       → Check consolidation threshold
```

### Read (Recall)
```
Query → Indexer (generate query embedding)
     → Hybrid Store (vector search + structured filters)
     → Ranker (combine similarity + recency + importance + frequency)
     → Consolidator (reinforce accessed memories)
     → Return ranked results
```

### Consolidation
```
All memories → Reinforce (boost accessed memories)
            → Merge (combine similar memories)
            → Promote (frequent episodic → semantic)
            → Forget (old + unimportant + unaccessed)
```

## Design Decisions

1. **Async-first**: All I/O operations are async with sync wrappers for convenience.
2. **Zero config**: SQLite + ChromaDB + sentence-transformers work out of the box.
3. **Local-first**: No cloud dependency by default. OpenAI embeddings are optional.
4. **Hybrid search**: Combining vector similarity with structured filters gives better results than either alone.
5. **Type system**: Three memory types mirror cognitive science research on human memory.
