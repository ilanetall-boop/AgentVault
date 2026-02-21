# PROMPT CLAUDE CODE — Conception d'un Système de Mémoire Persistante pour Agents IA

---

## 🎯 CONTEXTE DU PROJET

Tu es un architecte logiciel senior spécialisé en systèmes d'IA. Tu dois concevoir et développer **AgentVault** — un système de mémoire persistante universel pour agents IA.

### Vision
AgentVault est une infrastructure open source qui permet à n'importe quel agent IA de stocker, retrouver et raisonner sur des souvenirs à long terme. C'est le **"cerveau persistant"** qui manque à tous les agents aujourd'hui.

### Positionnement
- **Ce que c'est** : une couche mémoire universelle, rapide, fiable et scalable pour agents IA
- **Ce que ce n'est pas** : un simple vector store, un chatbot, ou un wrapper autour de ChromaDB
- **Différenciation** : mémoire structurée multi-couches (pas juste des embeddings), support natif multi-agents, API simple

---

## 📐 ARCHITECTURE À CONCEVOIR

### Phase 1 — Core Memory Engine (MVP)

Conçois et implémente l'architecture suivante :

```
┌─────────────────────────────────────────────────┐
│                  AgentVault API                  │
│              (REST + SDK Python/TS)              │
├─────────────────────────────────────────────────┤
│              Memory Manager                      │
│  ┌───────────┬──────────────┬─────────────────┐ │
│  │ Episodic  │  Semantic    │  Procedural     │ │
│  │ Memory    │  Memory      │  Memory         │ │
│  │ (events)  │  (facts)     │  (skills/how-to)│ │
│  └───────────┴──────────────┴─────────────────┘ │
├─────────────────────────────────────────────────┤
│            Memory Processing Pipeline            │
│  ┌──────────┬───────────┬──────────────────┐    │
│  │ Indexer  │ Ranker    │ Consolidator     │    │
│  │          │           │ (merge/forget)   │    │
│  └──────────┴───────────┴──────────────────┘    │
├─────────────────────────────────────────────────┤
│              Storage Layer                       │
│  ┌──────────────┬────────────────────────┐      │
│  │ Vector Store │  Structured Store      │      │
│  │ (embeddings) │  (SQLite/Postgres)     │      │
│  └──────────────┴────────────────────────┘      │
└─────────────────────────────────────────────────┘
```

### Les 3 types de mémoire (CRITIQUE)

1. **Mémoire Épisodique** — "Ce qui s'est passé"
   - Stocke les événements, interactions, conversations
   - Horodatée, contextuelle
   - Exemple : "Le 15 mars, l'utilisateur a demandé de refactorer le module auth"

2. **Mémoire Sémantique** — "Ce que je sais"
   - Stocke les faits, préférences, connaissances extraites
   - Structurée en graphe de connaissances léger
   - Exemple : "L'utilisateur préfère TypeScript à JavaScript"

3. **Mémoire Procédurale** — "Comment faire"
   - Stocke les patterns, workflows, compétences apprises
   - Permet à l'agent d'apprendre de ses erreurs
   - Exemple : "Pour déployer sur Vercel, suivre ces 5 étapes..."

---

## 🛠 STACK TECHNIQUE

```
Langage principal : Python 3.11+
Framework API    : FastAPI
Vector Store     : ChromaDB (local) / Qdrant (production)
Base relationnelle : SQLite (local) / PostgreSQL (production)
Embeddings       : sentence-transformers (local) / OpenAI ada-002 (option)
Sérialisation    : msgpack + JSON
Tests            : pytest + pytest-asyncio
Packaging        : pyproject.toml (pas setup.py)
CLI              : Typer
```

---

## 📁 STRUCTURE DU PROJET

Génère cette structure exacte :

```
agentvault/
├── README.md                    # Documentation principale
├── pyproject.toml               # Config projet + dépendances
├── LICENSE                      # MIT License
│
├── agentvault/                  # Package principal
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── memory_manager.py    # Orchestrateur principal
│   │   ├── episodic.py          # Mémoire épisodique
│   │   ├── semantic.py          # Mémoire sémantique
│   │   ├── procedural.py        # Mémoire procédurale
│   │   └── types.py             # Types/modèles Pydantic
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── indexer.py           # Indexation + embeddings
│   │   ├── ranker.py            # Scoring de pertinence
│   │   ├── consolidator.py      # Fusion/oubli de mémoires
│   │   └── extractor.py         # Extraction de faits depuis le texte
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py              # Interface abstraite storage
│   │   ├── vector_store.py      # Implémentation vector store
│   │   ├── structured_store.py  # Implémentation SQL
│   │   └── hybrid_store.py      # Combiner vector + structured
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py            # FastAPI app
│   │   ├── routes/
│   │   │   ├── memories.py      # CRUD mémoires
│   │   │   ├── search.py        # Recherche/recall
│   │   │   ├── agents.py        # Gestion agents
│   │   │   └── health.py        # Health checks
│   │   └── middleware.py        # Auth, rate limiting
│   │
│   ├── sdk/
│   │   ├── __init__.py
│   │   ├── client.py            # Client Python sync/async
│   │   └── decorators.py        # @remember, @recall decorators
│   │
│   ├── multi_agent/
│   │   ├── __init__.py
│   │   ├── shared_memory.py     # Mémoire partagée entre agents
│   │   ├── permissions.py       # Contrôle d'accès
│   │   └── sync.py              # Synchronisation
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py              # CLI Typer
│
├── tests/
│   ├── conftest.py
│   ├── test_episodic.py
│   ├── test_semantic.py
│   ├── test_procedural.py
│   ├── test_memory_manager.py
│   ├── test_consolidator.py
│   ├── test_multi_agent.py
│   └── test_api.py
│
├── examples/
│   ├── basic_usage.py           # Exemple simple
│   ├── multi_agent_team.py      # Équipe de 3 agents
│   ├── langchain_integration.py # Intégration LangChain
│   └── crewai_integration.py    # Intégration CrewAI
│
├── docs/
│   ├── architecture.md
│   ├── quickstart.md
│   └── api_reference.md
│
├── docker-compose.yml           # Dev environment
├── Dockerfile
└── .github/
    └── workflows/
        └── ci.yml               # Tests + lint
```

---

## 💻 IMPLÉMENTATION — FICHIERS CLÉS

### 1. Types de base (core/types.py)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import uuid

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    type: MemoryType
    content: str
    metadata: dict[str, Any] = {}
    embedding: Optional[list[float]] = None
    importance: float = 0.5  # 0-1 score
    access_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    tags: list[str] = []
    source: Optional[str] = None  # quel agent/outil a créé ce souvenir
    related_memories: list[str] = []  # IDs de mémoires liées

class MemoryQuery(BaseModel):
    query: str
    agent_id: str
    types: list[MemoryType] = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
    top_k: int = 10
    min_importance: float = 0.0
    time_range: Optional[tuple[datetime, datetime]] = None
    tags: list[str] = []
    include_shared: bool = True  # inclure mémoires partagées multi-agents

class RecallResult(BaseModel):
    memories: list[Memory]
    relevance_scores: list[float]
    total_found: int
    query_time_ms: float
```

### 2. SDK Client — L'API que les devs utiliseront (sdk/client.py)

L'API doit être ULTRA simple. C'est ce qui fera l'adoption.

```python
# Usage cible — ce que les développeurs écriront :

from agentvault import AgentVault

# Initialisation
vault = AgentVault(agent_id="mon-agent")

# Stocker un souvenir
vault.remember("L'utilisateur préfère les réponses courtes", 
               type="semantic", 
               importance=0.8,
               tags=["preference", "style"])

# Retrouver des souvenirs pertinents
memories = vault.recall("comment l'utilisateur aime les réponses ?", top_k=5)

# Stocker un épisode
vault.remember_episode(
    event="L'utilisateur a demandé un refactor du module auth",
    context={"file": "auth.py", "action": "refactor"},
    importance=0.7
)

# Stocker une procédure apprise
vault.remember_procedure(
    name="deploy_vercel",
    steps=["npm run build", "vercel deploy --prod"],
    learned_from="conversation_123"
)

# Mémoire partagée entre agents
vault.share("fait_important", with_agents=["agent-2", "agent-3"])

# Oubli / nettoyage
vault.forget(older_than="30d", importance_below=0.2)

# Decorators pour intégration facile
@vault.auto_remember
async def handle_conversation(message: str):
    # La mémoire est automatiquement extraite et stockée
    pass
```

### 3. Memory Manager (core/memory_manager.py)

Le Memory Manager est le cerveau. Il doit :
- Router les mémoires vers le bon type (épisodique/sémantique/procédural)
- Calculer l'importance automatiquement
- Gérer la consolidation (fusionner les doublons, oublier l'inutile)
- Gérer le recall intelligent (combiner vector search + filtre structuré)

### 4. Consolidator (processing/consolidator.py)

Le Consolidator est la VRAIE innovation. Il simule comment le cerveau humain consolide les souvenirs :
- **Fusion** : 2 souvenirs similaires → 1 souvenir enrichi
- **Promotion** : un souvenir épisodique fréquent → devient sémantique (fait établi)
- **Oubli** : mémoires jamais accédées + faible importance → supprimées
- **Renforcement** : chaque accès augmente l'importance

### 5. Multi-Agent Shared Memory (multi_agent/shared_memory.py)

Gestion de la mémoire partagée :
- **Namespaces** : chaque agent a son espace + espaces partagés
- **Permissions** : read/write/admin par agent
- **Sync** : event-driven (un agent écrit → les autres sont notifiés)
- **Conflits** : stratégie last-write-wins avec historique

---

## 🧪 TESTS REQUIS

Écris des tests pour chaque composant. Priorité :

1. **test_memory_manager.py** — Tests d'intégration du flux complet
   - remember → recall → vérifier pertinence
   - remember 100 mémoires → recall doit être < 50ms
   - consolidation automatique après seuil

2. **test_consolidator.py** — Tests de la logique de consolidation
   - 2 mémoires similaires → fusion correcte
   - promotion épisodique → sémantique
   - oubli des mémoires périmées

3. **test_multi_agent.py** — Tests multi-agents
   - Agent A écrit → Agent B lit (si partagé)
   - Agent A écrit → Agent C ne lit PAS (si pas partagé)
   - Permissions respectées

4. **test_api.py** — Tests API endpoints
   - CRUD complet
   - Search avec filtres
   - Rate limiting

---

## 📊 BENCHMARKS À INCLURE

Crée un script `benchmarks/performance.py` qui mesure :

| Métrique | Cible |
|----------|-------|
| Temps d'écriture (1 mémoire) | < 10ms |
| Temps de recall (top 10 sur 10K mémoires) | < 50ms |
| Temps de recall (top 10 sur 100K mémoires) | < 200ms |
| Consolidation (1000 mémoires) | < 5s |
| Mémoire RAM (10K mémoires) | < 500MB |

---

## 📝 README.md

Le README doit être viral. Structure :

```markdown
# 🧠 AgentVault — Persistent Memory for AI Agents

> Give your AI agents a brain that remembers.

[badges: PyPI, tests, license, stars]

## The Problem
AI agents forget everything. Every conversation starts from zero.

## The Solution
AgentVault gives any AI agent persistent, intelligent memory — 
episodic, semantic, and procedural — in 3 lines of code.

## Quick Start (30 seconds)
pip install agentvault
[3 lignes de code qui marchent immédiatement]

## Why AgentVault?
- 🧠 3 memory types (like the human brain)
- ⚡ < 50ms recall on 100K+ memories
- 🤝 Multi-agent shared memory
- 🔌 Works with LangChain, CrewAI, AutoGen
- 🧹 Smart consolidation (merge, forget, reinforce)
- 🔒 Per-agent permissions
- 📦 Zero config local mode, scales to production

## Integrations
[LangChain, CrewAI, AutoGen, custom agents]

## Architecture
[Diagramme ASCII de l'architecture]
```

---

## ⚡ ORDRE D'EXÉCUTION

Suis cet ordre strict :

1. **Setup** : pyproject.toml, structure dossiers, dépendances
2. **Types** : core/types.py (les modèles Pydantic)
3. **Storage** : storage layer (SQLite + vector store local)
4. **Core** : les 3 types de mémoire
5. **Processing** : indexer, ranker, consolidator
6. **Memory Manager** : orchestrateur
7. **SDK Client** : l'API simple
8. **Tests** : tous les tests
9. **API** : FastAPI server
10. **Multi-agent** : shared memory
11. **Examples** : les 4 exemples d'intégration
12. **README** : documentation virale
13. **Docker** : docker-compose pour dev
14. **CI** : GitHub Actions

---

## 🚨 CONTRAINTES CRITIQUES

- **Python 3.11+** minimum
- **Async-first** : tout doit être async avec fallback sync
- **Type hints** partout, Pydantic pour la validation
- **Zero config** : `vault = AgentVault()` doit marcher sans rien configurer (SQLite local par défaut)
- **Pas de dépendance à OpenAI** : embeddings locaux par défaut (sentence-transformers), OpenAI en option
- **Tests** : coverage > 80%
- **Docstrings** : sur chaque classe et méthode publique
- **Pas de print()** : utiliser `logging` partout
- **Code propre** : suit PEP 8, ruff pour le linting

---

## 🎯 CRITÈRE DE SUCCÈS

Le projet est réussi quand :

1. `pip install agentvault` fonctionne
2. En 3 lignes de code, un agent peut stocker et retrouver des souvenirs
3. Le recall est pertinent (les bons souvenirs remontent)
4. La consolidation fonctionne (fusion, oubli, renforcement)
5. 2 agents peuvent partager de la mémoire
6. Tous les tests passent
7. Le README donne envie de star le repo

---

**Commence maintenant. Étape 1 : Setup du projet.**
