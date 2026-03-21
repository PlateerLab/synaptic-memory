# Synaptic Memory

Brain-inspired knowledge graph with spreading activation, Hebbian learning, and memory consolidation.

## Features

- **Hybrid Search** — 3-stage fallback: FTS + fuzzy + vector → synonym expansion → LLM query rewrite
- **Hebbian Learning** — Co-activated nodes strengthen connections; failures weaken them
- **Memory Consolidation** — L0 (raw, 72h TTL) → L1 → L2 → L3 (permanent, proven knowledge)
- **Resonance Scoring** — 4-axis ranking: relevance × importance × recency × vitality
- **Korean + English** — Bidirectional synonym map, unicode tokenizers, fuzzy matching
- **Zero Core Deps** — Pure Python core; backends are optional extras
- **Async-first** — All I/O uses async/await

## Install

```bash
pip install synaptic-memory                   # Core (MemoryBackend only)
pip install synaptic-memory[sqlite]           # + SQLite backend
pip install synaptic-memory[postgresql]       # + PostgreSQL backend (v0.2)
pip install synaptic-memory[all]              # Everything
```

## Quick Start

```python
from synaptic.backends.memory import MemoryBackend
from synaptic import SynapticGraph, NodeKind, EdgeKind

async def main():
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)

    # Add knowledge
    n1 = await graph.add("CI/CD 파이프라인", "배포 자동화 구현", kind=NodeKind.LESSON)
    n2 = await graph.add("테스트 커버리지", "80% 이상 유지", kind=NodeKind.RULE)
    await graph.link(n1.id, n2.id, kind=EdgeKind.RELATED)

    # Search (FTS + fuzzy + synonym expansion + spreading activation)
    result = await graph.search("배포")
    for activated in result.nodes:
        print(f"  [{activated.node.kind}] {activated.node.title} (score: {activated.resonance:.2f})")

    # Reinforce (Hebbian learning)
    await graph.reinforce([n1.id, n2.id], success=True)

    # Consolidate (TTL cleanup + level promotion)
    await graph.consolidate()

    await backend.close()
```

## With SQLite (Persistent)

```python
from synaptic.backends.sqlite import SQLiteBackend
from synaptic import SynapticGraph

backend = SQLiteBackend("knowledge.db")
await backend.connect()
graph = SynapticGraph(backend)
# ... same API as above
```

## Architecture

```
SynapticGraph (Facade)
  ├── Store (CRUD + tag extraction)
  ├── HybridSearch (3-stage fallback + spreading activation + resonance)
  ├── HebbianEngine (co-activation learning)
  ├── ConsolidationCascade (L0→L3 memory lifecycle)
  └── MarkdownExporter
       │
   StorageBackend (Protocol)
       │
  MemoryBackend │ SQLiteBackend │ PostgreSQLBackend
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design.
See [docs/ROADMAP.md](docs/ROADMAP.md) for development plan.

## API

### SynapticGraph

| Method | Description |
|--------|-------------|
| `add(title, content, kind=..., tags=..., embedding=...)` | Add a knowledge node |
| `link(source_id, target_id, kind=..., weight=...)` | Link two nodes |
| `search(query, limit=10, embedding=...)` | Hybrid search with resonance scoring |
| `get(node_id)` | Get a node (increments access count) |
| `remove(node_id)` | Delete a node and its edges |
| `reinforce(node_ids, success=True)` | Hebbian learning on co-activated nodes |
| `consolidate(digester=..., context=...)` | Run memory lifecycle (TTL + promotion) |
| `prune()` | Remove weak edges (weight < 0.1) |
| `decay()` | Decay all node vitality (×0.95) |
| `export_markdown()` | Export as Markdown |
| `stats()` | Node counts by kind and level |

### Backends

| Backend | Deps | Use Case |
|---------|------|----------|
| `MemoryBackend` | None | Testing, ephemeral |
| `SQLiteBackend` | aiosqlite | Embedded, single-process |
| `PostgreSQLBackend` | asyncpg, pgvector | Production, multi-process |

### Protocols (Extensible)

| Protocol | Description |
|----------|-------------|
| `StorageBackend` | Pluggable storage (implement for custom DB) |
| `Digester` | Convert structured context → nodes/edges |
| `QueryRewriter` | LLM-based query expansion |
| `TagExtractor` | Auto-tag extraction from text |

## Dev

```bash
uv sync --extra dev --extra sqlite
uv run pytest -v                    # 93 tests
ruff check --fix && ruff format     # Lint
pyright                             # Type check (strict)
```

## License

MIT
