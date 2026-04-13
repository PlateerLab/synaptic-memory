# Synaptic Memory

Knowledge graph + MCP tool server for LLM agents.

Any data in, structured graph out. LLM agents explore it with 29 atomic tools.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> [한국어 README](README.ko.md)

---

## 2 Lines to Start

```python
from synaptic import SynapticGraph

# Any data → knowledge graph (CSV, JSONL, directory)
graph = await SynapticGraph.from_data("./my_data/")

# Or directly from a database — SQLite / PostgreSQL / MySQL / Oracle / MSSQL
graph = await SynapticGraph.from_database(
    "postgresql://user:pass@host:5432/dbname"
)

# Search
result = await graph.search("my question")
```

That's it. Auto-detects file format or DB schema, generates ontology profile, ingests, indexes, builds FK edges.

---

## What it does

```
Your data (CSV, JSONL, PDF, SQL database)
  ↓  auto-detect format / auto-discover DB schema + FKs
  ↓  DocumentIngester (text) / TableIngester / DbIngester
  ↓
Knowledge Graph
  ├─ Documents: Category → Document → Chunk
  └─ Structured: table rows as ENTITY nodes + RELATED edges (FKs)
  ↓
29 MCP tools → LLM agent explores via graph-aware multi-turn tool use
```

**Two jobs, nothing else:**
1. **Build the graph well** — cheap extraction, no LLM at index time
2. **Give the LLM good tools** — the agent decides what to search

---

## Install

```bash
pip install synaptic-memory                # Core (zero deps)
pip install synaptic-memory[sqlite]        # + SQLite FTS5 backend
pip install synaptic-memory[korean]        # + Kiwi morphological analyzer
pip install synaptic-memory[vector]        # + usearch HNSW index
pip install synaptic-memory[mcp]           # + MCP server for Claude
pip install synaptic-memory[all]           # Everything
```

---

## Quick Start

### Option A: Two lines (easiest)

```python
from synaptic import SynapticGraph

# CSV file
graph = await SynapticGraph.from_data("products.csv")

# JSONL documents
graph = await SynapticGraph.from_data("documents.jsonl")

# Entire directory (scans all CSV/JSONL)
graph = await SynapticGraph.from_data("./my_corpus/")

# With embedding (optional, improves semantic search)
graph = await SynapticGraph.from_data(
    "./my_corpus/",
    embed_url="http://localhost:11434/v1",
)

# Search
result = await graph.search("my question")
```

### Option B: MCP server (Claude Desktop / Code)

```bash
synaptic-mcp --db my_graph.db
synaptic-mcp --db my_graph.db --embed-url http://localhost:11434/v1
```

Claude can now call 29 tools to explore your graph.

### Option C: Full control

```python
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.document_ingester import DocumentIngester, JsonlDocumentSource

profile = DomainProfile.load("my_profile.toml")
backend = SqliteGraphBackend("graph.db")
await backend.connect()

source = JsonlDocumentSource("docs.jsonl", "chunks.jsonl")
ingester = DocumentIngester(profile=profile, backend=backend)
await ingester.ingest(source)
```

---

## 3rd-Generation Retrieval

| Generation | Approach | LLM cost at indexing |
|-----------|----------|---------------------|
| 1st (GraphRAG) | LLM extracts entities + relations + summaries | High |
| 2nd (LightRAG) | Delays LLM to query time | Medium |
| **3rd (this)** | **Relation-free graph, hybrid retrieval** | **Zero** |

No LLM at indexing. The graph is a search index, not a knowledge base.

---

## Agent Tools (29 total)

### Text search tools
| Tool | Purpose |
|------|---------|
| `deep_search` | **Recommended.** Search → expand → read documents in ONE call |
| `compare_search` | Auto-decompose multi-topic queries, search in parallel |
| `search` | FTS + vector hybrid search |
| `expand` | 1-hop graph neighbours |
| `get_document` | Full document with query-relevant chunks |
| `search_exact` | Literal substring match for IDs/codes |
| `follow` | Walk a specific edge type |

### Structured data tools
| Tool | Purpose |
|------|---------|
| `filter_nodes` | Property filter (>=, <=, contains) — returns `{total, showing}` for accurate counting |
| `aggregate_nodes` | GROUP BY + COUNT/SUM/AVG/MAX/MIN with optional WHERE pre-filter |
| `join_related` | FK-based related record lookup — walks RELATED edges (O(degree)) |

### Navigation tools
| Tool | Purpose |
|------|---------|
| `list_categories` | Category list with document counts |
| `count` | Structural count by kind/category/year |
| `session_info` | Multi-turn session state |

All tools return `{ data, hints, session }`. The `SearchSession` tracks seen nodes across turns so the agent never re-reads the same chunk.

---

## Retrieval Pipeline

```
Query
  ↓  Kiwi morphological analysis (Korean) or regex (other)
  ↓  BM25 FTS + title 3x boost + substring fallback
  ↓  Vector search (usearch HNSW, optional)
  ↓  Vector PRF (pseudo relevance feedback, 2-pass)
  ↓  PPR graph discovery (personalized pagerank)
  ↓  GraphExpander (1-hop: category siblings, chunk-next, entity mentions)
  ↓  HybridReranker (lexical + semantic + graph + structural + authority + temporal)
  ↓  MaxP document aggregation (coverage bonus)
  ↓  Cross-encoder reranker (bge-reranker-v2-m3 via TEI, optional)
  ↓  EvidenceAggregator (MMR diversity + per-doc cap + category coverage)
Result
```

---

## Benchmarks

### Single-shot retrieval (EvidenceSearch + embed + reranker)

| Dataset | Type | Nodes | MRR | Hit |
|---------|------|-------|-----|-----|
| KRRA Easy | Korean documents | 19,720 | **0.967** | 20/20 |
| KRRA Hard | Korean documents | 19,720 | **1.000** | 15/15 |
| X2BEE Easy | PostgreSQL (e-commerce) | 19,843 | **1.000** | 20/20 |
| assort Easy | Fashion CSV | 13,909 | **0.867** | 13/15 |
| HotPotQA-24 | English multi-hop | 226 | **0.964** | 24/24 |
| Allganize RAG-ko | Korean enterprise | 200 | **0.905** | — |
| Allganize RAG-Eval | Finance/medical/legal KO | 300 | **0.874** | — |
| PublicHealthQA | Korean public health | 77 | **0.600** | 56/77 |

### Multi-turn agent (GPT-4o-mini, 5 turns max)

| Dataset | Result |
|---------|--------|
| KRRA Hard agent | 10-13/15 (67-87%) |
| **X2BEE Hard agent** | **17/19 (89%)** |
| **assort Hard agent** | **12/15 (80%)** |

Structured data queries (filter / aggregate / FK join / count) work end-to-end through graph-aware tools.

---

## Architecture

```
SynapticGraph.from_data("./data/")          ← Easy API
  ↓
Auto-detect → DomainProfile → Ingest → Index
  ↓
StorageBackend (Protocol)
  ├── MemoryBackend        (testing)
  ├── SqliteGraphBackend   (recommended, FTS5 + HNSW)
  ├── KuzuBackend          (embedded Cypher)
  ├── PostgreSQLBackend    (pgvector)
  └── CompositeBackend     (mix backends)
  ↓
Retrieval pipeline (BM25 + vector + PRF + PPR + reranker + MMR)
  ↓
Agent tools (29) → MCP server → LLM agent
```

---

## Backends

| Backend | Vector Search | Scale | Use Case |
|---------|--------------|-------|----------|
| `MemoryBackend` | cosine | ~10K | Testing |
| `SqliteGraphBackend` | **usearch HNSW** | ~100K | **Default** |
| `KuzuBackend` | HNSW | ~10M | Graph-heavy |
| `PostgreSQLBackend` | pgvector | ~1M | Production |
| `CompositeBackend` | Qdrant | Unlimited | Scale-out |

---

## Optional Extras

| Extra | What it adds |
|-------|-------------|
| `korean` | Kiwi morphological analyzer for Korean FTS |
| `vector` | usearch HNSW index (100x faster vector search) |
| `embedding` | aiohttp for embedding API calls |
| `mcp` | MCP server for Claude Desktop/Code |
| `sqlite` | aiosqlite backend |

---

## Documentation

| Doc | What it is |
|-----|-----------|
| [docs/GUIDE.md](docs/GUIDE.md) | Friendly intro — what/why/how, zero jargon |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | 30-minute hands-on walkthrough |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | 3rd-gen GraphRAG + pipeline internals |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Original neural-inspired design |
| [docs/COMPARISON.md](docs/COMPARISON.md) | vs GraphRAG / LightRAG / LazyGraphRAG |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Future plans |

## Dev

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q                   # 687+ tests
uv run ruff check --fix
```

## License

MIT
