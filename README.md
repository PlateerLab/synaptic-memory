# Synaptic Memory

Knowledge graph + MCP tool server for LLM agents.

Any data in, structured graph out. LLM agents explore it with 29 atomic tools.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 2 Lines to Start

```python
from synaptic import SynapticGraph

# Any data → knowledge graph (CSV, JSONL, directory)
graph = await SynapticGraph.from_data("./my_data/")

# Search
result = await graph.search("my question")
```

That's it. Auto-detects file format, generates ontology profile, ingests, indexes.

---

## What it does

```
Your data (CSV, JSONL, PDF, any format)
  ↓  auto-detect format + auto-generate DomainProfile
  ↓  DocumentIngester (text) / TableIngester (structured)
  ↓
Knowledge Graph (Category → Document → Chunk)
  ↓
29 MCP tools → LLM agent explores via multi-turn tool use
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
| `filter_nodes` | Property filter (>=, <=, contains) — like SQL WHERE |
| `aggregate_nodes` | GROUP BY + COUNT/SUM/AVG |
| `join_related` | FK-based related record lookup — like SQL JOIN |

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

### Single-shot retrieval

| Dataset | Type | Nodes | Easy MRR | Hard MRR |
|---------|------|-------|----------|----------|
| KRRA (Korean public sector) | Text | 19,720 | **0.967** | 0.507 |
| assort (fashion e-commerce) | CSV | 13,909 | **0.880** | 0.127 |

### Multi-turn agent (Claude Sonnet 4.6)

| Query type | Example | Turns | Result |
|-----------|---------|-------|--------|
| Factoid | "인권영향평가 결과" | 6 | Detailed table |
| Cross-document | "운영계획과 인권경영" | 10 | Multi-source synthesis |
| Absence proof | "환불 예외 있나?" | 7 | Found 3 exception clauses |
| Paraphrase | "말 복지 프로그램" | 8 | Found 재활힐링승마 |
| **Hard (single-shot fails)** | **4 queries** | **6-10** | **4/4 solved** |

Single-shot MRR 0.507 → Multi-turn **100% solved**.

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

## Dev

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q                   # 687+ tests
uv run ruff check --fix
```

## License

MIT
