# Synaptic Memory

Knowledge graph + MCP tool server for LLM agents.

LLM agents call atomic search tools to explore a graph built from any domain data. The agent decides what to search, when to expand, and when to stop. The library only provides data and tools — all judgment lives in the LLM.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## What it does

```
Your data (JSONL, CSV, any format)
  ↓  DomainProfile (auto-generated or hand-tuned)
  ↓  DocumentIngester / TableIngester
  ↓
Category → Document → Chunk  (relation-free graph)
  ↓
7 atomic MCP tools  →  LLM agent explores via multi-turn tool use
```

**Two jobs, nothing else:**
1. **Build the graph well** — cheap extraction, no LLM needed at index time
2. **Give the LLM good tools** — search, expand, get_document, count, list_categories, search_exact, follow

---

## 3rd-generation retrieval

| Generation | Approach | LLM cost at indexing |
|-----------|----------|---------------------|
| 1st (GraphRAG) | LLM extracts entities + relations + community summaries | High |
| 2nd (LightRAG) | Delays LLM to query time, lighter indexing | Medium |
| **3rd (this)** | **Relation-free graph, encoder-based extraction, hybrid retrieval** | **Zero** |

Synaptic Memory follows the 3rd-gen pattern: no LLM at indexing, hierarchical graph structure, and hybrid FTS + graph retrieval. The graph is a search index, not a knowledge base.

---

## Quick Start

### Install

```bash
pip install synaptic-memory              # Core (zero deps)
pip install synaptic-memory[sqlite]      # + SQLite FTS5 backend
pip install synaptic-memory[mcp]         # + MCP server for Claude
pip install synaptic-memory[all]         # Everything
```

### 1. Build a graph from your documents

```python
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.document_ingester import DocumentIngester, JsonlDocumentSource

# Auto-generate a profile from your data (or write one by hand)
profile = DomainProfile(name="my_corpus", locale="ko")

backend = SqliteGraphBackend("my_graph.db")
await backend.connect()

source = JsonlDocumentSource("docs.jsonl", "chunks.jsonl")
ingester = DocumentIngester(profile=profile, backend=backend)
stats = await ingester.ingest(source)
# → Category → Document → Chunk graph with FTS index
```

### 2. Let an LLM agent search it

```python
from synaptic.agent_tools import search_tool, expand_tool, get_document_tool
from synaptic.search_session import SearchSession

session = SearchSession()

# The LLM calls these tools in a loop:
result = await search_tool(backend, session, "인권영향평가 결과")
# → evidence list + suggested next actions

result = await get_document_tool(backend, session, "doc_abc123")
# → full document with all chunks in reading order

result = await expand_tool(backend, session, "chunk_xyz")
# → 1-hop neighbours (same-doc chunks, category siblings, etc.)
```

### 3. Or use as an MCP server (Claude Desktop / Claude Code)

```bash
synaptic-mcp --db my_graph.db

# Claude can now call:
#   agent_search, agent_expand, agent_get_document,
#   agent_list_categories, agent_count, agent_search_exact,
#   agent_follow, agent_session_info
```

---

## Auto profile generation

Don't want to manually configure stopwords and ontology hints? The ProfileGenerator does it for you:

```python
from synaptic.extensions.profile_generator import ProfileGenerator

# Rule-based (no LLM, no cost):
gen = ProfileGenerator()
profile = await gen.generate(
    name="my_corpus",
    samples=[doc.content for doc in docs[:50]],
    categories=[doc.category for doc in docs[:50]],
)
profile.save("my_profile.toml")

# With BYO embedder for ontology classification:
from synaptic.extensions.ontology_classifier import OntologyClassifier
classifier = OntologyClassifier(embedder=my_ollama_embedder)
gen = ProfileGenerator(classifier=classifier)
# → auto-maps category labels to NodeKind (RULE, DECISION, OBSERVATION...)
```

---

## Agent tool layer

7 atomic tools designed for multi-turn LLM exploration:

| Tool | Purpose |
|------|---------|
| `search` | FTS-seeded hybrid search with anchor extraction + graph expansion + reranking |
| `expand` | 1-hop graph neighbours (category siblings, chunk-next, entity mentions) |
| `get_document` | Full document with all chunks in reading order |
| `list_categories` | Enumerate categories with document counts |
| `count` | Structural count by kind / category / year |
| `search_exact` | Literal substring match for IDs and codes |
| `follow` | Walk a specific edge type (contains, part_of, next_chunk, mentions) |

Every tool returns:
- `data` — the actual payload
- `hints` — suggested next actions (LLM can ignore)
- `session` — budget remaining, seen nodes, queries tried

The `SearchSession` tracks state across turns so the agent never re-reads the same chunk.

---

## Validated on real data

| Dataset | Type | Nodes | MRR | Pipeline |
|---------|------|-------|-----|----------|
| KRRA (Korean public sector) | Text documents | 19,720 | 0.95 | FTS + graph |
| assort (fashion e-commerce) | Structured CSV | 13,909 | 0.95 | FTS + graph |

Multi-turn agent validation (Claude Sonnet 4.6, 5 query types):

| Query type | Example | Turns | Result |
|-----------|---------|-------|--------|
| Factoid | "인권영향평가 결과" | 6 | Detailed table with scores |
| Cross-document | "운영계획과 인권경영 충돌" | 9 | 4-stage framework cited |
| Absence proof | "환불 예외 있나?" | 7 | Found 3 exception clauses |
| Enumeration | "규정 총 몇 건?" | 3 | 235건 + full category breakdown |
| Temporal | "최신 운영계획" | 8 | 2024 document summarized |

---

## Architecture

```
DomainProfile (TOML)
  ↓
DocumentIngester / TableIngester
  ↓
StorageBackend (Protocol)
  ├── MemoryBackend     (testing)
  ├── SQLiteBackend     (FTS5, lightweight)
  ├── SqliteGraphBackend(+ graph traversal)
  ├── KuzuBackend       (embedded Cypher)
  ├── PostgreSQLBackend (pgvector)
  ├── QdrantBackend     (vector-only)
  └── CompositeBackend  (mix backends)
  ↓
3rd-gen retrieval pipeline
  ├── QueryAnchorExtractor   (categories + entities + keywords)
  ├── GraphExpander          (1-hop shallow expansion)
  ├── HybridReranker         (4-signal fusion)
  └── EvidenceAggregator     (MMR + per-doc cap + category coverage)
  ↓
Agent tools (7) → MCP server (8 tools) → LLM agent
```

### Brain-inspired modules (still available)

| Module | What it does |
|--------|-------------|
| `ResonanceScorer` | 4-axis ranking: relevance x importance x recency x vitality |
| `HebbianEngine` | Co-activation strengthening / weakening |
| `ConsolidationCascade` | L0 (raw) → L1 (sprint) → L2 (monthly) → L3 (permanent) |
| `OntologyRegistry` | Type hierarchy + relation constraints |
| `ActivityTracker` | Agent session / tool call / decision / outcome capture |
| `PPR` | Personalized PageRank for graph-aware discovery |

---

## Backends

| Backend | Graph Traversal | Vector Search | Scale | Use Case |
|---------|----------------|--------------|-------|----------|
| `MemoryBackend` | Python BFS | cosine | ~10K | Testing |
| `SQLiteBackend` | CTE recursive | - | ~100K | Lightweight |
| `SqliteGraphBackend` | + shortest_path | - | ~100K | **Recommended default** |
| `KuzuBackend` | Cypher (embedded) | HNSW | ~10M | Graph-heavy |
| `PostgreSQLBackend` | CTE recursive | pgvector | ~1M | Production |
| `CompositeBackend` | Kuzu | Qdrant | Unlimited | Scale-out |

---

## MCP Server

```bash
synaptic-mcp --db knowledge.db
synaptic-mcp --embed-url http://localhost:11434/v1 --embed-model qwen3-embedding:4b
```

**24 tools total:**
- Knowledge (7): search, add, link, reinforce, stats, export, consolidate
- Agent workflow (4): start_session, log_action, record_decision, record_outcome
- Semantic search (3): find_similar, get_reasoning_chain, explore_context
- Ontology (2): define_type, query_schema
- **Agent tools (8)**: search, expand, get_document, list_categories, count, search_exact, follow, session_info

---

## Dev

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q                   # 683+ tests
uv run ruff check --fix && uv run ruff format
```

## License

MIT
