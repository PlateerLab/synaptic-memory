# Synaptic Memory

**Default path: zero API calls at index time. Zero infra. Zero lock-in.**
A knowledge graph + MCP tool server for LLM agents, with hybrid retrieval,
CDC-based live database sync, and Korean FTS built in.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> [한국어 README](README.ko.md)

---

## 5-minute start

```bash
pip install "synaptic-memory[sqlite,korean,vector]"
synaptic-quickstart --db quickstart.db
```

That command builds a tiny SQLite-backed graph and runs three searches — all
**without calling any LLM** at indexing time. Omit `--db` for an in-memory,
zero-dependency smoke test. Full source for the expanded example:
[`examples/quickstart.py`](examples/quickstart.py).

---

## Build and search

```python
import asyncio
from synaptic import SynapticGraph

async def main():
    # Any data → knowledge graph (CSV, JSONL, directory)
    graph = await SynapticGraph.from_data("./my_data/", preset="rag")
    try:
        result = await graph.search("my question")
        print(result.nodes[0].node.title if result.nodes else "no result")
    finally:
        await graph.close()

asyncio.run(main())
```

That's it. Auto-detects file format or DB schema, generates an ontology profile, ingests, indexes, builds FK edges.

Presets keep the common knobs compact:

```python
# local: deterministic, no external services (default)
graph = await SynapticGraph.from_chunks(chunks, preset="local")

# rag: reads SYNAPTIC_EMBED_URL / SYNAPTIC_RERANK_URL if set
graph = await SynapticGraph.from_data("./docs/", preset="rag")

# agent: rag + deterministic component bridging for multi-turn exploration
graph = await SynapticGraph.from_data("./docs/", preset="agent")
```

> **Live database sync (CDC)** — `mode="cdc"` enables incremental
> updates: tables with an `updated_at`-style column are read with a
> watermark filter, others fall back to per-row content hashing.
> Deletes are detected via a TEMP TABLE LEFT JOIN; FK rewires
> re-link the corresponding RELATED edges. Search results are
> identical to a full reload (locked in by a regression test).
> Supports SQLite, PostgreSQL, MySQL/MariaDB.

> **Office files (PDF/DOCX/PPTX/XLSX/HWP)** are supported through the **optional** `xgen-doc2chunk` package. Install with `pip install synaptic-memory[docs]` or use `from_chunks()` with your own parser.

---

## What it does

```
Your data (CSV, JSONL, PDF/DOCX/PPTX/XLSX/HWP, SQL database)
  ↓  auto-detect format / auto-discover DB schema + FKs
  ↓  DocumentIngester (text) / TableIngester / DbIngester
  ↓
Knowledge Graph
  ├─ Documents: Category → Document → Chunk
  └─ Structured: table rows as ENTITY nodes + RELATED edges (FKs)
  ↓
MCP tools → LLM agent explores via graph-aware multi-turn tool use
```

**Two jobs, nothing else:**
1. **Build the graph well** — cheap deterministic extraction by default
2. **Give the LLM good tools** — the agent decides what to search

---

## Install

```bash
# Recommended local graph + MCP setup
pip install "synaptic-memory[sqlite,korean,vector,mcp]"

# Add this for the LangChain retriever example
pip install "synaptic-memory[langchain]"

# Or everything, including Postgres / Kuzu / Qdrant / MinIO
pip install "synaptic-memory[all]"
```

<details>
<summary>Pick-your-own extras</summary>

```bash
pip install synaptic-memory                # Core (zero deps, in-memory only)
pip install synaptic-memory[sqlite]        # + SQLite FTS5 backend
pip install synaptic-memory[korean]        # + Kiwi morphological analyzer
pip install synaptic-memory[vector]        # + usearch HNSW index
pip install synaptic-memory[mcp]           # + MCP server for Claude
pip install synaptic-memory[embedding]     # + aiohttp for embedding APIs
pip install synaptic-memory[reranker]      # + flashrank cross-encoder
pip install synaptic-memory[langchain]     # + LangChain retriever adapter
pip install synaptic-memory[postgresql]    # + asyncpg + pgvector
pip install synaptic-memory[docs]          # + xgen-doc2chunk (PDF/DOCX/PPTX/XLSX/HWP)
```

</details>

---

## Quick Start

### Option A: Two lines (easiest)

```python
import asyncio
from synaptic import SynapticGraph

async def main():
    # CSV file
    graph = await SynapticGraph.from_data("products.csv")
    try:
        result = await graph.search("my question")
        for activated in result.nodes[:5]:
            print(activated.node.title, activated.activation)
    finally:
        await graph.close()

asyncio.run(main())
```

You can pass `preset="rag"` to read `SYNAPTIC_EMBED_URL` and
`SYNAPTIC_RERANK_URL`, or use `GraphBuildOptions` when you want one reusable
configuration object across `from_data()`, `from_chunks()`, and
`from_database()`.

### Option B: MCP server (Claude Desktop / Code)

```bash
synaptic-mcp --db my_graph.db
synaptic-mcp --db my_graph.db --embed-url http://localhost:11434/v1
```

Claude can now call MCP tools to explore your graph — search, ingest
new files into the graph mid-conversation, and sync from a live
database without dropping to a CLI.

A ready-to-paste `claude_desktop_config.json` snippet is in
[`examples/mcp_claude_desktop.json`](examples/mcp_claude_desktop.json).

### Option BX: LangChain retriever (drop-in)

```bash
pip install "synaptic-memory[sqlite,korean,vector,langchain]"
```

```python
import asyncio
from synaptic import SynapticGraph
from synaptic.integrations.langchain import SynapticRetriever

async def main():
    graph = await SynapticGraph.from_data("./docs/")
    try:
        retriever = SynapticRetriever(graph=graph, k=5)
        docs = await retriever.ainvoke("my question")
        for doc in docs:
            print(doc.page_content[:80], "   ", doc.metadata["score"])
    finally:
        await graph.close()

asyncio.run(main())
```

Runnable example: [`examples/langchain_retriever.py`](examples/langchain_retriever.py).
Each hit becomes a LangChain `Document` with the node id, title,
score, and any structured properties in `metadata` — works unmodified
in RetrievalQA chains, agents, and RAG graphs.

### Option C: Full control

```python
import asyncio
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.document_ingester import DocumentIngester, JsonlDocumentSource

async def main():
    profile = DomainProfile.load("my_profile.toml")
    backend = SqliteGraphBackend("graph.db")
    await backend.connect()

    source = JsonlDocumentSource("docs.jsonl", "chunks.jsonl")
    ingester = DocumentIngester(profile=profile, backend=backend)
    await ingester.ingest(source)

asyncio.run(main())
```

### Option D: relation ontology (multi-hop)

When your documents cite each other by a canonical identifier — statute
article numbers, standard clause codes, manual section ids — Synaptic
turns those citations into `REFERENCES` graph edges so the agent can
follow them. Still **zero LLM at index time** — it is rule-based
extraction, auto-derived from your corpus's own identifier values.

```python
from synaptic import SynapticGraph
from synaptic.extensions.domain_profile import DomainProfile

# Each document carries its identifier in an `article_no` property,
# scoped by `law`. That is all the configuration needed.
graph = await SynapticGraph.from_data(
    "./statutes.jsonl",
    profile=DomainProfile.with_references(
        key_property="article_no", scope_property="law"
    ),
)
```

The linker is **self-gating**: if the corpus has no clean identifier
inventory it writes nothing (no-op), so this is safe to leave on.
Measured impact — on a financial-statute multi-hop corpus, retrieval
goes from **32% (standard dense RAG)** / **31% (HippoRAG2)** to
**73% (synaptic)** — 2.3× the nearest competitor:
[`docs/REPORT-rag-vs-synaptic.md`](docs/REPORT-rag-vs-synaptic.md).

> Requirement: each document must carry the identifier as a node
> property. Structured input (JSONL with `properties`) provides it
> directly; raw text files do not — relation linking needs identifiers.

---

## Indexing cost comparison

| Approach | LLM at indexing | Trade-off |
|----------|-----------------|-----------|
| GraphRAG-style (MS GraphRAG, Cognee, Graphiti) | LLM extracts entities + relations + community summaries | Highest recall on narrative corpora, but every new document costs LLM tokens |
| LightRAG-style | LLM deferred to query time | Less index cost, but each query pays |
| **Synaptic default** | **None.** Structural + statistical signals (FK, NEXT_CHUNK, phrase DF hubs, MENTIONS) + rule-based REFERENCES edges | Cheapest, deterministic; extracts explicit cross-references without LLM calls |

By default, indexing is LLM-free. The graph is a search index, not a knowledge base.
Cross-references that documents state explicitly (statute article
citations, clause codes) are turned into `REFERENCES` edges with
zero LLM (see [Option D](#option-d-relation-ontology-multi-hop)).
If you opt into OpenIE, Synaptic can add bounded, revertible
LLM-extracted semantic relations; that is not part of the default
deterministic path.

> **Current API**: `graph.search()` has one retrieval path: the hybrid
> EvidenceSearch pipeline (BM25 + HNSW + PPR + cross-encoder + MMR).
> The old `engine=` switch was removed, so examples should call
> `graph.search("question")` directly.

---

## Agent Tools

### Text search tools
| Tool | Purpose |
|------|---------|
| `deep_search` | **Recommended.** Search → expand → read documents in ONE call |
| `compare_search` | Auto-decompose multi-topic queries, search in parallel |
| `knowledge_search` | Core semantic search through EvidenceSearch |
| `agent_search` | FTS + vector hybrid search with intent routing |
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
| `top_nodes` | **Single-call top-N ranking** — "가장 X한" / "top N" / "최대/최소" / "최근" questions without composing aggregate_nodes. Each row carries `sort_value` for chaining into join_related / filter_nodes(from_ids=...). |

All four structured tools emit `hints` on 0-result returns (alternate
operator, dropped WHERE, fuzzy column match) so the agent's next turn
gets a concrete corrective action instead of a retry loop.

### Ingest / CDC tools
Mid-conversation ingestion so Claude can teach itself new material without leaving the chat.

| Tool | Purpose |
|------|---------|
| `knowledge_add_document` | Ingest a long-text document with automatic sentence-boundary chunking |
| `knowledge_add_table` | Ingest structured rows → ENTITY nodes + FK edges |
| `knowledge_add_chunks` | BYO-chunker path for pre-split content |
| `knowledge_ingest_path` | Ingest a CSV / JSONL / text file from the local filesystem |
| `knowledge_remove` | Delete a single node with edge cascade |
| `knowledge_sync_from_database` | Incremental sync from a live database (CDC) |
| `knowledge_backfill` | Repair graphs missing embeddings or phrase hubs |

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
  ↓  HybridReranker (lexical + semantic + graph + structural + memory + authority + temporal)
  ↓  MaxP document aggregation (coverage bonus)
  ↓  Cross-encoder reranker (bge-reranker-v2-m3 via TEI, optional)
  ↓  EvidenceAggregator (MMR diversity + per-doc cap + category coverage)
Result
```

**Usage/time memory axis (opt-in, off by default).** The reranker carries a
fifth weighted signal — `memory` — that scores each node by *how it has been
used*: importance (reinforced successes vs failures), recency (`updated_at`),
and vitality. With `memory=0.0` (the default) ranking is unchanged. Turn it on
and retrieval *evolves* — reinforcing the results that answered a query lifts
them on later searches, and decayed nodes fade, which a static index cannot do.

```python
from synaptic.extensions.hybrid_reranker import RerankerWeights

# Enable the memory axis (rebalance the others so weights still sum to ~1).
graph.reranker_weights = RerankerWeights(
    lexical=0.35, semantic=0.20, graph=0.10, structural=0.10, memory=0.25,
)
await graph.reinforce([node_id], success=True)  # this result helped → lift it next time
```

### Memory operating layer

Retrieval can be observed without making every search stateful:

```python
from synaptic import FeedbackSignal, MemoryScope

scope = MemoryScope(workspace_id="docs", user_id="alice")
result = await graph.search("refund exception", record=True, scope=scope)

await graph.record_feedback(
    event_id=result.event_id,
    signal=FeedbackSignal.EXPLICIT_POSITIVE,
    success=True,
    scope=scope,
)

health = await graph.memory_health(scope=scope)
signals = await graph.scan_memory_signals(scope=scope)
```

Events, feedback, provenance, and health signals are stored as graph metadata.
They are not appended to `Node.content`, and they are not automatically dumped
into LLM prompts.

---

## Benchmarks And Reports

The root README stays current with the install path and public API. Detailed
numbers are versioned in reports so old measurements do not look like the
current API contract.

Run a quick local smoke:

```bash
synaptic-quickstart --json
```

Run the lightweight Korean FTS benchmark:

```bash
pip install "synaptic-memory[korean]"
python examples/benchmark_allganize.py
```

Run the competitor harness when optional packages/API keys are available:

```bash
python examples/benchmark_vs_competitors/run_comparison.py --only synaptic
```

Reference reports:

| Report | What it covers |
|--------|----------------|
| [docs/comparison/synaptic_results.md](docs/comparison/synaptic_results.md) | Reproducible Synaptic benchmark results with provenance |
| [docs/REPORT-rag-vs-synaptic.md](docs/REPORT-rag-vs-synaptic.md) | RAG vs synaptic-memory on multi-hop financial-statute retrieval |
| [docs/REPORT-memory-operating-layer-eval.md](docs/REPORT-memory-operating-layer-eval.md) | Memory operating layer evaluation and health/reporting gates |
| [examples/benchmark_vs_competitors/README.md](examples/benchmark_vs_competitors/README.md) | Fairness caveats for competitor adapters |

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
Agent tools → MCP server → LLM agent
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
| `langchain` | LangChain retriever adapter |
| `postgresql` | asyncpg + pgvector |
| `docs` | xgen-doc2chunk for PDF/DOCX/PPTX/XLSX/HWP loading |

---

## Documentation

| Doc | What it is |
|-----|-----------|
| [docs/GUIDE.md](docs/GUIDE.md) | Friendly intro — what/why/how, zero jargon (Korean) |
| [docs/TUTORIAL.en.md](docs/TUTORIAL.en.md) | **30-minute hands-on walkthrough (English)** |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | 30-minute hands-on walkthrough (Korean) |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | 3rd-gen GraphRAG + pipeline internals |
| [docs/REPORT-rag-vs-synaptic.md](docs/REPORT-rag-vs-synaptic.md) | **RAG vs synaptic-memory — measured head-to-head (multi-hop)** |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Original neural-inspired design |
| [docs/COMPARISON.md](docs/COMPARISON.md) | vs GraphRAG / LightRAG / LazyGraphRAG |
| [docs/comparison/synaptic_results.md](docs/comparison/synaptic_results.md) | Reproducible Synaptic numbers with provenance |
| [docs/comparison/published_numbers.md](docs/comparison/published_numbers.md) | Competitor self-reported numbers (with sources) |
| [docs/paper/draft.md](docs/paper/draft.md) | arXiv preprint draft — Streaming Retrieval with Top-K Invariance |
| [docs/paper/theorem.md](docs/paper/theorem.md) | Formal theorem + proof sketch |
| [docs/ADOPTION.md](docs/ADOPTION.md) | Install, presets, and first integration path |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Historical roadmap |

## Dev

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q
uv run ruff check --fix
```

## License

Apache-2.0 — see [LICENSE](LICENSE). Permits commercial use, modification, and
redistribution as long as the copyright/attribution notice is preserved.
