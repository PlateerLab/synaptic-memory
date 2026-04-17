# Synaptic Memory

**Zero API calls at index time. Zero infra. Zero lock-in.**
A knowledge graph + MCP tool server for LLM agents, with hybrid retrieval,
CDC-based live database sync, and Korean FTS built in.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> [한국어 README](README.ko.md)

---

## 5-minute start

```bash
pip install "synaptic-memory[sqlite,korean,vector]"
python examples/quickstart.py
```

That command ingests [`examples/data/products.csv`](examples/data/products.csv)
into a SQLite-backed graph and runs three searches — all **without calling
any LLM** at indexing time. Full source: [`examples/quickstart.py`](examples/quickstart.py).

---

## Two calls to build a graph

```python
import asyncio
from synaptic import SynapticGraph

async def main():
    # Any data → knowledge graph (CSV, JSONL, directory)
    graph = await SynapticGraph.from_data("./my_data/")

    # Or directly from a database — SQLite / PostgreSQL / MySQL / Oracle / MSSQL
    graph = await SynapticGraph.from_database(
        "postgresql://user:pass@host:5432/dbname"
    )

    # Live database? Use CDC mode and only re-read what changed.
    graph = await SynapticGraph.from_database(
        "postgresql://user:pass@host:5432/dbname",
        db="knowledge.db",
        mode="cdc",       # deterministic node IDs + sync state recorded
    )
    result = await graph.sync_from_database(
        "postgresql://user:pass@host:5432/dbname"
    )
    print(result.added, result.updated, result.deleted)

    # Or bring your own chunker (LangChain, Unstructured, custom OCR, ...)
    chunks = my_parser.split("manual.pdf")
    graph = await SynapticGraph.from_chunks(chunks)

    # Search
    result = await graph.search("my question", engine="evidence")

asyncio.run(main())
```

That's it. Auto-detects file format or DB schema, generates an ontology profile, ingests, indexes, builds FK edges.

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
36 MCP tools → LLM agent explores via graph-aware multi-turn tool use
```

**Two jobs, nothing else:**
1. **Build the graph well** — cheap extraction, no LLM at index time
2. **Give the LLM good tools** — the agent decides what to search

---

## Install

```bash
# Recommended — covers every example in this README
pip install "synaptic-memory[sqlite,korean,vector,mcp]"

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
    result = await graph.search("my question", engine="evidence")
    for activated in result.nodes[:5]:
        print(activated.node.title, activated.activation)

asyncio.run(main())
```

### Option B: MCP server (Claude Desktop / Code)

```bash
synaptic-mcp --db my_graph.db
synaptic-mcp --db my_graph.db --embed-url http://localhost:11434/v1
```

Claude can now call 36 tools to explore your graph — search, ingest
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
    retriever = SynapticRetriever(graph=graph, k=5, engine="evidence")

    docs = await retriever.ainvoke("my question")
    for doc in docs:
        print(doc.page_content[:80], "   ", doc.metadata["score"])

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

---

## Indexing cost comparison

| Approach | LLM at indexing | Trade-off |
|----------|-----------------|-----------|
| GraphRAG-style (MS GraphRAG, Cognee, Graphiti) | LLM extracts entities + relations + community summaries | Highest recall on narrative corpora, but every new document costs LLM tokens |
| LightRAG-style | LLM deferred to query time | Less index cost, but each query pays |
| **Synaptic** | **None.** Structural + statistical signals only (FK, NEXT_CHUNK, phrase DF hubs, MENTIONS) | Cheapest, deterministic, but won't synthesize new relations on its own |

No LLM at indexing. The graph is a search index, not a knowledge base.
If you need LLM-synthesized summaries on top of the graph, layer them
with your own agent — Synaptic gives you the primitives and leaves
the synthesis choice to you.

> **v0.15.0**: pass `engine="evidence"` to `graph.search()` to use the
> hybrid pipeline (BM25 + HNSW + PPR + cross-encoder + MMR). The
> default flips to `"evidence"` in v0.16.0 and the legacy engine is
> removed in v0.17.0. A migration guide will ship with v0.16.0.

---

## Agent Tools (36 total)

### Text search tools
| Tool | Purpose |
|------|---------|
| `deep_search` | **Recommended.** Search → expand → read documents in ONE call |
| `compare_search` | Auto-decompose multi-topic queries, search in parallel |
| `knowledge_search` | Core semantic search (routes through EvidenceSearch in v0.14.2+) |
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

### Ingest / CDC tools (v0.14.0+)
Mid-conversation ingestion so Claude can teach itself new material without leaving the chat.

| Tool | Purpose |
|------|---------|
| `knowledge_add_document` | Ingest a long-text document with automatic sentence-boundary chunking |
| `knowledge_add_table` | Ingest structured rows → ENTITY nodes + FK edges |
| `knowledge_add_chunks` | BYO-chunker path for pre-split content |
| `knowledge_ingest_path` | Ingest a CSV / JSONL / text file from the local filesystem |
| `knowledge_remove` | Delete a single node with edge cascade |
| `knowledge_sync_from_database` | Incremental sync from a live database (CDC) |
| `knowledge_backfill` | Repair graphs missing embeddings or phrase hubs (v0.14.4+) |

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

### Reproducible FTS-only baseline (< 2 seconds on a laptop)

```bash
pip install "synaptic-memory[korean]"
python examples/benchmark_allganize.py
```

Output (deterministic, v0.16.0):

```
Dataset                  Corpus  Queries      MRR     R@10        Hit     Time
--------------------------------------------------------------------------------
Allganize RAG-ko            200      200    0.947    1.000   200/200     9.3s
Allganize RAG-Eval          300      300    0.911    0.950   285/300     5.9s
```

This is the **embedder-free baseline** (EvidenceSearch pipeline: BM25 +
PPR + MMR, no vector index, no cross-encoder). Full source:
[`examples/benchmark_allganize.py`](examples/benchmark_allganize.py).
Data source: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO).

> **v0.16.0 — engine default flipped to `"evidence"`.** Combined with
> the v0.15.1 query-mode Kiwi improvement, FTS-only Korean retrieval
> moved from **Allganize RAG-ko MRR 0.621 (v0.15.0) → 0.947
> (v0.16.0)** without any embedder or reranker. English (HotPotQA-24)
> 0.727 → 0.875. Full ablation:
> [`examples/ablation/run_ablation.py`](examples/ablation/run_ablation.py).
> Reproducibility under streaming ingest sharpens in lockstep: top-1
> rank invariance rose from 54.5 % → **100 %**, bit-wise top-10
> identical from 51.5 % → **96 %**, with MRR drift exactly zero.

### Head-to-head vs Mem0 / Cognee / HippoRAG2

A runnable harness (BEIR-style corpus, same MRR / R@10 scoring code
for every system):

```bash
# Synaptic only (no API keys, ~2 s)
python examples/benchmark_vs_competitors/run_comparison.py --only synaptic

# Adapters for Mem0, Cognee, HippoRAG2 ship in-tree — add them to
# --only when the respective packages are installed and API keys are set
python examples/benchmark_vs_competitors/run_comparison.py --only synaptic,mem0 --subset 10
```

See [examples/benchmark_vs_competitors/README.md](examples/benchmark_vs_competitors/README.md)
for fairness caveats. Competitor self-reported numbers (Mem0 LoCoMo
91.6, HippoRAG2 MuSiQue F1 51.9, the Zep 84→58 correction incident,
etc.) are catalogued with sources in
[docs/comparison/published_numbers.md](docs/comparison/published_numbers.md).

### Embedder-free full-dataset summary (v0.16.0)

Run via `python examples/ablation/run_ablation.py`:

| Dataset | Lang | Queries | MRR | Hit @ 10 |
|---------|------|---------|-----|----------|
| Allganize RAG-ko | ko | 200 | **0.947** | 200/200 |
| Allganize RAG-Eval | ko | 300 | **0.911** | 285/300 |
| AutoRAG KO | ko | 114 | **0.906** | 114/114 |
| PublicHealthQA KO | ko | 77 | **0.546** | 64/77 |
| HotPotQA-24 EN | en | 24 | **0.875** | 24/24 |

### English multi-hop standard benchmarks (v0.16.0, subset)

Run via:

```bash
pip install "synaptic-memory[eval]"   # adds `datasets` for HuggingFace download
python examples/ablation/download_benchmarks.py
python examples/ablation/run_tier1_benchmarks.py --subset 500
```

Adds HotPotQA-dev (66 k corpus), MuSiQue-Ans-dev (21 k), and
2WikiMultihopQA-dev (57 k) — the three retrieval corpora the
HippoRAG / GraphRAG line of research uses for head-to-head.
Numbers go in
[docs/comparison/synaptic_results.md](docs/comparison/synaptic_results.md#tier-15--english-multi-hop-standard-benchmarks-v0160).

### Full pipeline (embedder + reranker) — pre-v0.16.0 measurements

The numbers below predate the v0.16.0 engine flip. They were
measured with the **EvidenceSearch pipeline plus an embedder
(Ollama `qwen3-embedding:4b`) and a cross-encoder reranker
(TEI `bge-reranker-v2-m3`)**, which is why they match or beat the
embedder-free numbers above on private corpora. Reproducing these
requires a GPU-backed embedder and reranker — see
[`eval/run_all.py`](eval/run_all.py). A v0.16.0 rerun is scheduled
after the Home-server rebuild.

| Dataset | Type | Nodes | MRR | Hit |
|---------|------|-------|-----|-----|
| KRRA Easy | Korean documents (private) | 19,720 | **0.967** | 20/20 |
| KRRA Hard | Korean documents (private) | 19,720 | **1.000** | 15/15 |
| X2BEE Easy | PostgreSQL e-commerce (private) | 19,843 | **1.000** | 20/20 |
| assort Easy | Fashion CSV (private) | 13,909 | **0.867** | 13/15 |
| HotPotQA-24 | English multi-hop (public subset) | 226 | **0.964** | 24/24 |

> HotPotQA-24 is a 24-question subset. A full HotPotQA-dev (7,405 q)
> run is planned for v0.16.1 — we won't claim parity with published
> numbers until then.

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
Agent tools (36) → MCP server → LLM agent
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
| `docs` | xgen-doc2chunk for PDF/DOCX/PPTX/XLSX/HWP loading |

---

## Documentation

| Doc | What it is |
|-----|-----------|
| [docs/GUIDE.md](docs/GUIDE.md) | Friendly intro — what/why/how, zero jargon (Korean) |
| [docs/TUTORIAL.en.md](docs/TUTORIAL.en.md) | **30-minute hands-on walkthrough (English)** |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | 30-minute hands-on walkthrough (Korean) |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | 3rd-gen GraphRAG + pipeline internals |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Original neural-inspired design |
| [docs/COMPARISON.md](docs/COMPARISON.md) | vs GraphRAG / LightRAG / LazyGraphRAG |
| [docs/comparison/synaptic_results.md](docs/comparison/synaptic_results.md) | Reproducible Synaptic numbers with provenance |
| [docs/comparison/published_numbers.md](docs/comparison/published_numbers.md) | Competitor self-reported numbers (with sources) |
| [docs/paper/draft.md](docs/paper/draft.md) | arXiv preprint draft — Streaming Retrieval with Top-K Invariance |
| [docs/paper/theorem.md](docs/paper/theorem.md) | Formal theorem + proof sketch |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Future plans |

## Dev

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q                   # 809+ tests
uv run ruff check --fix
```

## License

MIT
