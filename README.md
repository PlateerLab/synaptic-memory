# Synaptic Memory

**Zero API calls at index time. Zero infra. Zero lock-in.**
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

> **v0.16.0+**: `graph.search()` defaults to the hybrid
> EvidenceSearch pipeline (BM25 + HNSW + PPR + cross-encoder + MMR).
> `engine="legacy"` still works but raises `DeprecationWarning`;
> removal is pushed to v0.18.0 to bundle with HippoRAG2-style
> architecture work.

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
| `top_nodes` | **Single-call top-N ranking** — "가장 X한" / "top N" / "최대/최소" / "최근" questions without composing aggregate_nodes. Each row carries `sort_value` for chaining into join_related / filter_nodes(from_ids=...). v0.18-β2+. |

All four structured tools emit `hints` on 0-result returns (alternate
operator, dropped WHERE, fuzzy column match) so the agent's next turn
gets a concrete corrective action instead of a retry loop.

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

### RAG vs synaptic-memory — multi-hop retrieval (v0.24)

Vanilla RAG (chunk → top-k → one LLM answer) measured head-to-head
against the synaptic-memory agent on the same corpus, ground truth,
LLM-judge, and model (`finreg` — 4,417 Korean financial-statute
articles):

```
Query type                 vanilla RAG    synaptic-memory
-----------------------------------------------------------
single-hop (1 article)          94%             94%
multi-hop (follow citation)      0%             83%
```

A statute article that cites another article ("제30조에 따라 …") is a
**multi-hop** query: the cited provision shares no query vocabulary, so
single-shot retrieval structurally cannot reach it. synaptic-memory
turns cross-references into `REFERENCES` graph edges
([`StructuralReferenceLinker`](src/synaptic/extensions/structural_reference_linker.py),
LLM-free, auto-derived from the corpus) and the agent follows them.
Full report — including limits and what does *not* help:
[`docs/REPORT-rag-vs-synaptic.md`](docs/REPORT-rag-vs-synaptic.md).

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

### Full pipeline (BGE-M3 + cross-encoder) — v0.17.1

Measured 2026-04-19 on an H100, `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3`
loaded in-process via `transformers` (no external TEI / Ollama
endpoint needed):

```bash
python eval/run_all.py --quick --local-bge
```

#### 14-bench single-shot (5 public + 9 private)

| Dataset | Lang | Queries | FTS-only | Full pipeline | Δ |
|---------|------|--------:|---------:|--------------:|---:|
| HotPotQA-24 | en | 24 | 0.875 | **0.979** | +0.104 |
| Allganize RAG-ko | ko | 200 | 0.947 | **0.983** | +0.036 |
| Allganize RAG-Eval | ko | 300 | 0.911 | **0.955** | +0.044 |
| PublicHealthQA | ko | 77 | 0.547 | **0.748** | +0.201 |
| AutoRAG KO | ko | 114 | **0.906** | 0.806 | −0.100 ⚠ |
| KRRA Easy | ko | 20 | 0.967 | **0.975** | +0.008 |
| KRRA Hard | ko | 40 | 0.583 | **0.589** | +0.006 |
| KRRA Conv | ko | 30 | 0.146 | **0.166** | +0.020 |
| assort Easy | ko | 15 | 0.760 | **0.856** | +0.096 |
| assort Hard | ko | 40 | 0.000 | 0.000 | 0 |
| assort Conv | ko | 30 | 0.425 | **0.472** | +0.047 |
| X2BEE Easy | en | 20 | 1.000 | 1.000 | 0 |
| X2BEE Hard | en/ko | 20 | **0.379** | 0.368 | −0.011 |
| X2BEE Conv | en/ko | 30 | 0.167 | 0.164 | −0.003 |
| **Mean** | | | 0.615 | **0.647** | **+0.032 (+5.2pp)** |

v0.17.1 is the **first release where mean Full-pipeline MRR exceeds
mean FTS-only MRR** across all 14 benches (v0.17.0 was net −1.1 %).
12/14 benches improve or hold; the 3 mild residual regressions
(X2BEE Hard / Conv, AutoRAG) are at or below single-query noise
except for AutoRAG where the structural reranker mismatch persists.

**When the cross-encoder helps — and when it hurts.** The reranker
shines on paraphrase-heavy corpora (PublicHealthQA +20 pp,
HotPotQA +10 pp). On retrieval-style corpora where FTS ranking is
already near-optimal (AutoRAG, X2BEE Hard) it injects noise that
displaces the gold rank. v0.17.1's adaptive blend (
`std/3` discriminator) and structured-row reranker skip recover
most of that — AutoRAG went −0.264 → −0.100 — but pure
**`reranker=None` is the only way to hit the FTS-only ceiling** on
those corpora. Diagnostic: [`examples/ablation/diagnose_autorag.py`](examples/ablation/diagnose_autorag.py).

#### Multi-turn agent (Qwen3.5-27B via vLLM, 5 turns, LLM-judge)

| Dataset | Single-shot | Agent solved | vs v0.13 (gpt-4o-mini) |
|---------|------------:|------------:|----------------------:|
| KRRA Hard | 0.589 | 30/39 (77 %) | 11/15 (73 %) +4 pp |
| assort Hard | 0.000 | 30/33 (91 %) | 13/15 (87 %) +4 pp |
| X2BEE Hard | 0.368 | **19/19 (100 %)** | 17/19 (89 %) +11 pp |
| KRRA Conv | 0.166 | 14/30 (47 %) | 21/30 (70 %) −23 pp |
| assort Conv | 0.472 | 22/24 (92 %) | 20/24 (83 %) +9 pp |
| X2BEE Conv | 0.164 | 25/27 (93 %) | 22/27 (81 %) +12 pp |
| **Mean** | | **140/172 = 81 %** | |

**This is Synaptic's real number.** Single-shot retrieval is the
floor; the multi-turn agent (`deep_search` + `compare_search` +
graph-context injection) brings the same questions from 0–47 %
to 47–100 %. assort Hard 0/40 → 91 % under agent shows what the
graph + structured tools can do that no single-shot pipeline can.
5 of 6 benches beat the v0.13 GPT-4o-mini baseline (Qwen3.5-27B
upgrade). KRRA Conv regression (Qwen Korean conversational gap)
is the one open issue — v0.18 track.

#### Known structural gap — MuSiQue (English multi-hop)

MuSiQue-Ans-dev 500q full pipeline R@5 **0.453** vs HippoRAG2's
published **0.747** (−0.294). Three rounds of targeted fixes (LLM
query decomposition, inline phrase hub, DF-filtered entity linker)
all regressed the score — the gap is structural. Closing it
requires OpenIE triple extraction + query→triple dense linking,
which is a v0.18.0+ research track rather than a default pipeline
change. Synaptic's strength is Korean / structured-data RAG;
English Wikipedia multi-hop is honestly documented as a trade-off.
See [`docs/PLAN-v0.18-architecture.md`](docs/PLAN-v0.18-architecture.md#q2--indexing--llm-free-유지-vs-selective-llm-도입).

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
| [docs/REPORT-rag-vs-synaptic.md](docs/REPORT-rag-vs-synaptic.md) | **RAG vs synaptic-memory — measured head-to-head (multi-hop)** |
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

Apache-2.0 — see [LICENSE](LICENSE). Permits commercial use, modification, and
redistribution as long as the copyright/attribution notice is preserved.
