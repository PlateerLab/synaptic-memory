# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.14.0] - 2026-04-14

### Added — Live database CDC (Change Data Capture)

- **`SynapticGraph.from_database(mode="cdc")`** — opt-in deterministic
  node IDs derived from `(source_url, table, primary_key)`. Re-running
  the same source against the same graph file is now a true upsert;
  random-UUID `mode="full"` is preserved as the default for one-shot
  exports.
- **`SynapticGraph.sync_from_database(dsn)`** — incremental sync.
  Tables with an `updated_at`-style column use a `WHERE col >= watermark`
  filter; tables without one fall back to per-row content hashing.
  Both strategies share delete detection (TEMP TABLE + LEFT JOIN, no
  Python set diff) and FK edge re-computation.
- **`mode="auto"`** — uses prior CDC state when the graph file has
  one, otherwise falls back to `mode="full"`.
- **Supported dialects**: SQLite, PostgreSQL, MySQL/MariaDB.
  Oracle / MSSQL still use the legacy full-reload path.
- **Self-contained graph files**: CDC bookkeeping (`syn_cdc_state`,
  `syn_cdc_pk_index`) lives inside the same SQLite file as the graph,
  so a single `.db` round-trips cleanly.
- **Search regression test** locks in that `mode="cdc"` returns the
  same top-k results as `mode="full"` — CDC only changes node
  identification, never search algorithms.
- **Production validation** against X2BEE PostgreSQL (19,843 rows,
  7 tables): initial CDC load 51s, idempotent re-sync **6s** (~6×
  faster than full reload), 4/4 user queries return identical
  top-1 results vs `mode="full"`.

### Added — MCP ingest + CDC tools

Brings knowledge-base maintenance into the MCP tool surface so
Claude (or any MCP client) can ingest and sync from live data
without dropping to a CLI. Tool count 29 → 35.

- **`knowledge_add_document`** — wraps `graph.add_document()` with
  automatic sentence-boundary chunking and the PART_OF /
  NEXT_CHUNK edge scaffolding.
- **`knowledge_add_table`** — wraps `graph.add_table()`: column
  definitions + row list → typed ENTITY nodes, FK edges, and
  auto-registration of the table schema in the ontology.
- **`knowledge_add_chunks`** — BYO-chunker path. Accepts a list of
  `{title, content, tags, source, properties}` dicts for users
  whose upstream tooling (LangChain, Unstructured, custom OCR)
  already produced chunks.
- **`knowledge_ingest_path`** — ingest a single CSV / JSONL / text
  file from the local filesystem into the current graph. Uses
  sync helpers to keep the async tool body free of blocking I/O.
- **`knowledge_remove`** — single-node deletion with edge cascade.
  Bulk removal is intentionally not exposed.
- **`knowledge_sync_from_database`** — CDC incremental sync from
  MCP. First call seeds state, subsequent calls read only changed
  rows. Accepts a per-call `connection_string` or falls back to
  the new `--source-dsn` CLI flag.
- **`--source-dsn` CLI flag** on `synaptic-mcp` for binding a
  default CDC source.
- MCP graph now uses a `ChunkEntityIndex` so `add_document`
  produces nodes of `NodeKind.CHUNK` (required for the PART_OF
  validation path).
- `build_agent_ontology()` gains `document` / `chunk` types and
  the existing `part_of` constraint is widened so chunk → chunk
  edges validate alongside the existing agent_activity → session
  rule. Required because `validate_edge` AND-s across every
  matching constraint; a single permissive rule is the only way
  to express an OR between two legal shapes.

### Fixed — CDC bugs caught by production validation

- **Canonical PK normalization** (`canonical_pk()` in
  `extensions/cdc/ids.py`): the row-read path went through
  `_cast_row` (numeric → `float(1.0)` → `'1.0'`) while the PK-read
  path used raw asyncpg (`Decimal('1')` → `'1'`). The `LEFT JOIN`
  in `find_deleted_pks` therefore matched none of the rows, and
  every sync flapped the table by re-deleting + re-inserting
  every row. Integer-valued floats / Decimals now collapse to a
  single canonical string used everywhere a PK is hashed,
  stored, looked up, or compared.
- **Skip CDC sync for tables without a real primary key**
  (`TableSchema.has_explicit_pk` propagated by every schema
  reader): falling back to `columns[0]` collapses rows whose
  fallback column isn't unique (e.g. an AWS DMS validation
  table with 47 rows but only 1 distinct `TASK_NAME`) into a
  single deterministic node ID, losing 46 rows and producing
  endless update churn. Such tables are now skipped with a
  clear warning + error entry in the `SyncResult`; users can
  still ingest them via legacy `mode="full"`.

## [0.13.0] - 2026-04-13

### Added — Graph-aware agent search + structured data tools

- **`SynapticGraph.from_database()`** — one-line DB → ontology migration.
  Supports SQLite, PostgreSQL, MySQL, Oracle, SQL Server. Auto-discovers
  schema, foreign keys, and M:N join tables (2+ FKs → RELATED edges
  instead of intermediate nodes). Batch processing (10K rows default).
- **Structured data tools** — `filter_nodes`, `aggregate_nodes`,
  `join_related` for SQL-like queries on graph-stored tables. All three
  now return `{total, showing, truncated}` for accurate counting.
- **`aggregate_nodes` WHERE pre-filter** — conditional aggregation
  (`where_property`/`where_op`/`where_value`). Enables "count 5-star
  reviews per product" in one call.
- **Graph-aware expansion for structured data** — `GraphExpander` now
  follows RELATED edges for ENTITY nodes, so search surfaces FK-linked
  rows (product → sales, product → reviews) automatically.
- **`join_related` edge-first strategy** — walks RELATED edges when
  available, falls back to property scan. O(degree) instead of O(N).
- **Graph composition hint** in `build_graph_context()` — tells the
  agent which tools fit the data (documents → search, structured →
  filter/aggregate/join). Distinguishes mixed graphs.
- **Foreign key metadata** surfaced in graph context — agents see
  `table.column → target_table` mappings automatically.
- **Table schema metadata** — column names, sample values, row counts
  for every structured table, auto-injected into agent system prompt.
- **Value-centric row content** — `TableIngester` now orders row values
  by semantic priority (name > description > category > rest), giving
  search the most meaningful tokens first. Removes `key=value` noise
  from content generation.
- **`SearchSession.expanded_nodes`** — tracks which nodes the agent has
  already expanded for better multi-turn coordination.
- **LLM-as-Judge evaluation** — `eval/run_all.py --judge` adds
  semantic answer validation alongside ID matching. Essential for
  filter/aggregate queries where "correct but different IDs" is common.
- **X2BEE benchmark dataset** — 40 queries (20 easy + 20 hard) over
  real production AWS RDS PostgreSQL (19,843 rows from ai_lab_main).

### Changed

- **`build_graph_context()`** — now includes structured data schemas
  and FK relationships in addition to categories. Composition section
  tells agents which tools match their query type.
- **Agent system prompt** — explicit guidance on tool selection,
  fallback strategies (try English keywords when Korean fails), and
  structured data patterns (node title format, FK chaining).
- **`HybridReranker._REASON_PRIOR`** — added `"related": 0.50` for
  RELATED edge expansion priors.
- **Public dataset runner** — now uses `EvidenceSearch` pipeline with
  optional embeddings/reranker, matching custom dataset quality.

### Fixed

- `filter_nodes` no longer early-breaks at limit, so total counts
  reported to agents are accurate.
- `aggregate_nodes` groups now include `node_title` field for FK group
  values, eliminating `goodss:` / `pr_product_base:` heuristic failures.
- `from_database()` async row_reader for PostgreSQL (asyncpg returns
  coroutines where aiosqlite returns sync iterators).

### Performance

- Agent benchmarks:
  - X2BEE Hard: 1/19 (5%) → **17/19 (89%)**
  - assort Hard: 1/15 (7%) → **12/15 (80%)**
  - KRRA Hard MRR: 0.808 → **1.000** (15/15 hit)
- Public benchmarks with EvidenceSearch + embed + reranker:
  - HotPotQA-24: 0.727 → **0.964**
  - Allganize RAG-ko: 0.621 → **0.905**
  - PublicHealthQA: 0.318 → **0.600**

## [0.12.0] - 2026-04-12

### Added — 3rd-generation retrieval + agent tool layer

- **3rd-gen retrieval pipeline** — relation-free graph, LLM-free indexing.
  `QueryAnchorExtractor` → `GraphExpander` → `HybridReranker` →
  `EvidenceAggregator` → `EvidenceSearch` facade.
- **Agent tool layer** — 7 atomic tools for multi-turn LLM exploration:
  `search`, `expand`, `get_document`, `list_categories`, `count`,
  `search_exact`, `follow`. Each returns structured `ToolResult` with
  `data`, `hints`, and `session` state.
- **SearchSession** — stateful context for multi-turn agent use. Tracks
  seen nodes, budget, query history, category coverage.
- **MCP server** — 8 new `agent_*` tools: `agent_search`, `agent_expand`,
  `agent_get_document`, `agent_list_categories`, `agent_count`,
  `agent_search_exact`, `agent_follow`, `agent_session_info`.
- **DomainProfile** — TOML-based domain configuration injection point.
  `to_dict()`, `save(path)` for round-trip serialization. New fields:
  `authority_by_kind`, `enrich_document_content`, `document_preview_chars`.
- **ProfileGenerator** — 3-tier auto profile generation (rule-based →
  OntologyClassifier → LLM). Detects locale, suggests stopwords,
  maps categories to NodeKind.
- **OntologyClassifier** — BYO embedder NodeKind classification via
  embedding cosine similarity. No torch dependency.
- **DocumentIngester** — generic JSONL → graph ingestion with
  `JsonlDocumentSource`, `InMemoryDocumentSource`, `CorpusSource` protocol.
  Document content enrichment (first chunks joined). Authority metadata.
  NFC normalization for categories/titles.
- **EntityLinker** — post-processing DF-filtered entity hub creation.
- **SqliteGraphBackend** — SQLite + `GraphTraversal` protocol
  (`shortest_path` BFS, `find_by_type_hierarchy`).
- **SQLiteBackend** improvements — NFC normalization on save, title 3x
  BM25 weight, LIKE substring fallback for Korean compound words.
- **Phrase extractors** — `KoreanPhraseExtractor`, `EnglishPhraseExtractor`,
  `create_phrase_extractor()` locale dispatcher.
- **node_metadata** helpers — `year_of()`, `authority_of()`, `is_current()`,
  `authority_ranked()`.
- **eval harness** — `ingest_krra`, `score_krra`, `score_krra_evidence`,
  `ingest_assort`, `score_assort`, `generate_profile` CLI scripts.
  KRRA + assort domain profiles and GT queries.
- **Multi-turn demo** — `examples/multi_turn_search.py` with Claude Sonnet
  validation (5/5 difficulty tiers passing).
- **683+ tests** (up from 504).

### Changed
- README rewritten for v0.12 — 3rd-gen retrieval positioning, agent tool
  quickstart, MCP server guide.

## [0.11.0] - 2026-04-09

### Added
- **KuzuBackend** — embedded property graph database using Kuzu 0.11.3.
  Native openCypher, FTS extension (Okapi BM25), and built-in graph traversal.
  Zero-config deployment (`pip install synaptic-memory[kuzu]` — no Docker, no server).
- `SynapticGraph.kuzu(db_path)` factory method for one-line setup.
- `tests/test_backend_kuzu.py` — 25 unit tests covering CRUD, search, traversal,
  batch ops, and maintenance. Runs in CI without external infrastructure.

### Removed — BREAKING
- **Neo4jBackend removed.** GPLv3 licensing on Neo4j Community, clustering limits,
  and operational overhead did not fit an MIT-licensed embedded library.
  Users still needing Neo4j can depend on the `neo4j` driver directly and
  implement the `StorageBackend` protocol themselves.
- `synaptic-memory[neo4j]` optional dependency removed.
- `tests/test_backend_neo4j.py` deleted.
- `docker-compose.yml` Neo4j service removed.
- `pytest.mark.neo4j` marker removed.

### Changed
- `CompositeBackend` now routes graph operations to Kuzu by default.
- `SynapticGraph.full(...)` and the scale preset reference Kuzu in docstrings.
- `pyproject.toml` — `scale` and `all` extras swap `neo4j>=5.25` for `kuzu>=0.11.0`.
- README Quick Start reorganized with Kuzu as the recommended embedded backend.
- Refactored README Quick Start to use factory functions.
- Refactored public API: factory functions, type stubs, reduced code duplication.

### Migration guide
- **Before:** `SynapticGraph(Neo4jBackend("bolt://localhost:7687", auth=("neo4j", "password")))`
- **After:** `SynapticGraph.kuzu("knowledge.kuzu")`

The Kuzu backend implements the same `StorageBackend` + `GraphTraversal`
protocols so Phase-level graph operations (PPR, Hebbian, consolidation)
work identically.

## [0.7.0] - 2026-03-22

### Added
- **Evidence Chain Assembly** — small LLM augmentation for multi-hop reasoning, HotPotQA Correctness 0.856 (+9.2%).
- **Personalized PageRank (PPR) engine** — replaced spreading activation, multi-hop retrieval +28%.
- **End-to-end QA benchmark** — HotPotQA 24-question suite for Cognee comparison (Correctness 0.784).
- **Auto-ontology optimization** — HybridClassifier, batch LLM processing, EmbeddingRelation, PhraseExtractor.

### Fixed
- PhraseExtractor search noise — phrase filtering and optimization.

### Changed
- Removed `__pycache__` from repo and updated `.gitignore`.

## [0.6.0] - 2026-03-21

### Added
- **Auto-ontology construction** — LLM-based ontology building with search-optimized metadata generation.
- **LLM classifier prompt optimization** — few-shot examples improved accuracy from 50% to 86%.
- **FTS + embedding hybrid scoring** — S7 Auto+Embed achieved MRR 0.83.
- **Kind/tag/search_keywords utilization** in search — FTS and ranking boost.
- **Ontology auto-construction + benchmark framework + search engine improvements** (combined release).

### Changed
- Updated README with auto-ontology, benchmark results, and differentiation points.

## [0.5.0] - 2026-03-21

### Added
- **Ontology Engine** — dynamic type hierarchy, property inheritance, relation constraint validation (`OntologyRegistry`).
- **Agent Activity Tracking** — session/tool call/decision/outcome capture (`ActivityTracker`).
- **Intent-based Agent Search** — 6 search strategies: similar_decisions, past_failures, related_rules, reasoning_chain, context_explore, general (`AgentSearch`).
- **Neo4j Backend** — native Cypher graph traversal, dual label, typed relationships, fulltext index.
- **Auto-embedding** — automatic vector generation on `add()` / `search()`.
- **Qdrant + MinIO + CompositeBackend** — storage separation by purpose.
- **5-axis Resonance Scoring** — added context axis (session tag Jaccard similarity).
- **GraphTraversal Protocol** — `shortest_path()`, `pattern_match()`, `find_by_type_hierarchy()`.
- **Node.properties** — ontology extension attributes, supported across all backends.
- **MCP 9 new tools** (total 16): agent session/action/decision/outcome tracking, ontology tools.
- 6 new `NodeKind` values: tool_call, observation, reasoning, outcome, session, type_def.
- 5 new `EdgeKind` values: is_a, invoked, resulted_in, part_of, followed_by.
- `docker-compose.yml` for Neo4j dev environment.
- `docs/COMPARISON.md` — comparison with existing agent memory systems.
- 185+ unit tests, 22 Neo4j integration tests.

### Fixed
- MemoryBackend fuzzy search ineffectiveness bug + 12 edge case QA tests added.
- Library distribution quality: `__version__`, `py.typed`, lazy imports, embedding extra.

## [0.4.0] - 2026-03-21

### Added
- **MCP Server** — 7 tools (knowledge search/add/link/reinforce/stats/export/consolidate).
- **SQLite Backend** — FTS5, recursive CTE, WAL mode.
- **QA Test Suite** — 169 Wikipedia + 368 GitHub real-data verification cases.
- `synaptic-mcp` CLI entry point.

## [0.3.0] - 2026-03-21

### Added
- **Protocol implementations** — LLM QueryRewriter, RegexTagExtractor, EmbeddingProvider.
- **LRU Cache** — NodeCache with hit rate tracking.
- **JSON Exporter** — structured JSON export.
- **Node Merge** — duplicate node merging with edge reconnection.
- **Find Duplicates** — title similarity-based duplicate detection.

## [0.2.0] - 2026-03-21

### Added
- **PostgreSQL backend** — asyncpg + pgvector HNSW + pg_trgm + recursive CTE.
- Vector search with cosine distance (pgvector).
- Trigram fuzzy matching with graceful ILIKE fallback.
- Hybrid search: FTS + fuzzy + vector merged results.
- Connection pooling (asyncpg Pool, min=2, max=10).
- Configurable `embedding_dim` parameter.
- `ResonanceWeights` added to public exports.
- Configurable consolidation thresholds (TTL, promotion access counts).
- README.md, ARCHITECTURE.md, ROADMAP.md documentation.
- GitHub Actions CI (Python 3.12/3.13).
- Integration test suite for PostgreSQL (13 tests).

### Changed
- Consolidation constants now accept `__init__` parameters instead of module globals.

## [0.1.0] - 2026-03-21

### Added
- Core models: Node, Edge, ActivatedNode, SearchResult, DigestResult.
- Enums: NodeKind (9), EdgeKind (7), ConsolidationLevel (4).
- Protocols: StorageBackend, Digester, QueryRewriter, TagExtractor.
- SynapticGraph facade: add, link, search, reinforce, consolidate, prune, decay.
- Hybrid 3-stage search: FTS + fuzzy, synonym expansion, query rewrite.
- Hebbian learning engine: co-activation reinforcement with anti-resonance.
- 4-axis resonance scoring: relevance x importance x recency x vitality.
- Memory consolidation cascade: L0 -> L1 -> L2 -> L3 with TTL and promotion.
- Korean/English synonym map (38 groups).
- Markdown exporter.
- MemoryBackend (dict-based, zero dependencies).
- SQLiteBackend (FTS5, recursive CTE, WAL mode).
- 93 unit tests, pyright strict, ruff clean.
