# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Refactored README Quick Start to use factory functions.
- Refactored public API: factory functions, type stubs, reduced code duplication.

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
