# Changelog

## v0.2.0 (2026-03-21)

### Added
- **PostgreSQL backend** — asyncpg + pgvector HNSW + pg_trgm + recursive CTE
- Vector search with cosine distance (pgvector)
- Trigram fuzzy matching with graceful ILIKE fallback
- Hybrid search method: FTS + fuzzy + vector merged results
- Connection pooling (asyncpg Pool, min=2, max=10)
- Configurable `embedding_dim` parameter
- `execute_raw()` for admin/testing SQL
- `ResonanceWeights` added to public exports
- Configurable consolidation thresholds (TTL, promotion access counts)
- Edge direction type safety: `Literal["both", "incoming", "outgoing"]`
- SQLite batch operations with rollback on error
- README.md, ARCHITECTURE.md, ROADMAP.md documentation
- GitHub Actions CI (Python 3.12/3.13)
- Integration test suite for PostgreSQL (13 tests)

### Changed
- Consolidation constants now accept `__init__` parameters instead of module globals

## v0.1.0 (2026-03-21)

### Added
- Core models: Node, Edge, ActivatedNode, SearchResult, DigestResult
- Enums: NodeKind (9), EdgeKind (7), ConsolidationLevel (4)
- Protocols: StorageBackend, Digester, QueryRewriter, TagExtractor
- SynapticGraph facade: add, link, search, reinforce, consolidate, prune, decay
- Hybrid 3-stage search: FTS + fuzzy → synonym expansion → query rewrite
- Hebbian learning engine: co-activation reinforcement with anti-resonance
- 4-axis resonance scoring: relevance × importance × recency × vitality
- Memory consolidation cascade: L0→L1→L2→L3 with TTL and promotion
- Korean/English synonym map (38 groups)
- Markdown exporter
- MemoryBackend (dict-based, zero deps)
- SQLiteBackend (FTS5, recursive CTE, WAL mode)
- 93 unit tests, pyright strict, ruff clean
