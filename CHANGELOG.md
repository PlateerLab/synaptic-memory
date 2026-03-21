# Changelog

## v0.5.0 (2026-03-21)

### Added
- **Ontology Engine** — 동적 타입 계층, 속성 상속, 관계 제약 검증 (`OntologyRegistry`, `TypeDef`, `PropertyDef`, `RelationConstraint`)
- **Agent Activity Tracking** — 세션/tool call/decision/outcome 캡처 (`ActivityTracker`)
- **Intent-based Agent Search** — 6가지 검색 전략: similar_decisions, past_failures, related_rules, reasoning_chain, context_explore, general (`AgentSearch`, `SearchIntent`)
- **Intent 자동 추론** — 쿼리 키워드에서 intent 자동 판별 (`suggest_intent()`, `intent="auto"` 기본값)
- **Neo4j Backend** — native Cypher 그래프 순회, dual label, typed relationship, fulltext index
- **GraphTraversal Protocol** — `shortest_path()`, `pattern_match()`, `find_by_type_hierarchy()`
- **5축 Resonance Scoring** — context axis 추가 (세션 태그 Jaccard 유사도)
- **Node.properties** — 온톨로지 확장 속성 (dict[str, str]), 전 백엔드 지원
- **Ontology 영속화** — `save_ontology()` / `load_ontology()`로 그래프에 저장/복원
- **L3 강등 메커니즘** — 성공률 60% 미만 시 L3 → L2 강등
- **Consolidation 페이지네이션** — limit=1000 제한 제거, 전체 노드 배치 처리
- **link() 온톨로지 검증** — 관계 제약 위반 시 ValueError 발생
- **Hebbian adaptive learning rate** — `delta / (1 + 0.02 × maturity)`로 초기 빠른 학습, 이후 안정화
- **HybridSearch node_kinds 필터** — 검색 시 노드 타입 필터링
- **기본 에이전트 온톨로지** — `build_agent_ontology()`로 knowledge/agent_activity 타입 트리 제공
- **MCP 9개 tool 추가** (총 16개): agent_start_session, agent_log_action, agent_record_decision, agent_record_outcome, agent_find_similar, agent_get_reasoning_chain, agent_explore_context, ontology_define_type, ontology_query_schema
- **NodeKind 6개 추가**: tool_call, observation, reasoning, outcome, session, type_def
- **EdgeKind 5개 추가**: is_a, invoked, resulted_in, part_of, followed_by
- `docker-compose.yml` — Neo4j 개발 환경
- `docs/COMPARISON.md` — 기존 Agent Memory 시스템 비교 분석
- 185+ unit tests, 22 Neo4j integration tests

### Changed
- `graph.agent_search()` 기본 intent가 `"auto"` (키워드 기반 자동 추론)
- `ResonanceWeights`에 `context` 필드 추가 (기본값 0.0, 하위호환)
- SQLite/PostgreSQL backend에 `properties_json` 컬럼 자동 마이그레이션 (v0.4 → v0.5)
- pyproject.toml: `neo4j`, `scale` extras 추가, version 0.5.0

## v0.4.0 (2026-03-21)

### Added
- **MCP Server** — 7개 tool (knowledge_search/add/link/reinforce/stats/export/consolidate)
- **SQLite Backend** — FTS5, recursive CTE, WAL mode
- **QA Test Suite** — Wikipedia 169건 + GitHub 368건 실제 데이터 검증
- `synaptic-mcp` CLI entry point

## v0.3.0 (2026-03-21)

### Added
- **Protocol 구현체** — LLM QueryRewriter, RegexTagExtractor, EmbeddingProvider
- **LRU Cache** — NodeCache with hit rate tracking
- **JSON Exporter** — 구조화된 JSON export
- **Node Merge** — 중복 노드 병합 + 엣지 재연결
- **Find Duplicates** — 제목 유사도 기반 중복 탐지

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
