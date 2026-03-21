# Synaptic Memory — Roadmap

## 현재 상태: v0.1.0 ✅

뇌 신경망 기반 knowledge graph 라이브러리 코어 완성.
93 tests, pyright strict 0 errors, zero core deps.

---

## v0.2.0 — PostgreSQL + Vector Search + 문서화

> 목표: 프로덕션 백엔드 + 벡터 검색 + 오픈소스 릴리즈 준비

### P1: 코드 품질 개선 (현재 코드 기반)

| # | Task | 파일 | 설명 |
|---|------|------|------|
| P1-1 | Edge direction 타입 강화 | protocols.py, backends/* | `direction: str` → `Literal["both", "incoming", "outgoing"]` |
| P1-2 | Consolidation 상수 설정 가능하게 | consolidation.py | TTL/threshold를 `__init__` 파라미터로 |
| P1-3 | Batch 작업 트랜잭션 | backends/sqlite.py | `save_nodes_batch()` 에 rollback 지원 |
| P1-4 | Vector search 경고 로그 | backends/sqlite.py | 빈 리스트 반환 시 warning 로그 |
| P1-5 | `__init__.py` export 보강 | __init__.py | `ResonanceWeights`, `MemoryBackend`, `SQLiteBackend` 추가 |

### P2: PostgreSQL 백엔드

| # | Task | 파일 | 설명 |
|---|------|------|------|
| P2-1 | PostgreSQL 스키마 SQL | backends/postgresql.sql | AGE 그래프 + 관계형 테이블 + 인덱스 |
| P2-2 | PostgreSQLBackend 구현 | backends/postgresql.py | asyncpg 기반, StorageBackend 전체 구현 |
| P2-3 | 듀얼 스토리지 동기화 | backends/postgresql.py | AGE 그래프 ↔ 관계형 테이블 일관성 유지 |
| P2-4 | Cypher 기반 Spreading Activation | backends/postgresql.py | AGE 2-hop 순회 쿼리 |
| P2-5 | pgvector HNSW 벡터 검색 | backends/postgresql.py | `search_vector()` — cosine distance |
| P2-6 | pg_trgm 한글 퍼지 매칭 | backends/postgresql.py | `search_fuzzy()` — similarity() |
| P2-7 | 하이브리드 단일 쿼리 | backends/postgresql.py | FTS + fuzzy + vector를 1개 SQL로 결합 |
| P2-8 | 마이그레이션 스크립트 | backends/migrate.py | SQLite → PostgreSQL 데이터 이관 |
| P2-9 | Integration 테스트 | tests/test_backend_postgresql.py | `@pytest.mark.integration` (실제 PG) |

### P3: 문서화 + PyPI

| # | Task | 파일 | 설명 |
|---|------|------|------|
| P3-1 | README.md | README.md | Quick start, 아키텍처 다이어그램, 설치 가이드 |
| P3-2 | ARCHITECTURE.md | docs/ARCHITECTURE.md | 시스템 설계, 데이터 흐름, 설계 결정 |
| P3-3 | API Reference | docs/API.md | 전체 public API 문서 |
| P3-4 | 사용 예제 | docs/EXAMPLES.md | hive-corp 통합 사례 + 독립 사용법 |
| P3-5 | CHANGELOG | CHANGELOG.md | v0.1.0 → v0.2.0 변경 이력 |
| P3-6 | GitHub Actions CI | .github/workflows/ci.yml | ruff + pyright + pytest |
| P3-7 | PyPI 배포 설정 | pyproject.toml | classifiers, URLs, entry points |
| P3-8 | PyPI 배포 | - | `synaptic-memory==0.2.0` |

---

## v0.3.0 — 프로토콜 구현체 + 고급 기능

> 목표: LLM 연동 프로토콜 구현 + 성능 최적화

### P4: 프로토콜 구현체

| # | Task | 파일 | 설명 |
|---|------|------|------|
| P4-1 | LLM QueryRewriter | extensions/rewriter.py | Haiku 기반 쿼리 재작성 (optional dep) |
| P4-2 | LLM TagExtractor | extensions/tagger.py | Haiku 기반 태그 추출 |
| P4-3 | LLM Digester | extensions/digester.py | 문서 → 노드 변환 (요약 + 엔티티 추출) |
| P4-4 | Regex TagExtractor | extensions/tagger_regex.py | 정규식 기반 태그 추출 (zero-LLM) |
| P4-5 | Embedding Provider | extensions/embedder.py | OpenAI/로컬 모델 임베딩 생성기 |

### P5: 성능 + 모니터링

| # | Task | 파일 | 설명 |
|---|------|------|------|
| P5-1 | 벤치마크 스위트 | benchmarks/ | search latency, insert throughput, memory footprint |
| P5-2 | 쿼리 플래너 로깅 | search.py | 각 stage 소요 시간 + 후보 수 기록 |
| P5-3 | 메트릭 export | metrics.py | Prometheus/OpenTelemetry 연동 |
| P5-4 | Connection pooling | backends/postgresql.py | asyncpg pool 관리 |
| P5-5 | 캐시 레이어 | cache.py | 자주 조회되는 노드 LRU 캐시 (maxsize 제한) |

### P6: 고급 그래프 기능

| # | Task | 파일 | 설명 |
|---|------|------|------|
| P6-1 | 서브그래프 추출 | graph.py | 특정 노드 중심 N-hop 서브그래프 반환 |
| P6-2 | 그래프 시각화 | exporter.py | Mermaid/Graphviz DOT 형식 export |
| P6-3 | JSON export | exporter.py | nodes + edges JSON 형식 |
| P6-4 | 노드 병합 | graph.py | 중복 노드 탐지 + 자동 병합 |
| P6-5 | 동의어 자동 학습 | synonyms.py | 검색 로그에서 co-occurrence 기반 동의어 발견 |

---

## v0.4.0 — 분산 + 멀티테넌트

> 목표: 다수 소비자(hive, openclaw, gwanjong-mcp) 동시 지원

### P7: 멀티테넌트

| # | Task | 설명 |
|---|------|------|
| P7-1 | Namespace 지원 | 테넌트별 격리 (schema prefix 또는 별도 DB) |
| P7-2 | RBAC | 노드/엣지 접근 권한 제어 |
| P7-3 | 감사 로그 | 누가 언제 무엇을 변경했는지 기록 |

### P8: MCP 서버

| # | Task | 설명 |
|---|------|------|
| P8-1 | MCP Protocol | synaptic-memory를 MCP 서버로 노출 |
| P8-2 | Tool 정의 | knowledge_search, knowledge_add, knowledge_link |
| P8-3 | graph-tool-call 통합 | MCP Proxy 경유 접근 |

---

## 구현 순서 (권장)

```
v0.2.0 Phase 1: P1 (코드 품질) → P3-1 (README) → P3-6 (CI)
v0.2.0 Phase 2: P2 (PostgreSQL 백엔드)
v0.2.0 Phase 3: P3 (나머지 문서 + PyPI 배포)
v0.3.0 Phase 1: P4 (프로토콜 구현체)
v0.3.0 Phase 2: P5 (성능) + P6 (고급 기능)
v0.4.0: P7 (멀티테넌트) + P8 (MCP)
```

---

## 설계 원칙 (전 버전 공통)

1. **Zero core deps** — 코어 로직은 순수 Python, 백엔드/확장만 extras
2. **Protocol 기반** — 인터페이스 교체 가능, 테스트 용이
3. **Async-first** — 모든 I/O는 async/await
4. **메모리 안전** — 캐시 크기 제한, context manager 강제, 무한 증가 방지
5. **한/영 이중 언어** — 동의어 맵, 토크나이저, 퍼지 매칭 모두 한국어 지원
6. **Token 효율** — LLM 호출은 최소화, 가능한 것은 코드로 해결
