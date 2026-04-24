# Synaptic Memory — Roadmap

> 마지막 업데이트: 2026-04-20 (v0.17.2 PyPI + v0.18-alpha 진행 중)

## 현재 상태

- **PyPI stable**: [`synaptic-memory==0.17.2`](https://pypi.org/project/synaptic-memory/0.17.2/)
- **진행 트랙**: v0.18-alpha (`graph.chat()` agent-loop public API)
- **라이선스**: Apache-2.0 (v0.17.2 에서 MIT → Apache-2.0 전환)
- **테스트**: **942** 단위 테스트 통과
- **MCP 도구**: 36개
- **코어 의존성**: 0 (torch-free, BYO embedder/reranker)

### 측정 베이스라인 (v0.17.1, 2026-04-19)

| 모드 | 범위 | 평균 MRR / 성공률 |
|---|---|---:|
| 22-벤치 FTS-only | Korean enterprise 16 + English diverse 6 | **0.650** |
| 14-벤치 Full pipeline | `bge-m3` + `bge-reranker-v2-m3` | **0.647** (FTS-only 0.615 → 첫 net positive) |
| 6-벤치 Agent | Qwen3.5-27B vLLM, 5턴 | **140/172 = 81.4%** |

알려진 한계: MuSiQue-Ans 500q R@5 **0.453** vs HippoRAG2 publish 0.747 (−0.294).
v0.18 아키텍처 교체 트랙.

---

## 최근 ship 이력

| Release | 날짜 | 핵심 |
|---|---|---|
| v0.17.2 | 2026-04-20 | License 전환 (MIT → Apache-2.0), `agent_loop` ID-extraction fix, ruff clean |
| v0.18-alpha | 2026-04-19 | `graph.chat()` agent-loop public API, auto-corpus calibration (measured negative) |
| v0.17.1 | 2026-04-19 | Kind-aware aggregator + reranker skip on `_table_name`, adaptive cross-encoder blend, `DomainProfile.table_query_hints`, `LLMReranker` / `HyDEEmbedder` opt-in modules, lenient profile loader |
| v0.17.0 | 2026-04-17 | `rerank_blend` 0.4→0.1, `QueryDecomposer` Protocol + `LLMChainDecomposer` (opt-in, measured negative in MuSiQue), `--local-bge` flag |
| v0.16.0 | 2026-04-16 | `graph.search()` default engine flip (`legacy`→`evidence`), CDC batching, 30× eval coverage |
| v0.15.0 | 2026-04-15 | `graph.backfill()`, `engine="evidence"` opt-in + deprecation timeline |
| v0.14.0 | 2026-04-14 | Live DB CDC (7-phase), MCP ingest tools, X2BEE 프로덕션 검증 |

자세한 변경은 [CHANGELOG.md](../CHANGELOG.md). 알고리즘 설계 결정과
measured negatives 는 [PLAN-v0.17-ontology.md](PLAN-v0.17-ontology.md) ·
[PLAN-v0.18-architecture.md](PLAN-v0.18-architecture.md) · [CONCEPTS.md §13](CONCEPTS.md).

---

## v0.18 트랙 (진행 중)

v0.17.1 이 제기한 5개 open question 해결 + agent-loop 공개화.
상세 설계: [PLAN-v0.18-architecture.md](PLAN-v0.18-architecture.md).

### α1. `graph.chat()` 안정화 (진행 중)

v0.18-alpha 에서 공개. 현재 172q 평균 81.4%, 6개 벤치 중 5개에서 v0.13
GPT-4o-mini baseline 초과.

| # | 작업 | 상태 |
|---|------|---|
| α1-1 | `graph.chat()` public API + session 관리 | ✅ ship (v0.18-alpha, commit `4fa7df7`) |
| α1-2 | KRRA Conv **−23pp 회귀** 원인 조사 — Qwen 한국어 conversational 약점 또는 prompt 이슈 | ✅ 진단 완료 — FTS recall ceiling, query-time LLM rewrite (HyDE) 없이는 회복 불가 → v0.19+ |
| α1-3 | Agent-loop latency 감축 — 첫 1-2 turn priming 으로 탐색 turn 절약 | ✅ α2 snapshot priming 으로 ship |
| α1-4 | Context overflow 회피 (현재 172q 중 10q = 5.8% vLLM 16k 초과로 fail) | ✅ ship — `project_tool_result` (commit TBD) |

### α2. G1 — Auto graph snapshot / agent priming ✅ ship

**문제**: Agent 가 cold start 시 corpus 구조 모르고 시작 → 첫 1-2 turn 이
탐색에 낭비. Graphify (`safishamsi/graphify`) 의 UX 패턴 흡수.

| # | 작업 | 상태 |
|---|------|---|
| α2-1 | `synaptic-snapshot <db> --output graph.md` CLI | ✅ ship — `synaptic.cli.snapshot` |
| α2-2 | 출력 내용: 카테고리 트리, top phrase hub (DF), entity-table 분포, edge 통계, sample queries | ✅ ship — `synaptic.snapshot` |
| α2-3 | `knowledge_snapshot()` MCP 도구 — agent 시작 시 1회, system prompt inject | ✅ ship — `mcp/server.py` |
| α2-4 | `graph.chat()` 기본 경로에 통합 — `prime_with_snapshot=True` (default) | ✅ ship |

11 unit tests / 0.85 s on KRRA (720 docs / 18.6k chunks / 70k entities).
G2-G5 는 [v0.18 architecture doc](PLAN-v0.18-architecture.md) 에서 격하됨.

### α3. OpenIE triple 실험 (MuSiQue 한계 회복)

현재 relation-free 구조는 영어 shortcut-heavy multi-hop 에 구조적으로 약함
(MuSiQue R@5 0.453 vs HippoRAG2 0.747). 3-round ablation
(decomposer / inline phrase / DF-filtered entity linker) 모두 실패.

| # | 작업 | 비고 |
|---|------|------|
| α3-1 | Typed relation extraction opt-in CLI sweep (`scripts/extract_typed_relations.py`) | LLM-free 인덱싱 원칙 유지 — opt-in 만 |
| α3-2 | EdgeKind 확장 (`WORKS_FOR`, `LOCATED_IN`, `SUBSIDIARY_OF` 등) + PPR `_EDGE_TYPE_WEIGHTS` tuning | |
| α3-3 | HippoRAG 2 스타일 query→triple dense linking 프로토타입 | 핵심 mechanism — MuSiQue +12.5pp 의 진짜 원인 |
| α3-4 | MuSiQue 500q ablation — R@5 ≥ 0.55 목표 | decomposer 는 이미 `LLMChainDecomposer` 로 측정 후 negative 확정 |

---

## Carry-over (v0.16~v0.17 에서 미완)

### C1. CDC schema drift 감지

v0.14.0 CDC 에서 `syn_cdc_state.schema_fingerprint` 를 **저장하지만 비교하지
않음**. `ALTER TABLE` 이 소스 DB 에 일어나도 sync 가 그대로 진행되고 결과는
silent 하게 틀어짐. P1 gap.

| # | 작업 |
|---|------|
| C1-1 | `TimestampTableSyncer` / `HashTableSyncer` 시작 시 `prior_state.schema_fingerprint` vs fresh 비교 |
| C1-2 | 변경 감지 시 해당 테이블만 force full reload + state 초기화 |
| C1-3 | `SyncResult.tables[i].schema_changed` 플래그 신설 |
| C1-4 | 회귀 테스트 — 컬럼 추가/제거 시나리오 |

### C2. PostgreSQL 백엔드 feature parity

`PostgreSQLBackend` 가 SQLite 와 비교해 누락된 기능 점검 필요. 특히
HNSW 사이드카 persist, CDC 테이블 (`syn_cdc_state`, `syn_cdc_pk_index`),
`ensure_cdc_tables()` 메서드.

| # | 작업 |
|---|------|
| C2-1 | SQLite vs PostgreSQL 기능 매트릭스 작성 |
| C2-2 | 누락된 기능 채우기 — wrapper 작업이 대부분 |
| C2-3 | `test_backend_postgresql.py` 회귀 테스트 확장 |

### C3. Self-calibrating cosine probe

v0.14.1 의 relative threshold 도 결국 default 값 (`min_cosine=0.10`,
`relative_drop=0.30`) 을 가짐. 임베더 분포가 극단적이면 사용자가 튜닝
필요. v0.18-alpha 에서 auto-corpus calibration 1차 시도했지만 measured
negative. 재설계 필요.

| # | 작업 |
|---|------|
| C3-1 | Corpus 에서 N 개 (title, content) 쌍 sample → cosine 분포 측정 |
| C3-2 | 측정값으로 `min_cosine = p10`, `relative_drop = (p50-p10)/p50` 자동 계산 |
| C3-3 | Cache + fingerprint (embedder 모델 기반 invalidation) |
| C3-4 | `synaptic-cli calibrate --db ... --embed-url ...` 명령 |

### C4. Oracle / MSSQL CDC 지원

v0.14.0 Phase 6 follow-up. Dialect 별 placeholder 차이 (`:1` vs `?`) 만
해결하면 나머지는 기존 orchestrator 재사용 가능.

| # | 작업 |
|---|------|
| C4-1 | `_read_oracle_rows(..., where_clause=, where_params=)` + `_read_oracle_pks()` |
| C4-2 | `_read_mssql_rows(..., where_clause=, where_params=)` + `_read_mssql_pks()` |
| C4-3 | `_translate_placeholders()` 확장 — Oracle `:1`, MSSQL `?` |
| C4-4 | `sync_from_oracle()` / `sync_from_mssql()` 오케스트레이터 |
| C4-5 | Env-var opt-in integration 테스트 |

### C5. Legacy HybridSearch 제거

v0.17.0 원래 목표였으나 carry-over. `engine="legacy"` 는 여전히 escape
hatch 로 유지 중 (`src/synaptic/search.py` + `agent_search.py` 에 잔존).

| # | 작업 |
|---|------|
| C5-1 | `HybridSearch` 클래스 + `search.py` 제거 |
| C5-2 | `graph.search(engine=)` 파라미터 제거 |
| C5-3 | `agent_search.py` 의 HybridSearch 기반 intent routing 정리 |
| C5-4 | Migration 가이드 `docs/MIGRATION-0.19.md` |

---

## v0.19+ 장기 (우선순위 미확정)

사용자 요청 / 실사용 데이터를 본 후 결정.

### 평가 인프라

- **LLM-as-Judge 벤치 모드** — `eval/run_all.py --judge-llm` 옵션. 현재
  ID 매칭 GT 는 집계/패러프레이즈 쿼리에 부정확.
- **CI 벤치 회귀 가드** — GitHub Action 에 `--quick` bench 정기 실행.
  `qa_latest.json` vs 비교해서 5% 이상 회귀면 CI fail. Stale baseline
  함정 영구 봉인.
- **Synthetic query generation** — 새 corpus 추가 시 자동 query 보강.

### 검색 품질

- **Doc2Query++** — 인제스트 시 문서당 예상 쿼리 5개 LLM 생성 →
  properties 에 저장 → FTS 인덱싱. PublicHealthQA 같은 도메인 특화
  recall 개선.
- **Multi-vector / late interaction** — ColBERT-style.
  `reranker_colbert.py` 골격은 있지만 default 아님.
- **Cross-encoder default fallback** — 현재 BYO(TEI) 만 지원. Default
  경량 모델 1개 ship.
- **Multimodal converter pack** — G4 격하분. Whisper / pdfplumber /
  vision LLM. 사용자 요청 받으면 재평가.

### 운영 인프라

- **Observability** — 메트릭 export (Prometheus/OpenTelemetry), health
  check endpoint, connection pooling.
- **Streaming MCP responses** — 긴 ingest/sync 작업에 progress 스트림.
- **Cost tracking** — 임베딩/LLM API 호출 누적 집계.
- **Quantized embeddings** (int8/binary) — 100k+ corpus 메모리 절감.
- **Multi-embedder routing** — 노드 종류별 다른 임베더.
- **Edge confidence + provenance** — G2 격하분. Neo4j 등 표준이지만
  Synaptic 용도로는 자체 설계 필요.

### 확장성

- **A/B testing 인프라** — 두 검색 설정을 같은 query set 으로 비교.
- **Continuous learning** — 사용자 피드백(reinforce) 기반 weight
  auto-tuning. 설계는 있으나 장기 효과 미입증.
- **Multi-tenant** — 테넌트별 격리, RBAC, 감사 로그.

---

## 알고리즘 영감의 출처 정리

v0.18 트랙의 alg-level 영감은 **paper 들이 원전**. v0.17.x 에서 Graphify
같은 productization 프로젝트를 비교하며 흡수 항목 5개 (G1-G5) 검토했고,
**algorithm 측면 신규 가치는 G1 (auto agent priming) 1가지만** 확정.
나머지는 GraphRAG / Neo4j / standard converter 의 차용.

### Algorithm-level inspiration (paper 직접 reference)

- **HippoRAG 2** (arXiv:2502.14802) — query→triple linking,
  MuSiQue R@5 +12.5pp 의 진짜 mechanism
- **GraphRAG** (Microsoft, arXiv:2404.16130) — community detection +
  summary 의 원전
- **PropRAG** (arXiv:2504.18070) — proposition graphs vs triples
- **Adaptive RAG / Self-RAG / CRAG** (2024) — agent-internal critique
  mechanism
- **HyDE** (Gao 2022) — query→hypothetical answer (우리 시도 후 KRRA
  도메인에서 실패 측정 완료)
- **Late Chunking** (Jina 2024) — embedding-then-chunk
- **LightRAG** (arXiv:2410.05779) — dual-level retrieval

이들이 v0.18 main track 의 진짜 영감원. 자세한 분석:
[PLAN-v0.18-architecture.md](PLAN-v0.18-architecture.md).

### G2-G5 격하 (v0.18 main track 외부로 이동)

| ID | 항목 | 격하 이유 | 재배치 |
|---|---|---|---|
| ~~G2~~ | Edge confidence + provenance | Neo4j 표준. Graphify 의 `EXTRACTED/INFERRED/AMBIGUOUS` 라벨은 임의적. 알고리즘 신규 X | 운영 인프라 (v0.19+) |
| ~~G3~~ | Leiden community detection | **Microsoft GraphRAG (2024) 가 원전**, Graphify 는 차용. | Algorithm-level inspiration 으로 흡수 |
| ~~G4~~ | Multimodal converter pack | Whisper / pdfplumber / vision LLM 모두 standard. 알고리즘 0, 단순 packaging. | 검색 품질 (v0.19+) |
| ~~G5~~ | Hyperedges | 학술적 흥미, 실용 미미. niche 사용 사례. | Skip. 향후 사용자 요청 시 재검토 |

### 의도적 비흡수 (Synaptic 차별성 보존)

- **NetworkX + JSON storage** — production scale 불가, multi-backend 유지
- **Indexing-time LLM 호출** — LLM-free 원칙 위반, 핵심 차별
- **Tree-sitter AST 25-lang 코드 파싱** — 1차 도메인 (한국어 enterprise) 무관

---

## Measured negatives (ship 안 함)

v0.17 개발 중 측정 후 **품질 악화로 확정된 4개 접근**. 상세:
[CONCEPTS.md §13](CONCEPTS.md).

| 접근 | 측정 | 현재 상태 |
|---|---|---|
| LLM query decomposer | MuSiQue R@5 −10.6%, search 4× 느림 | `LLMChainDecomposer` 코드 잔존, **opt-in default-off** |
| Inline phrase hub (DF filter 없이) | MuSiQue R@5 −6.6%, build 15× 느림 | Inline 경로 유지하되 default 는 post-hoc DF-filtered `EntityLinker` |
| DF-filtered EntityLinker | 공개 5벤치 평균 ±1% (neutral) | `--entity-linker` flag, **default-off** |
| `rerank_blend=0.4` | AutoRAG −29%, retrieval-style corpus 에서 파괴적 | v0.17.0 에서 default 0.1 로 변경 |
| Auto-corpus calibration | measured negative | v0.18-alpha 에서 1차 시도 후 재설계 필요 (C3) |
| LLMReranker / HyDEEmbedder | measured negative | v0.17.1 에 opt-in module 만 유지 |

**교훈**: "Mechanism 추가 = 품질 개선" 은 corpus 유형에 따라 부합하지 않는다.
항상 current-code FTS-only 를 재측정한 뒤 비교할 것.

---

## 완료된 작업 (historical)

### v0.1.0 ~ v0.5.0

뇌 신경망 기반 코어 라이브러리 + PostgreSQL 백엔드 + 문서화 + PyPI 배포.
전체 내용은 git history 참조.

### v0.6.0 ~ v0.12.0

3세대 retrieval 파이프라인, HippoRAG2 phrase hub, DomainProfile, MCP 서버
(29개 도구), 멀티턴 에이전트, 구조적 쿼리 도구 (`filter_nodes` /
`aggregate_nodes` / `join_related`), KRRA / assort / X2BEE 벤치마크
데이터셋.

### v0.13.0 — 2026-04-13

Graph-aware agent search + structured data tools. `from_database()`
one-liner. 29 MCP tools. 687 tests. v0.14.x 시리즈의 출발 베이스.

### v0.14.0 → v0.15.0 — 2026-04-14 / 2026-04-15

- Live database CDC (7-phase 구현, X2BEE 프로덕션 검증)
- MCP ingest / CDC tools (29 → 36 tools)
- Embedder-agnostic vector threshold (magic number 제거)
- `knowledge_search` → EvidenceSearch
- Phrase hub wiring fix (cross-document bridges)
- `graph.backfill()` — 기존 그래프 복구 도구
- `graph.search(engine="evidence")` opt-in + deprecation timeline

### v0.16.0 — 2026-04-16

- `graph.search()` default engine flip (`legacy`→`evidence`)
- CDC batching 개선
- 30× eval coverage 확장

### v0.17.0 → v0.17.1 — 2026-04-17 ~ 2026-04-19

- `rerank_blend` 0.4→0.1 default 변경 (AutoRAG 회귀 해소)
- `QueryDecomposer` Protocol + `LLMChainDecomposer` (opt-in, measured
  negative in MuSiQue)
- `--local-bge` flag — FTS-only vs Full pipeline 이중 측정 표준화
- Kind-aware aggregator + reranker skip on `_table_name` rows
- Adaptive cross-encoder blend (`std/3` discriminator)
- `DomainProfile.table_query_hints`
- `LLMReranker` / `HyDEEmbedder` opt-in modules (둘 다 measured negative)
- Lenient profile loader
- **22-벤치 baseline** 확장 (Korean 16 + English 6)

### v0.18-alpha — 2026-04-19

- `graph.chat()` agent-loop public API
- Auto-corpus calibration 1차 시도 (measured negative, 재설계 예정)

### v0.17.2 — 2026-04-20

- License 전환: MIT → **Apache-2.0**
- `agent_loop` ID-extraction fix
- Ruff check + format clean (CI lint 통과)

자세한 변경은 [CHANGELOG.md](../CHANGELOG.md).

---

## 설계 원칙 (전 버전 공통)

1. **Zero core deps** — 코어 로직은 순수 Python, 백엔드/확장만 extras
2. **Protocol 기반** — 인터페이스 교체 가능, 테스트 용이
3. **Async-first** — 모든 I/O 는 async/await
4. **Memory safe** — 캐시 크기 제한, context manager 강제
5. **한/영 이중 언어** — 동의어 맵, 토크나이저, 퍼지 매칭 모두 한국어 지원
6. **LLM-free indexing** — 인덱스 시점 LLM 호출 0. 벡터 임베딩만 API 호출.
7. **BYO embedder/reranker** — torch-free. 사용자가 Ollama / TEI / API 직접 선택.
8. **Silent failure 는 버그** — v0.14.x 시리즈의 핵심 교훈. 기능이 wire
   안 되면 명확한 에러나 warning 이 나야 함. 조용히 기능이 죽어 있는
   건 안 됨.
9. **Measured negative 도 문서화한다** — v0.17 의 교훈. 시도 후 실패한
   접근은 CONCEPTS §13 / ROADMAP Measured negatives 에 이유와 측정치
   함께 기록. 다음 세션/기여자가 같은 실수를 반복하지 않도록.
