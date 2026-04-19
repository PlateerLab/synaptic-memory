# Synaptic Memory — Roadmap

> 마지막 업데이트: 2026-04-19 (v0.17.1 PyPI 배포 + v0.18 reform 트랙 시작 직전)

## 현재 상태: v0.17.1 ✅ (PyPI 배포 완료 2026-04-19)

- **PyPI**: [`synaptic-memory==0.17.1`](https://pypi.org/project/synaptic-memory/0.17.1/)
- **테스트**: 820 단위 테스트 통과
- **MCP 도구**: 36개
- **코어 의존성**: 0 (torch-free, BYO embedder/reranker)
- **22-벤치 baseline (FTS-only, 2026-04-19)**: 평균 MRR **0.650** (Korean enterprise 16 + English diverse 6 domains)
- **14-벤치 Full pipeline (bge-m3 + bge-reranker-v2-m3)**: 평균 MRR **0.647** (FTS-only 0.615 → 첫 번째 net positive)
- **6-벤치 Agent (Qwen3.5-27B vLLM)**: 평균 solved **81%**

**v0.16.0 → v0.17.1 시리즈에서 ship한 것**:

| Release | 핵심 |
|---|---|
| v0.16.0 | `graph.search()` default engine flip (`evidence`), CDC batching, 30× eval coverage |
| v0.17.0 | `rerank_blend` 0.4→0.1, `QueryDecomposer` Protocol + `LLMChainDecomposer`, `--local-bge` flag |
| **v0.17.1** | Kind-aware aggregator + reranker skip, adaptive cross-encoder blend, `DomainProfile.table_query_hints`, `LLMReranker` / `HyDEEmbedder` opt-in modules, lenient profile loader |

자세한 내용은 [CHANGELOG.md](../CHANGELOG.md). 알고리즘 설계 결정과 measured negatives는 [PLAN-v0.17-ontology.md](PLAN-v0.17-ontology.md), [PLAN-v0.18-architecture.md](PLAN-v0.18-architecture.md), [CONCEPTS.md §13](CONCEPTS.md).

---

## v0.16.0 — Default engine flip + follow-up cleanup (다음 minor)

> 목표: v0.15.0에서 opt-in으로 도입한 `engine="evidence"`를 기본값으로 전환하고, v0.14.x 시리즈에서 미뤘던 후속 작업 정리.

### P1: `graph.search()` default engine 전환

| # | 작업 | 비고 |
|---|------|------|
| P1-1 | `graph.search(engine=)` default `"legacy"` → `"evidence"` 전환 | v0.15.0에서 deprecation timeline 예고됨 |
| P1-2 | Release notes에 breaking change 상세 명시 | `stages_used == "synonym"` / `"rewriter"` branch 있는 외부 코드는 break |
| P1-3 | `test_search.py` / `test_graph.py` 갱신 — 새 default에 맞는 expectation | 47+ 테스트가 legacy 경로 동작에 의존 |
| P1-4 | Bench 회귀 검증 | KRRA / X2BEE / assort / 공개 데이터셋 전부 재측정 후 CHANGELOG에 diff 공개 |
| P1-5 | `engine="legacy"` escape hatch는 유지 | v0.17.0에서 제거 예정 |

### P2: 임베더/리랭커 모드 베이스라인 재측정

v0.14.x의 검색 경로 변경(HybridSearch threshold + knowledge_search 마이그레이션) 이후 embedder + reranker 모드 점수가 한 번도 측정 안 됨. [CLAUDE.md](../CLAUDE.md)의 "현재 베이스라인" 섹션에 FTS-only 점수만 있는 상태.

| # | 작업 |
|---|------|
| P2-1 | Home 서버(Ollama qwen3-embedding:4b + TEI bge-reranker-v2-m3) 기동 |
| P2-2 | `eval/run_all.py --embed-url ... --reranker-url ...` 풀 모드 측정 |
| P2-3 | `eval/baselines/qa_latest.json._meta` 에 `mode: "fts-only" | "full"` 분리해서 두 버전 유지 |
| P2-4 | Agent 벤치마크(5턴 GPT-4o-mini)도 재측정 — v0.13.0 때 값이 아직 CLAUDE.md에 남아 있음 |
| P2-5 | CLAUDE.md 베이스라인 표 재작성 — FTS-only / embedder / agent 3가지 모드 구분 |

### P3: CDC schema drift 감지

v0.14.0 CDC에서 `syn_cdc_state.schema_fingerprint`를 저장하지만 **비교는 안 함**. `ALTER TABLE`이 소스 DB에 일어나도 sync가 그대로 진행되고 결과는 silent하게 틀어짐. 현재 알려진 P1 gap.

| # | 작업 |
|---|------|
| P3-1 | `TimestampTableSyncer` / `HashTableSyncer` 시작 시 `prior_state.schema_fingerprint` vs fresh 비교 |
| P3-2 | 변경 감지 시 해당 테이블만 force full reload + state 초기화 |
| P3-3 | `SyncResult.tables[i].schema_changed` 플래그 신설 |
| P3-4 | 회귀 테스트 — 컬럼 추가/제거 시나리오 |

### P4: PostgreSQL 백엔드 feature parity 점검

`PostgreSQLBackend`가 SQLite와 비교해 어떤 기능이 누락됐는지 명확하지 않음. 특히:
- HNSW 사이드카 persist (v0.14.0 SQLite에 추가)
- CDC 테이블 (`syn_cdc_state`, `syn_cdc_pk_index`)
- `ensure_cdc_tables()` 메서드

| # | 작업 |
|---|------|
| P4-1 | SQLite vs PostgreSQL 기능 매트릭스 작성 |
| P4-2 | 누락된 기능 채우기 — wrapper 작업이 대부분 |
| P4-3 | `test_backend_postgresql.py` 회귀 테스트 확장 |

---

## v0.17.0 — Legacy 제거 + 근본 개선

> 목표: v0.15 deprecation timeline 완료 + 구조적 개선.

> **온톨로지 트랙 — Case B 확정 (2026-04-18)**: 사용자 요청 "온톨로지 고도화"는
> [PLAN-v0.17-ontology.md](PLAN-v0.17-ontology.md) 에서 별도 평가.
> MuSiQue 500q (bge-m3 + bge-reranker-v2-m3 ON) 재측정 결과 R@5 **0.453**
> (< 0.5 threshold, HippoRAG2 0.747 대비 -0.294) → 임베더 강화만으론 구조적 상한
> 확인. **P8 (Query decomposer 통합)** 추가. Typed relation 은 opt-in CLI sweep
> 으로만 제공 (LLM-free 인덱싱 원칙 유지). 기존 P5/P6/P7은 그대로 진행.

### P8: Query decomposer 통합 (v0.17.0 온톨로지 트랙)

PLAN-v0.17-ontology §6 작업 분해(W-1 ~ W-9) 요약:

| # | 작업 |
|---|------|
| P8-1 | `QueryDecomposer` Protocol 정의 (`protocols.py`) |
| P8-2 | EvidenceSearch 에 decomposer 분기 — 서브쿼리 병렬 seed → RRF(k=60) 통합 → rerank |
| P8-3 | Rule-based decomposer 리팩터 (`query_decomposer.py` 184줄 prototype → Protocol 구현) + 테스트 보강 |
| P8-4 | LLM decomposer 구현체 (BYO, opt-in — `query_decomposer_llm.py`) |
| P8-5 | `SynapticGraph(decomposer=None)` 파라미터 + `agent_tools_v2.deep_search` 위임 |
| P8-6 | MuSiQue / 2Wiki 500q decomposer ON/OFF ablation — 성공 기준 R@5 ≥ 0.55 |
| P8-7 | `scripts/extract_typed_relations.py` opt-in CLI sweep (기존 `relation_detector_llm.py` 활용) |
| P8-8 | EdgeKind 확장 (`WORKS_FOR`, `LOCATED_IN`, `SUBSIDIARY_OF` 등) + PPR `_EDGE_TYPE_WEIGHTS` |

### P5: Legacy HybridSearch 제거

| # | 작업 |
|---|------|
| P5-1 | `HybridSearch` 클래스 + `search.py` 제거 |
| P5-2 | `graph.search(engine=)` 파라미터 제거 (이제 항상 EvidenceSearch) |
| P5-3 | `agent_search.py`의 HybridSearch 기반 intent routing도 정리 |
| P5-4 | Migration 가이드 docs/MIGRATION-0.17.md 추가 |

### P6: Self-calibrating cosine probe

v0.14.1의 relative threshold도 결국 default 값 (`min_cosine=0.10`, `relative_drop=0.30`)을 가짐. 임베더 분포가 극단적이면 사용자가 튜닝해야 함. Self-calibration으로 해결:

| # | 작업 |
|---|------|
| P6-1 | `cdc/calibration.py` 신규 — corpus에서 N개 (title, content) 쌍 sample → cosine 분포 측정 |
| P6-2 | 측정값으로 `min_cosine = p10`, `relative_drop = (p50-p10)/p50` 자동 계산 |
| P6-3 | Cache + fingerprint (embedder 모델 기반 invalidation) |
| P6-4 | `synaptic-cli calibrate --db ... --embed-url ...` 명령 |
| P6-5 | EvidenceSearch에도 동일 hook |

### P7: Oracle / MSSQL CDC 지원

v0.14.0에서 Phase 6 follow-up으로 미룬 항목. dialect별 placeholder 차이(`:1` vs `?`)만 해결하면 나머지는 기존 orchestrator 재사용 가능.

| # | 작업 |
|---|------|
| P7-1 | `_read_oracle_rows(..., where_clause=, where_params=)` + `_read_oracle_pks()` |
| P7-2 | `_read_mssql_rows(..., where_clause=, where_params=)` + `_read_mssql_pks()` |
| P7-3 | `_translate_placeholders()` 확장 — Oracle `:1`, MSSQL `?` |
| P7-4 | `sync_from_oracle()` / `sync_from_mssql()` 오케스트레이터 |
| P7-5 | Env-var opt-in integration 테스트 |

---

## v0.18.0+ — 장기 개선 (미확정)

우선순위/일정 확정 안 된 항목. 사용자 요청 / 실사용 데이터를 본 후 결정.

### 평가 인프라

- **LLM-as-Judge 벤치 모드** — `eval/run_all.py --judge-llm` 옵션. 현재 ID 매칭 GT는 집계/패러프레이즈 쿼리에 부정확. LLM judge가 더 정확한 정답성 판정.
- **CI 벤치 회귀 가드** — GitHub Action에 `--quick` bench 정기 실행 (nightly or release tag push). `qa_latest.json` vs 비교해서 5% 이상 회귀면 CI fail. Stale baseline 함정 영구 봉인.
- **Synthetic query generation** — 새 corpus 추가 시 자동 query 보강.

### 검색 품질

- **Doc2Query++** — 인제스트 시 문서당 예상 쿼리 5개 LLM 생성 → properties에 저장 → FTS 인덱싱. PublicHealthQA 같은 도메인 특화 recall 개선.
- **Multi-vector / late interaction** — ColBERT-style. `reranker_colbert.py` 골격은 있지만 default 아님.
- **Query decomposition 고도화** — 복합 쿼리(A AND B, X OR Y) LLM 분해 → 병렬 검색 → RRF 융합. `query_decomposer.py` 기초만 있음.
- **Cross-encoder default fallback** — 현재 BYO(TEI)만 지원. Default 경량 모델 1개 ship.

### 운영 인프라

- **Observability** — 메트릭 export (Prometheus/OpenTelemetry), health check endpoint, connection pooling.
- **Streaming MCP responses** — 긴 ingest/sync 작업에 progress 스트림.
- **Cost tracking** — 임베딩/LLM API 호출 누적 집계.
- **Quantized embeddings** (int8/binary) — 100k+ corpus 메모리 절감.
- **Multi-embedder routing** — 노드 종류별 다른 임베더.

### 확장성

- **A/B testing 인프라** — 두 검색 설정을 같은 query set으로 비교.
- **Continuous learning** — 사용자 피드백(reinforce) 기반 weight auto-tuning.
- **Multi-tenant** — 테넌트별 격리, RBAC, 감사 로그. v0.4 시점에 P7으로 잡혔지만 실사용 요청 없어 미뤄짐.

### 알고리즘 영감의 출처 정리 (2026-04 재평가)

v0.18 트랙의 alg-level 영감은 **paper 들이 원전**. 기존 v0.17.x 시리즈에서 Graphify 같은 productization 프로젝트를 비교하며 흡수 항목 5개 (G1-G5) 검토했지만 **재평가 결과 algorithm 측면 신규 가치는 G1 (auto agent priming) 1 가지만**. 나머지는 GraphRAG / Neo4j / standard converter 의 차용.

#### Algorithm-level inspiration (paper 직접 reference)

- **HippoRAG 2** (arXiv:2502.14802) — query→triple linking, MuSiQue R@5 +12.5pp 의 진짜 mechanism
- **GraphRAG** (Microsoft, arXiv:2404.16130) — community detection + summary 의 원전
- **PropRAG** (arXiv:2504.18070) — proposition graphs vs triples
- **Adaptive RAG / Self-RAG / CRAG** (2024) — agent-internal critique mechanism
- **HyDE** (Gao 2022) — query→hypothetical answer (우리 시도 후 KRRA 도메인에서 실패 측정 완료)
- **Late Chunking** (Jina 2024) — embedding-then-chunk
- **LightRAG** (arXiv:2410.05779) — dual-level retrieval

이들이 v0.18 main track (Agent-Native B + Graph-Augmented Anchor A) 의 진짜 영감원. 자세한 분석: [`docs/PLAN-v0.18-architecture.md`](PLAN-v0.18-architecture.md).

#### Productization-inspired absorption — G1만

##### G1. Auto graph snapshot / agent priming

**문제**: Agent 가 cold start 시 corpus 구조 모르고 시작 → 첫 1-2 turn 은 탐색에 낭비. Graphify (`safishamsi/graphify`) 가 `GRAPH_REPORT.md` 자동 생성/주입으로 해결한 UX 패턴.

**Synaptic 적용**:
- `synaptic snapshot <db> --output graph.md` CLI 신규
- 출력: 카테고리 트리, top phrase hub (DF 기준), entity-table 분포, edge type 통계, sample queries
- MCP `knowledge_snapshot()` 도구 추가 → agent 시작 시 1회 호출, system prompt 에 inject
- Agent-Native (B) 트랙의 부분으로 통합

**예상 작업**: 1 주 (markdown generator + MCP tool + 테스트)
**우선순위**: 🟢 v0.18 alpha — Agent-Native (B) 와 묶음
**왜 가치 있나**: 측정상 agent 81% 의 첫 1-2 turn 이 탐색용. priming 으로 그 turn 절약 → latency ↓ + context overflow 회피

#### 차용 안 함 (G2-G5)

이전 검토에서 흡수 후보였던 4 항목, 재평가 후 **모두 v0.18 main track 외부로 격하**:

| ID | 항목 | 격하 이유 | 재배치 |
|---|---|---|---|
| ~~G2~~ | Edge confidence + provenance | Neo4j 등에서 옛날부터 표준. Graphify 의 `EXTRACTED/INFERRED/AMBIGUOUS` 라벨은 임의적. **알고리즘 신규 X** | "운영 인프라" 항목으로 이동, v0.19+ 자체 설계 |
| ~~G3~~ | Leiden community detection | **Microsoft GraphRAG (2024) 가 원전**, Graphify 는 차용. 직접 GraphRAG paper reference 가 옳음 | 위 algorithm-level inspiration 으로 흡수, 별도 트랙 X |
| ~~G4~~ | Multimodal converter pack | Whisper / pdfplumber / vision LLM 모두 standard. **알고리즘 0**, 단순 packaging. LangChain / LlamaIndex 가 더 잘 함 | "검색 품질" 항목으로 이동, 사용자 요청 받아 결정 |
| ~~G5~~ | Hyperedges | 학술적 흥미, 실용 미미. design rationale 같은 niche 사용 사례. | Skip. 향후 사용자 요청 시 재검토 |

#### 의도적 비흡수 (Synaptic 차별성 보존)

- NetworkX + JSON storage — production scale 불가, 우리 multi-backend 유지
- Indexing-time LLM 호출 — LLM-free 원칙 위반, 우리 핵심 차별
- Tree-sitter AST 25-lang 코드 파싱 — 1차 도메인 (한국어 enterprise) 무관

---

## 완료된 작업 (historical)

### v0.1.0 ~ v0.5.0
뇌 신경망 기반 코어 라이브러리 + PostgreSQL 백엔드 + 문서화 + PyPI 배포.
전체 내용은 git history 참조.

### v0.6.0 ~ v0.12.0
3세대 retrieval 파이프라인, HippoRAG2 phrase hub, DomainProfile, MCP 서버
(29개 도구), 멀티턴 에이전트, 구조적 쿼리 도구 (`filter_nodes` /
`aggregate_nodes` / `join_related`), KRRA / assort / X2BEE 벤치마크 데이터셋.

### v0.13.0 — 2026-04-13
Graph-aware agent search + structured data tools. `from_database()` one-liner.
29 MCP tools. 687 tests. 이 시점이 v0.14.x 시리즈의 출발 베이스.

### v0.14.0 → v0.15.0 — 2026-04-14 / 2026-04-15
- Live database CDC (7-phase 구현, X2BEE 프로덕션 검증)
- MCP ingest / CDC tools (29 → 36 tools)
- Embedder-agnostic vector threshold (magic number 제거)
- `knowledge_search` → EvidenceSearch
- Phrase hub wiring fix (cross-document bridges)
- `graph.backfill()` — 기존 그래프 복구 도구
- `graph.search(engine="evidence")` opt-in + deprecation timeline

자세한 내용은 [CHANGELOG.md](../CHANGELOG.md)의 [0.14.0] ~ [0.15.0] 섹션.

---

## 설계 원칙 (전 버전 공통)

1. **Zero core deps** — 코어 로직은 순수 Python, 백엔드/확장만 extras
2. **Protocol 기반** — 인터페이스 교체 가능, 테스트 용이
3. **Async-first** — 모든 I/O는 async/await
4. **Memory safe** — 캐시 크기 제한, context manager 강제
5. **한/영 이중 언어** — 동의어 맵, 토크나이저, 퍼지 매칭 모두 한국어 지원
6. **LLM-free indexing** — 인덱스 시점 LLM 호출 0. 벡터 임베딩만 API 호출.
7. **BYO embedder/reranker** — torch-free. 사용자가 Ollama / TEI / API 직접 선택.
8. **Silent failure는 버그** — v0.14.x 시리즈의 핵심 교훈. 기능이 wire 안 되면 명확한 에러나 warning이 나야 함. 조용히 기능이 죽어 있는 건 안 됨.
