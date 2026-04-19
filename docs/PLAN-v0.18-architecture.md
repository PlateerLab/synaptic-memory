# PLAN — v0.18.0 Architecture

> 작성일: 2026-04-19 (v0.17.1 measurement-driven 회고)
> 상태: Design draft — 5 architectural questions raised by 14-bench + agent measurement
> 목적: 표면 패치 사이클을 끝내고 **본질적 알고리즘 / 그래프 설계 결정** 정리

---

## 0. Why this doc

v0.17.1 까지 ship한 것:
- `rerank_blend` 0.4 → 0.1 (튜닝)
- Kind-aware aggregator (ENTITY → MMR/cap bypass)
- Cross-encoder reranker skip for `_table_name` 노드
- `DomainProfile.table_query_hints` + augmentation
- assort.toml 프로파일 (X2BEE는 net-negative 로 보류)

이 변경들이 측정치를 개선했지만 **5개의 깊은 설계 질문이 측정 데이터로 부각**됐다. 표면 패치만으로는 더 이상 진전 없음. 이 문서는:

1. 측정 데이터로 본 알고리즘 한계 정리 (§1)
2. 각 한계가 제기하는 아키텍처 질문 5개 (Q1-Q5, §2)
3. v0.17.1 / v0.18.0 / research backlog 분리 (§3-§5)

---

## 1. 측정 데이터 스냅샷 (2026-04-19)

### 1.1 14-벤치 single-shot

| 벤치 | FTS-only | v0.17.0 (R2) | v0.17.1 (R3-I) | Δ vs FTS | Δ vs v0.17.0 |
|---|---:|---:|---:|---:|---:|
| KRRA Easy | 0.967 | 0.967 | 0.967 | 0 | 0 |
| KRRA Hard | 0.583 | 0.593 | 0.606 | +0.023 | +0.013 |
| KRRA Conv | 0.146 | 0.139 | 0.155 | +0.009 | +0.016 |
| **assort Easy** | 0.760 | 0.767 | **0.856** | **+0.096** | +0.089 |
| assort Hard | 0.000 | 0.000 | 0.000 | 0 | 0 |
| **assort Conv** | 0.425 | 0.268 | **0.472** | +0.047 | **+0.204** |
| X2BEE Easy | 1.000 | 1.000 | 1.000 | 0 | 0 |
| X2BEE Hard | 0.379 | 0.250 | 0.368 | **−0.011** | +0.118 |
| X2BEE Conv | 0.167 | 0.123 | 0.164 | −0.003 | +0.041 |
| HotPotQA-24 | 0.875 | 0.979 | 0.979 | +0.104 | 0 |
| Allganize RAG-ko | 0.947 | 0.982 | 0.982 | +0.035 | 0 |
| Allganize RAG-Eval | 0.911 | 0.946 | 0.946 | +0.035 | 0 |
| **PublicHealthQA** | 0.547 | 0.734 | 0.732 | **+0.185** | −0.002 |
| **AutoRAG** | **0.906** | 0.766 | 0.767 | **−0.139** | +0.001 |
| **평균** | 0.615 | 0.608 | **0.642** | +0.027 | +0.034 |

**Highlight**:
- v0.17.1 평균 처음으로 FTS-only 상회 (+4.4%)
- 그러나 **AutoRAG 와 X2BEE Hard 는 여전히 FTS 단독이 우위**

### 1.2 6-벤치 agent (Qwen3.5-27B vLLM)

| 벤치 | Single-shot MRR | Agent solved | v0.13 agent | Δ vs v0.13 |
|---|---:|---:|---:|---:|
| KRRA Hard | 0.583 (FTS) → 0.606 | 30/39 (77%) | 11/15 (73%) | +4pp |
| assort Hard | 0.000 | 30/33 (91%)* | 13/15 (87%) | +4pp |
| X2BEE Hard | 0.379 (FTS) → 0.368 | **19/19 (100%)** | 17/19 (89%) | +11pp |
| KRRA Conv | 0.146 → 0.155 | **14/30 (47%)** | 21/30 (70%) | **−23pp** ⚠ |
| assort Conv | 0.425 → 0.472 | 22/24 (92%)* | 20/24 (83%) | +9pp |
| X2BEE Conv | 0.167 → 0.164 | 25/27 (93%)* | 22/27 (81%) | +12pp |
| **평균 solved** | | **140/172 = 81.4%** | | |

(*) Context overflow 로 일부 쿼리 fail (총 10/172 = 5.8%). 16k vLLM max_model_len 한계.

**Highlight**:
- 5/6 벤치에서 v0.13 GPT-4o-mini agent 결과 초과 (Qwen3.5-27B 더 강함)
- Single-shot 0.0 → agent 91% 같은 극적 변화 — **Synaptic의 진짜 알고리즘은 agent loop**
- KRRA Conv 만 회귀 — Qwen 한국어 conversational reasoning 또는 search quality 의심

---

## 2. 5가지 아키텍처 질문

### Q1 — Search default = single-shot vs agent-loop?

**측정 근거**:
- Single-shot Hard/Conv 평균 ~0.4
- Agent 평균 0.81 (2× 차이)
- v0.17 의 "kind-aware" 등 단일 쿼리 개선은 1-10pp 단위. Agent 변화는 30-100pp

**현 narrative**: "LLM-free retrieval + optional agent"
**대안 narrative**: "Agent-loop default. Single-shot = diagnostic fallback"

**Implications**:
- README / docs / examples 모두 agent-first 로 재구성
- `synaptic-mcp` 의 default tool 이 `deep_search` (agent) 가 됨
- **단점**: LLM 의존 — but v0.17 에서 BYO LLM 이 표준이 됐으니 자연스러움
- **단점 2**: Agent latency (KRRA Hard 22 분 / 40q = 33s/query)  — production batch 용도엔 너무 느림

**추천 (v0.17.1)**: Narrative 격상은 하되 single-shot deprecation 은 X.
- README Benchmarks 섹션을 "Single-shot vs Agent" 2-col 표로 재작성
- 본문에 "Single-shot is the floor. Agent-loop (deep_search) is recommended for hard / conv queries" 명시
- Agent latency → batch processing 고려 (v0.18 트랙)

### Q2 — Indexing = LLM-free 유지 vs Selective LLM 도입?

**측정 근거**:
- MuSiQue 격차 0.453 vs HippoRAG2 0.747 (R@5)
- 3-round mechanism 추가 모두 실패
- HippoRAG2 의 핵심은 LLM OpenIE triple

**현 원칙**: "인덱싱 LLM 0원" — Korean enterprise corpus (high volume) 가정
**대안**: opt-in `--llm-ingest` 모드

**Implications**:
- 기본은 LLM-free 유지 (CI / laptop scenario)
- `extensions/openie_extractor_llm.py` 신규 모듈 (BYO LLM, vLLM/Ollama 호환)
- 인제스트 시 chunk 별 triple 추출 → `Triple` 노드 + `(subject) -[predicate]- (object)` edges
- 검색 시 query → triple 임베딩 매칭 (HippoRAG2 mechanism)
- **트레이드오프**: 인제스트 비용 (21k MuSiQue docs × ~3s/doc = ~17h on Qwen3.5-27B). 한 번만 하면 영구.

**추천 (v0.18.0)**: Selective LLM ingest 트랙 진입
- v0.17.1 에 docs/PLAN-v0.18-openie.md 설계 문서만
- v0.18.0 알파에 prototype 구현
- v0.18.0 베타에 MuSiQue 재측정

### Q3 — Pipeline = uniform vs adaptive?

**측정 근거 (가장 강력)**:
- AutoRAG: FTS 0.906 → +reranker 0.766 (−15%)
- X2BEE Hard: FTS 0.379 → +reranker 0.250 (−34% v0.17.0, −3% v0.17.1 with skip)
- PublicHealthQA: FTS 0.547 → +reranker 0.734 (+34%)
- 같은 reranker 가 corpus 별로 정반대 효과

**현 알고리즘**: 모든 쿼리가 `rerank_blend=0.1` 고정 → corpus 평균 0.1 적합한 corpus만 이득

**대안 알고리즘들**:

(a) **Per-query adaptive blend (variance-based)**
- Reranker top-K 점수의 분산 측정. 분산 낮음 (모두 비슷) = reranker 무신호 → blend → 0
- 분산 높음 (일부만 강한 신호) = reranker 신호 있음 → blend = base
- **장점**: corpus 무관, query 단위 자동
- **단점**: variance 계산 1 추가 (cheap)

(b) **Per-corpus calibration at ingest**
- 인제스트 후 corpus 에서 N(=20) 합성 쿼리 자동 생성
- 각 step (vec / PRF / rerank / MMR) 의 효과 측정 → corpus-specific config 저장
- 검색 시 그 config 사용
- **장점**: corpus characteristic 깊이 반영
- **단점**: 합성 쿼리 품질 의존, 인제스트 시간 +1 분

(c) **Per-query routing (intent classifier)**
- LLM (or rule) 으로 쿼리 분류 → entity-seeking / passage-seeking / multi-hop
- 분류별 sub-pipeline 호출
- **장점**: 의미적 분기
- **단점**: classifier latency, training 필요

**추천 (v0.17.1)**: (a) Adaptive blend 즉시 구현
- 30 분 코드, 30 분 측정. AutoRAG 자동 회복 가능성 큼
- (b) calibration / (c) routing 은 v0.18.0 트랙

### Q4 — Graph schema = flat vs hierarchical?

**측정 근거**:
- assort q003: 10 개 동명 product → GT 1 개. canonical / variant 개념 부재
- X2BEE: pr_goods_base / pr_goods_user_feedback / pr_goods_sold_hist 가 평면 ENTITY
- 정형 / 비정형 mixed corpus 검색 시 score normalization 어려움 (CHUNK score 0.7 vs ENTITY score 0.95 비교 의미)

**현 schema**: NodeKind 17 개, EdgeKind 13 개. 모두 평면.
**대안**: 3-tier hierarchy
- **Top tier (TYPE/SCHEMA)**: 도메인 ontology — 카테고리, 타입 정의
- **Mid tier (INSTANCE)**: canonical entities — 중복 제거된 row, phrase hub
- **Bottom tier (TEXT)**: chunks, raw passages
- **Cross-tier edges**: HAS_TYPE, INSTANCE_OF, MENTIONED_IN, DESCRIBED_BY

**검색 routing**:
- "list all X" → top-tier (type filter)
- "what does Y say about Z" → bottom-tier (passage retrieval)
- "X's price/property" → mid-tier (entity attribute)

**Implications**:
- 큰 변경. backend schema migration 필요
- 기존 corpus 모두 재인제스트 (또는 lazy migration)
- 검색 라우팅 로직 신규
- **그러나** mixed corpus first-class support 가능

**추천 (v0.18.0+)**: 본격 트랙. v0.17.1 에서는 nothing.
- v0.17.1 까지의 kind-aware 는 "flat schema 위 patch"
- 진정한 해결은 schema 자체 hierarchy

### Q5 — Reranker integration = global default vs per-corpus auto-detect?

**측정 근거**: Q3 와 동일.
- Default 0.1 — 14-bench 평균 best, but AutoRAG / X2BEE Hard 음수
- corpus 별로 0.0 / 0.1 / 0.4 가 모두 best 가능

**Q3 (a) adaptive blend** 가 사실상 Q5 의 답. 별도 결정 사안 아님 — Q3 의 부분.

---

## 3. v0.17.1 최종 scope

이미 완료 (이 PR 까지):
- `EvidenceAggregator` kind-aware split
- `EvidenceSearch` cross-encoder skip for `_table_name` 노드  
- `DomainProfile.table_query_hints` + 로더
- `QueryAnchorExtractor.preferred_tables`
- `EvidenceSearch` table hint augmentation (gated by `<3 FTS hits in target table`)
- `assort.toml` 에 hints 추가
- `eval/run_all.py` profile loading
- `_llm_judge` model 파라미터화 (vLLM agent 모드 지원)
- 32 unit tests passing

**추가 ship (이 sprint, ~3-5 시간)**:
- **Q3 (a) Adaptive blend** 구현 — variance-based per-query
- Context overflow 완화 — agent_tools 결과 truncation (~30 분)
- KRRA Conv regression 진단 (1 시간) — fix 가능 시 추가, 불가능 시 known issue 문서화
- Round 4 측정 → adaptive blend 효과 검증
- CLAUDE.md / README / CHANGELOG v0.17.1 반영 (single-shot + agent 2-col)
- PyPI publish

## 4. v0.18.0 proposed scope (~1 개월)

핵심 트랙 4 개:

### 4.1 **Q3 (b) Per-corpus calibration at ingest**
- `synaptic.calibration` 신규 모듈
- 인제스트 후 합성 쿼리 N=20 자동 생성 (LLM-free: phrase hub + chunk title 기반)
- pipeline step 별 효과 측정 → `_meta.pipeline_config` 저장
- 검색 시 자동 적용

### 4.2 **Q4 Hierarchical schema (TYPE/INSTANCE/TEXT)**
- `synaptic.schema_v2` 신규 모듈 — 3-tier graph layout
- Backward compat: 기존 flat graph 는 single-tier 로 작동
- 마이그레이션 도구 `synaptic migrate v1-to-v2`
- Mixed corpus 시나리오 테스트 (e-commerce + FAQ + reviews)

### 4.3 **Q1 Agent-default narrative + tooling**
- `synaptic-mcp` default tool 변경: `search` → `deep_search`
- Agent latency 완화: batch agent runner (`synaptic agent batch <queries.json>`)
- README 전면 재구성

### 4.4 **Entity resolution at ingest**
- 중복 entity 검출 + canonical / variant 표시
- title clustering + property similarity
- `IS_VARIANT_OF` edge 신규
- v0.17 의 q003 ambiguity 같은 케이스 해결

## 5. Research backlog (v0.18.0+)

### 5.1 Q2 Selective LLM ingest (OpenIE triple)
- 가장 큰 잠재력 (MuSiQue 0.294 격차)
- 가장 큰 비용 (인제스트 비용 + 코어 의존 추가)
- Prototype 후 user demand 기반 정식 트랙 결정

### 5.2 Q3 (c) Per-query intent routing
- 자동 corpus calibration 우선. (c) 는 그 이후 추가 정밀화

### 5.3 Multi-turn agent context window 확장
- 현재 5 turn × ~3k token = 15k 도달 → context overflow 5.8%
- Tool result projection (only-relevant-properties)
- 또는 streaming summarization

### 5.4 Mixed corpus first-class
- Q4 hierarchical schema 가 enabler
- 별도 테스트 corpus 생성 필요

---

## 6. 결정 사항 요약

| Q | 사안 | v0.17.1 | v0.18.0 | Research |
|---|---|---|---|---|
| Q1 | Agent default? | Narrative 격상 | MCP default 변경 + batch runner | Streaming agent |
| Q2 | LLM ingest? | 유지 (LLM-free default) | 설계 문서 | Prototype |
| Q3 | Adaptive pipeline? | (a) Variance blend 즉시 | (b) Calibration | (c) Intent routing |
| Q4 | Hierarchical schema? | — | 본격 구현 | Mixed corpus support |
| Q5 | Reranker per-corpus? | Q3 (a) 의 부분 | Calibration 의 부분 | — |

---

## 7. 외부 참조 (HippoRAG2, GraphRAG 등)

- HippoRAG 2 (arXiv:2502.14802) — query→triple linking, +12.5pp R@5
- LightRAG (arXiv:2410.05779) — dual-level retrieval (low/high-level)
- GraphRAG (arXiv:2404.16130) — hierarchical community summary
- PropRAG (arXiv:2504.18070) — proposition graphs (vs triples)
- LinearRAG (2025) — 1-hop is enough

이들의 알고리즘적 통찰을 v0.18 calibration / hierarchical schema 설계에 흡수.

## 7.1 Competitive scan — Graphify (safishamsi/graphify)

2026-04 조사. 코드/멀티모달 corpus → NetworkX in-memory graph 변환 + MCP server. 동일 *agent + graph + MCP* 시장 정면 경쟁자.

### Graphify 의 표면적 차별 항목

| 항목 | Graphify 구현 | Synaptic 현재 |
|---|---|---|
| Edge metadata | typed + confidence (`EXTRACTED` 1.0 / `INFERRED` scored / `AMBIGUOUS`) + provenance | structural + weight only |
| Agent priming | auto-generated `GRAPH_REPORT.md` injected into context (top entities + community summary) | 없음 — agent 가 cold start |
| Community layer | Leiden detection + per-community summary | CONCEPT 카테고리만 (수동) |
| Multimodal | native pipeline (Whisper transcription + image/PDF/video) | 외부 변환기 의존 |
| Hyperedges | 3+ 노드 묶는 design rationale 보존 | binary edges only |

### 비판적 재평가 — 알고리즘 신규성은 G1 만

5 항목 중 algorithm-level 신규 가치 있는 건 **G1 (auto agent priming)** 뿐. 나머지 4 개는:

- **Edge confidence/provenance**: Neo4j 등이 옛날부터 표준 — `EXTRACTED/INFERRED/AMBIGUOUS` 라벨은 임의적
- **Leiden community detection**: **Microsoft GraphRAG (arXiv:2404.16130) 가 원전**, Graphify 는 차용
- **Multimodal converter**: Whisper/pdfplumber/vision LLM 등 standard tools 의 packaging — 알고리즘 0
- **Hyperedges**: 학술적 흥미, 실용 미미 (1970s 부터 존재)

또한 Graphify 의 "**71.5× 토큰 절감**" 마케팅은 **방법론 미공개** + raw-file dump 와 비교한 것으로 추정 — prompt cache 만 적용해도 비슷한 수치. **Retrieval 품질 (MRR/Recall) 측정은 부재**.

### 흡수 결정

| ID | 항목 | 결정 | 사유 |
|---|---|---|---|
| **G1** | Auto agent priming | ✅ 흡수 — v0.18 alpha, Agent-Native (B) 의 부분 | UX 패턴 신규, 우리 측정상 agent 첫 1-2 turn 절약 가치 |
| ~~G2~~ | Edge confidence | ❌ Skip Graphify 식 — production 표준 직접 따라감 | Neo4j 등 옛날부터 표준 |
| ~~G3~~ | Community detection | ❌ Skip Graphify 식 — GraphRAG paper 직접 reference | Graphify 는 GraphRAG 차용 |
| ~~G4~~ | Multimodal converter | ❌ Skip Graphify 식 — LangChain / 자체 구현 | Algorithm 0, packaging 차이만 |
| ~~G5~~ | Hyperedges | ❌ Skip — 사용자 요청 시 재검토 | Niche use case |

### 의도적 비흡수 (Synaptic 차별성 보존)

- **NetworkX in-memory + JSON 단일 파일 storage** — production scale 불가. 우리 multi-backend 유지.
- **Indexing 시 Claude subagent 호출** — LLM-free indexing 원칙 위반. Synaptic 의 차별성 핵심.
- **Tree-sitter AST 25-language 코드 파싱** — Synaptic 의 1차 도메인 (한국어 enterprise) 와 무관. 별도 extras 패키지로 community 기여 가능.

### 진짜 알고리즘 SOTA 의 출처 (Graphify 가 아닌 곳)

`§7` 외부 참조 — paper 직접 reference. Graphify 는 productization / marketing 영역에서 학습할 가치 있지만 **algorithm 측면 학습 가치는 G1 1 항목으로 한정**.

---

## 8. 문서 이력

- 2026-04-19: 초안. v0.17.1 measurement-driven 회고. 5 architectural questions 정리.
