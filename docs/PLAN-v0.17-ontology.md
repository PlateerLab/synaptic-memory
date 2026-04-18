# PLAN — v0.17 온톨로지 고도화 우선순위 재평가

> 작성일: 2026-04-18
> 상태: **Case B 확정** — MuSiQue 500q 재측정 완료 (R@5 0.453 < 0.5)
> 대상: v0.17.x 시리즈 scope 결정 + v0.18.0+ 재정렬

---

## 1. 배경

v0.16.0 Tier-1 벤치마크 결과:

| Dataset | R@5 | Hit@10 | HippoRAG2 (published R@5) |
|---|---:|---:|---:|
| HotPotQA dev (500q) | 0.585 | 91.8% | — (str.acc. 56.7%) |
| MuSiQue-Ans dev (500q) | **0.379** | 76.2% | **0.747** |
| 2Wiki dev (500q) | 0.501 | 91.2% | 0.904 |

**MuSiQue/2Wiki에서 HippoRAG2 대비 R@5 격차가 큼.** 사용자 요청("온톨로지 고도화")에 대한 초기 우선순위를 다각도로 재검증했고, 원래 제안이 **틀렸다**는 결론.

---

## 2. 기각된 원안

> v0.17.0 = Typed relation extraction을 인덱싱에 통합 (A) + Query decomposition 을 default 파이프라인에 통합 (B)

4개 관점(진단 / Shelfware / 경쟁사 실체 / 구현 공수)에서 동시에 반박됨 — §3 참조.

---

## 3. 4개 관점 교차검증 요약

### 3.1 진단 (MuSiQue 실패 원인)

**가장 큰 발견: MuSiQue 0.379는 "embedder OFF" 측정값.**

- `examples/ablation/run_tier1_benchmarks.py:123` — `SynapticGraph(backend)` 로 임베더 미주입
- `evidence_search.py:175-185` — embedder 받으면 vector seed + PRF 모두 작동하도록 이미 배선됨
- 본인 `CLAUDE.md` 에도 "embedder + reranker 켜면 점수가 훨씬 올라가지만 그 측정은 Home 서버 기동이 필요해 CI/일상 검증에선 FTS-only가 default" 라고 적혀있음

**순위** (확신도 순):
1. Query decomposition 미통합 (`query_decomposer.py` 있으나 `evidence_search.py` 에서 우회)
2. Typed semantic relation 부재 (indexing 시 `MENTIONS` / `PART_OF` / `CONTAINS` / `NEXT_CHUNK` 만 생성)
3. **Embedder OFF + per-doc cap=2** — 측정 조건 자체의 문제

### 3.2 Shelfware 스캔 — 켜기만 하면 되는 것들

| 작업 | 예상 임팩트 | 공수 | GPU |
|---|---:|---:|---|
| Embedder 주입 (qwen3-embedding:4b 등) | +0.25 MRR | 30분 | GPU inference |
| Cross-Encoder reranker 주입 (TEI bge-reranker-v2-m3) | +0.10 MRR | 1시간 | GPU 필수 |
| PPR 파라미터 튜닝 (damping, top_k) | +0.05 MRR | 2시간 | CPU |
| Query Rewriter 주입 (`rewriter.py:26`, default `None`) | +0.08 MRR | 4시간 | Haiku ~$0.001/q |
| Chunk size / overlap 튜닝 | +0.05 MRR | 1주 | — |

합계: **~7시간 작업으로 +0.5 MRR 가능성**. 전부 코드는 있고 wiring만 빠짐.

근거:
- `src/synaptic/graph.py:115-132` — SynapticGraph `__init__` defaults (전부 `None`)
- `src/synaptic/extensions/hybrid_reranker.py:122-126` — RerankerWeights
- `src/synaptic/extensions/evidence_search.py:304-312` — cross-encoder blend 코드

### 3.3 HippoRAG2 실체 해부

**HippoRAG2 의 R@5 0.747을 만드는 것은 "typed relation" 이 아니다.**

HippoRAG2 파이프라인 실제 (논문 arXiv:2502.14802, Table 5 ablation):
1. Llama-3.3-70B OpenIE → **schema-less untyped phrase node** 그래프 (= typed relation 아님)
2. **query-to-triple linking** (NV-Embed-v2로 쿼리를 트리플 임베딩에 매칭) → **+12.5%p Recall@5** (가장 큰 단일 기여)
3. passage node 추가 → +6.1%
4. LLM recognition filter → +0.7%

기준선: NV-Embed-v2 단독으로 이미 **MuSiQue R@5 69.7%**. 즉 0.747 중 대부분은 **강력 임베더**의 기여. typed relation 자체의 알파는 ~+5%p 수준.

**2025 트렌드**: PropRAG (arXiv:2504.18070) — "triple은 lossy compression" 이라 비판하며 proposition(자연어 그대로) + beam search 로 **HippoRAG2 +2.4% 능가** (MuSiQue R@5 75.4%). 학계는 이미 typed relation에서 proposition/late-interaction으로 이동 중.

**정체성 충돌**: LLM 기반 typed relation OpenIE 는 Synaptic 의 "LLM-free indexing, 코어 의존성 0, offline-first" 정체성과 정면충돌.

### 3.4 구현 공수 / 리스크

| 항목 | A. Typed relation 통합 | B. Query decomp 통합 |
|---|---|---|
| Protocol 정의 | ✅ `RelationDetector` 있음 (`protocols.py:116`) | ❌ 없음 |
| 테스트 | **0건** | 13 cases (rule-based만) |
| 실제 wiring 위치 | document_ingester / entity_linker 통과 안 함 | legacy 경로만, EvidenceSearch 우회 |
| 수정 파일 수 | 7개 (models / detector / linker / ingester / ppr / expander / backends) | 4개 (evidence_search / decomposer / agent_tools_v2 / graph) |
| 최대 리스크 | **LLM-free indexing 원칙 위반** + corpus sweep GPU 시간 폭증 (수시간~수십시간) | 결합 정책 모호 (RRF vs seed merge vs 평행) |

---

## 4. 재정립된 우선순위

### 🔥 P0 — 진단 재측정 (1-2일, GPU 활용 명분 100%)

**임베더 + 리랭커 켜고 Tier-1 다시 측정.** 이게 안 되면 v0.17 설계 전체가 모래성.

| 작업 | 내용 |
|---|---|
| P0-1 | H100 또는 home(14.6.220.78) 서버에 Ollama `qwen3-embedding:4b` + TEI `bge-reranker-v2-m3` 기동 |
| P0-2 | `examples/ablation/run_tier1_benchmarks.py` 에 `--embedder` / `--reranker` CLI flag 추가 |
| P0-3 | 5개 데이터셋 (HotPotQA 500q + MuSiQue 500q + 2Wiki 500q + Allganize RAG-ko + AutoRAG) 재측정 |
| P0-4 | `docs/comparison/synaptic_results.md` 에 "Embedder ON Tier-1" 섹션 추가 |
| P0-5 | `eval/baselines/qa_latest.json` 갱신 (v0.16.0 + embedder) |

**완료 조건**: Embedder ON 베이스라인 확정. MuSiQue R@5 회복 폭에 따라 P1 분기.

### 📊 P0 측정 결과 (2026-04-18, Stage 1 완료 · Stage 2/3 미완)

**구성**: H100, bge-m3 (FP16, cuda:0) + bge-reranker-v2-m3 (FP16, cuda:0), SqliteGraphBackend (usearch HNSW), embed batch=64, `examples/ablation/local_bge.py` 경로.

| Dataset | Corpus | Queries | MRR | R@5 | R@10 | Hit@10 | Build | Search | v0.16.0 FTS-only baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MuSiQue-Ans dev** | 21,100 docs | 500q | **0.729** | **0.453** | 0.480 | 400/500 (80.0%) | 98.9s | 476.1s | R@5 0.379 / Hit 76.2% |

**비교 기준**: HippoRAG2 공개값 MuSiQue R@5 **0.747** → 격차 **-0.294**.
Embedder OFF(v0.16.0) 대비 R@5 +0.074 (+19.5%p 상대) — 임베더 효과는 확인됐으나 근본 해결 불가.

#### Stage 2/3 미완 사유

- Stage 1 (MuSiQue 500q): ✅ 완료 — `examples/ablation/diagnostics/tier1_20260418_172317.md`
- Stage 2 (HotPotQA + 2Wiki 100q each): ❌ 3h20m 후 silent death. HotPotQA 66k corpus 인덱싱 중 CUDA OOM 추정 (vLLM Qwen3.5-27B 와 공존으로 free VRAM ~5.7GB).
- Stage 3 (Allganize RAG-ko / RAG-Eval): ❌ Stage 2 실패로 연쇄 skip (`set -e`).

**판단**: Stage 1 만으로도 v0.17.0 설계 분기(Case A/B) 결정에는 충분. 재측정은 publication 단계(v0.17.0 release note)로 이월.

#### 해석

1. **Embedder + Reranker 로는 R@5 0.5 선 돌파 불가.** 격차의 원인이 임베더 품질이 아닌 **쿼리-증거 매핑 구조** 에 있음.
2. MuSiQue 는 2-4 hop 멀티홉 추론 벤치 → 단일 dense retrieval + rerank 로는 중간 hop 문서 누락. **Query decomposition** 이 없으면 구조적 상한.
3. HippoRAG2 의 알파(+0.294)는 주로 **query-to-triple linking + PPR 부스트** 에서 나옴(§3.3). Synaptic 의 PPR 은 이미 있음 → **decomposer + phrase linker** 가 미싱 피스.

**결론**: **Case B 확정**. Query decomposer 통합이 v0.17.0 의 핵심.

### ⚙️ P1 — P0 결과에 따라 분기 → **Case B 확정**

> 2026-04-18 측정 결과: MuSiQue-Ans dev 500q, R@5 **0.453** (< 0.5 threshold).
> 임베더(bge-m3) + 리랭커(bge-reranker-v2-m3) ON 상태에서도 HippoRAG2 0.747 대비
> **-0.294** 격차. §4.5 참조.

#### ❌ Case A (기각됨): 임베더만으로 MuSiQue R@5 ≥ 0.6 회복

측정 결과 R@5 0.453 < 0.6. shelfware wiring 만으론 부족. 이 경로는 폐기.

아래 잡무는 v0.17.x 내 부분 작업으로만 유지 (메인 스코프 아님):

| 작업 | 근거 |
|---|---|
| Cross-encoder default 활성화 | §3.2 +0.10 MRR |
| Query-to-phrase linking 강화 | HippoRAG2 +12.5%p 의 진짜 기여 컴포넌트 |
| PPR 파라미터 corpus-adaptive 튜닝 | §3.2 +0.05 MRR |
| Chunk size 최적화 (MuSiQue 짧은 fact용) | §3.2 +0.05 MRR |

#### ✅ Case B (확정): 임베더 켜도 MuSiQue R@5 < 0.5

그래프 구조 문제 확정. v0.17.0 온톨로지 scope:

| 작업 | 근거 |
|---|---|
| **Query decomposer 를 EvidenceSearch 에 통합** | §3.1 1순위. `query_decomposer.py` 184줄 prototype 존재. 2주 이내 (§3.4) |
| **Typed relation 은 opt-in CLI sweep 도구만** | LLM-free 원칙 보존. `scripts/extract_typed_relations.py` 1회용 |
| **EdgeKind 확장 + PPR `_EDGE_TYPE_WEIGHTS` 추가** | 신규 ingest 에만 영향, 기존 그래프 호환 |

### ❌ 확정 제외

v0.17.0 에 넣지 않는다 (ROI 불명 + 원칙 충돌):

- **Typed relation 인덱싱 파이프라인 강제 통합**: HippoRAG2 도 typed 아님. LLM-free 정체성 위반. GPU sweep 수십시간. 진단도 부정확.
- **1주 사이클에 A+B 묶음**: §3.4 참조 — contract 정의 + 마이그레이션 + 회귀 측정 포함 시 최소 2주, 권장 4주.

---

## 5. 근거 파일 맵

핵심 파일 (수정 / 참조):

- `src/synaptic/graph.py:115-132` — SynapticGraph 생성자 defaults
- `src/synaptic/extensions/evidence_search.py:175-185` — embedder 주입 분기
- `src/synaptic/extensions/evidence_search.py:304-312` — cross-encoder blend
- `src/synaptic/extensions/hybrid_reranker.py:122-126` — reranker weights
- `src/synaptic/extensions/query_decomposer.py` — 184줄 prototype, Protocol 없음
- `src/synaptic/extensions/relation_detector_llm.py` — 269줄 prototype, 테스트 0건
- `src/synaptic/extensions/rewriter.py:26` — LLMQueryRewriter, default None
- `src/synaptic/models.py:54-72` — EdgeKind enum
- `src/synaptic/ppr.py` — `_EDGE_TYPE_WEIGHTS`
- `examples/ablation/run_tier1_benchmarks.py:123` — embedder 미주입 (측정 조건 문제의 진원)
- `eval/baselines/qa_latest.json` — v0.14.4 snapshot, 재측정 대기

---

## 6. v0.17.0 작업 분해 (Case B)

우선순위 순. P0-* 는 완료/이월, W-* 는 v0.17.0 신규 작업.

### ✅ 완료

- **P0-1 ~ P0-3**: `--local-bge` 경로 (`examples/ablation/local_bge.py`) + `run_tier1_benchmarks.py` / `benchmark_allganize.py` CLI flag 확장. SqliteGraphBackend 분기, 배치 임베딩 pre-compute. 완료 커밋 대기 중.
- **P0-4 (부분)**: MuSiQue 500q Embedder ON 측정 — `examples/ablation/diagnostics/tier1_20260418_172317.md`.

### 🔄 이월 (v0.17.0 release note 단계)

- **P0-4 (잔여) / P0-5**: HotPotQA + 2Wiki 100q + Allganize 재측정. vLLM 정리 또는 Home 서버 활용 후 실행. `eval/baselines/qa_latest.json` 갱신.

### 🎯 v0.17.0 신규 스코프

| ID | 작업 | 파일 | 예상 공수 |
|---|---|---|---:|
| W-1 | `QueryDecomposer` Protocol 정의 (`decompose(query) -> list[SubQuery]`) | `src/synaptic/protocols.py` | 0.5d |
| W-2 | `evidence_search.py` 에 decomposer 분기 추가 — 서브쿼리 병렬 seed → union → per-sub MMR | `src/synaptic/extensions/evidence_search.py` | 3d |
| W-3 | 서브쿼리 결합 정책: RRF (Reciprocal Rank Fusion, k=60) 로 통합 후 rerank | 동상 | W-2 포함 |
| W-4 | Rule-based decomposer 리팩터 + 테스트 보강 (`query_decomposer.py` 184줄 prototype → protocol 구현체) | `src/synaptic/extensions/query_decomposer.py` | 1d |
| W-5 | LLM decomposer 구현체 (BYO, opt-in) | `src/synaptic/extensions/query_decomposer_llm.py` | 1d |
| W-6 | SynapticGraph `__init__` 에 `decomposer=None` 파라미터 + `agent_tools_v2.deep_search` 위임 | `graph.py` / `agent_tools_v2.py` | 0.5d |
| W-7 | MuSiQue 500q + 2Wiki 500q 재측정 + decomposer ON/OFF ablation | `examples/ablation/` | 0.5d (측정) + GPU time |
| W-8 | `scripts/extract_typed_relations.py` opt-in CLI sweep 도구 (기존 `relation_detector_llm.py` 활용) | `scripts/` 신규 | 2d |
| W-9 | EdgeKind 확장 (`WORKS_FOR`, `LOCATED_IN`, `SUBSIDIARY_OF` 등) + PPR `_EDGE_TYPE_WEIGHTS` | `models.py` / `ppr.py` | 1d |

**합계**: ~9.5일 엔지니어링 + GPU 측정 시간. 2주 스프린트 내 가능.

**성공 기준**: MuSiQue 500q R@5 ≥ 0.55 (현재 0.453 → +0.1). HippoRAG2 0.747 까지는 못 가도 "구조적 개선 입증" 으로 v0.17.0 릴리즈 의미 충분.

---

## 7. 외부 참조

- HippoRAG 2 paper (arXiv:2502.14802) — <https://arxiv.org/html/2502.14802v1>
- HippoRAG GitHub — <https://github.com/OSU-NLP-Group/HippoRAG>
- PropRAG (arXiv:2504.18070) — <https://arxiv.org/html/2504.18070>
- Microsoft GraphRAG paper (arXiv:2404.16130)
- LightRAG paper (arXiv:2410.05779)

---

## 8. 문서 이력

- 2026-04-18: 초안 작성. 4개 서브에이전트 교차검증 결과 반영. P0 실행 전 상태.
- 2026-04-18 (addendum): P0 Stage 1 (MuSiQue 500q, bge-m3 + bge-reranker-v2-m3) 완료. R@5 0.453 측정 → **Case B 확정**. §4.5 / §6 재작성. Stage 2/3 은 vLLM 공존 VRAM 부족으로 이월.
