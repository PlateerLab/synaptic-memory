# PLAN — v0.17 온톨로지 고도화 우선순위 재평가

> 작성일: 2026-04-18
> 상태: **Round 1 완료** — 공개 벤치 재측정으로 v0.17.0 방향 확정 (강점 집중, MuSiQue gap 정직 공개)
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

| ID | 작업 | 상태 | 파일 / 커밋 |
|---|---|---|---|
| W-1 | `QueryDecomposer` Protocol 정의 (`decompose(query) -> list[str]`) | ✅ 완료 | `protocols.py` / `9e9dcf9` |
| W-2 | `evidence_search.py` 에 decomposer 분기 — Step 2d RRF seed fusion | ✅ 완료 | `evidence_search.py` / `9e9dcf9` |
| W-3 | RRF (k=60) 결합 정책 | ✅ W-2 포함 | 동상 |
| W-4 | Rule-based decomposer 리팩터 (`query_decomposer.py` prototype 유지 — 기존 `async decompose(query) -> list[str]` 이 이미 Protocol 만족) | ✅ Zero-change | 기존 클래스 그대로 |
| W-5 | `LLMChainDecomposer` — chain / multi-hop 전용, vLLM/Ollama/OpenAI 호환 | ✅ 완료 | `query_decomposer_llm.py` / `2eb2b3b` |
| W-6 | SynapticGraph facade 에서 EvidenceSearch 로 decomposer 위임 | ✅ 완료 | `graph.py` / `9e9dcf9` |
| W-7 | MuSiQue 500q decomposer ON/OFF ablation (Qwen3.5-27B @ vLLM 8012) | ❌ Negative | `diagnostics/tier1_20260418_220413.md` |
| W-8 | ~~`scripts/extract_typed_relations.py` CLI sweep~~ → **Query-to-phrase linking** (HippoRAG2 의 +12.5%p 컴포넌트) | 🔁 재스코프 | `evidence_search.py` / `phrase_extractor.py` |
| W-9 | EdgeKind 확장 + PPR `_EDGE_TYPE_WEIGHTS` 추가 | ⏳ 보류 (W-8 결과에 따라) | `models.py` / `ppr.py` |

**진행 상황 (2026-04-18 저녁)**: W-1~W-6 완료. 809 → 817 unit tests passing (+8 LLMChainDecomposer + 4 EvidenceSearch decomposer tests).

**중요 관찰 — rule decomposer는 MuSiQue 쿼리에서 4% split rate + 오분해 경향** (50/2 샘플, "Hannibal and Scipio" 같은 고유명사를 쪼갬). chain query 에는 LLM decomposer가 필수 — 이 발견 때문에 W-5 를 W-7 이전으로 이동.

**성공 기준 (W-7)**: MuSiQue 500q R@5 ≥ 0.55 (baseline 0.453 → +0.1). HippoRAG2 0.747 까지는 못 가도 "구조적 개선 입증" 으로 v0.17.0 릴리즈 의미 충분.

### 📉 W-7 결과 (2026-04-18 22:04) — Negative

| 지표 | Baseline (no decomposer) | LLMChainDecomposer ON | Δ |
|---|---:|---:|---:|
| MRR@10 | 0.729 | 0.696 | **−0.033** |
| R@5 | **0.453** | **0.405** | **−0.048 (−10.6%)** |
| R@10 | 0.480 | 0.421 | −0.059 |
| Hit@10 | 400/500 (80.0%) | 372/500 (74.4%) | −5.6%p |
| Search time | 476s | 1820s | 3.8× |

Decomposer가 **악화**. 성공 기준 (≥ 0.55) 미달일 뿐 아니라 baseline 보다도 낮음. 원인 분석:

1. **RRF 가중이 너무 민주적** — `1/(60+rank)` 로 원본·서브 동등 취급. 원본의 full semantic 신호가 서브쿼리 FTS 노이즈에 희석.
2. **서브쿼리가 노이즈 seed 끌어옴** — "Who distributed the film UHF?" → UHF 방송 일반 문서 대량. Weird Al 영화 맥락 손실.
3. **Reranker는 원본만 봄** — 서브쿼리가 끌어온 bridge doc이 원본 쿼리 기준으로 low score → top-N 에서 밀림. 정작 필요한 2-hop 중간 문서 사라짐.
4. **서브쿼리는 FTS-only** — bge-m3 power 활용 못함.

### 🔁 방향 재설정 — HippoRAG2 본질 재조명

HippoRAG2 의 +12.5%p Recall@5 기여분은 **query → sub-query** 가 아니라 **query → triple/phrase linking** (NV-Embed-v2 로 쿼리를 triple 임베딩에 매칭). 우리 decomposer 접근은 **잘못된 레이어**에서 작동.

**W-8 재스코프 (Query-to-phrase linking)**:

| ID | 작업 | 가설 |
|---|---|---|
| W-8a | `PhraseExtractor` 가 만든 phrase hub 노드에 bge-m3 embedding 부여 | seed 단계에서 쿼리-phrase dense match 가능 |
| W-8b | `EvidenceSearch` Step 2e 추가: 쿼리 embedding 과 phrase 노드 embedding top-K 매칭 → phrase → 연결된 문서 노드 seed | HippoRAG2 의 query-triple linking 등가물 |
| W-8c | Phrase 매칭으로 발견된 문서에 전용 `fts_scores` 밴드 부여 (vec_seeds 와 유사, 그러나 PPR teleport 점수로도 기능) | |
| W-8d | MuSiQue 500q 재측정 | 성공 기준 R@5 ≥ 0.50 |

**왜 이게 맞는 방향인가**: `phrase_extractor.py` 는 이미 corpus 인덱싱 시 phrase hub 를 만드는 중. MENTIONS 엣지 도 있음. 지금 빠진 건 **쿼리 사이드 dense 매칭**. 인제스트 비용 제로.

Decomposer 코드 (`query_decomposer_llm.py`, `QueryDecomposer` Protocol, EvidenceSearch Step 2d) 는 **유지** — opt-in default-off. 다른 corpus (compound 쿼리 많은 한국어 데이터) 에서 유용할 수 있음.

---

## 7. 외부 참조

- HippoRAG 2 paper (arXiv:2502.14802) — <https://arxiv.org/html/2502.14802v1>
- HippoRAG GitHub — <https://github.com/OSU-NLP-Group/HippoRAG>
- PropRAG (arXiv:2504.18070) — <https://arxiv.org/html/2504.18070>
- Microsoft GraphRAG paper (arXiv:2404.16130)
- LightRAG paper (arXiv:2410.05779)

---

## 9. Round 1 재측정 + blend weight 튜닝 (2026-04-18~19)

### 9.1 초기 Round 1 결과 — **+54% 내러티브는 오판**

첫 측정 (R1, 2026-04-18 23:30) 에서 공개 5 벤치 평균 MRR 0.575 → 0.843 (+54%) 보고. **하지만 FTS-only 베이스라인을 v0.14.4 값 (0.575 평균) 에서 끌어온 실수**. v0.16.0 engine flip + Kiwi 개선으로 이미 FTS-only 평균 0.837 였음.

### 9.2 재측정 삼각검증 (2026-04-19)

3 라운드 비교로 실제 component 효과 격리:

| 벤치 | R1b FTS-only | R1a bge only | R1 bge + EntityLinker | 해석 |
|---|---:|---:|---:|---|
| HotPotQA-24 | 0.875 | **1.000** | 0.979 | +14%. EL 살짝 음수 |
| Allganize RAG-ko | 0.947 | **0.972** | 0.967 | +3%. 이미 강함 |
| Allganize RAG-Eval | 0.911 | **0.925** | 0.924 | +1.5%. 이미 강함 |
| PublicHealthQA | 0.547 | **0.706** | 0.706 | +29%. 의료 paraphrase |
| **AutoRAG** | **0.906** | **0.642** | 0.638 | **−29% regression** ❗ |

**AutoRAG component isolation** (`examples/ablation/diagnose_autorag.py`):
- FTS-only: 0.906 (Hit 114/114)
- Embedder only: 0.879 (114/114) — vector seed/PRF 약간 음수
- **Reranker only: 0.641 (81/114)** ← 범인
- Embedder + Reranker: 0.642 (80/114)

**근본 원인**: `evidence_search.py:313` 의 blend 가중치 **0.4가 retrieval-style corpus 에 과도함**. AutoRAG 같이 FTS 순위가 이미 최적이면 cross-encoder 재정렬이 정답을 떨어뜨림.

### 9.3 Blend weight sweep (`examples/ablation/sweep_rerank_blend.py`)

5 벤치 × 3 blend 교차 측정:

| 벤치 | b=0.1 | b=0.2 | b=0.4 (구 default) | FTS-only |
|---|---:|---:|---:|---:|
| HotPotQA-24 | 0.979 | 1.000 | 1.000 | 0.875 |
| Allganize RAG-ko | **0.982** | 0.981 | 0.972 | 0.947 |
| Allganize RAG-Eval | **0.946** | 0.935 | 0.925 | 0.911 |
| PublicHealthQA | **0.734** | 0.719 | 0.706 | 0.547 |
| AutoRAG | **0.766** | 0.708 | 0.642 | 0.906 |
| **평균** | **0.881** | 0.869 | 0.849 | 0.837 |

**결정**: `EvidenceSearch.__init__` default `rerank_blend: float = 0.1`. 5 벤치 평균 +3.2%p, AutoRAG 19%p 회복. HotPotQA 1개 쿼리 손실은 허용 가능.

### 9.4 확정된 v0.17.0 narrative

**이전 (오판)**: "FTS-only → Full pipeline +54% 도약"
**확정**: "FTS-only 이미 강함 (v0.16.0 engine flip 덕분). Full pipeline 은 paraphrase-heavy corpora 에서 +10-34%, retrieval-style 에서는 ~−15% 까지 regression 가능. v0.17.0 의 blend=0.1 튜닝은 평균 +5.3% uplift 으로 안전하게 개선. Reranker 는 opt-in 으로 권장."

- 공개 벤치 5종 공식 값은 CLAUDE.md §"현재 베이스라인" 참조
- 9 custom 벤치 (KRRA/assort/X2BEE) 는 pre-built SQLite graph 가 H100 에 없어서 skip. Follow-up 에서 transfer or rebuild

### 9.2 v0.17.0 리포지셔닝

**이전 내러티브 (잘못된)**
> "MuSiQue 에서 HippoRAG2 따라잡자 → typed relation + query decomp 추가"

**확정 내러티브**
> "Synaptic 은 **한국어/정형 데이터 RAG** 강점. 공개 한국어 벤치 MRR 0.92+ 로 리더보드 경쟁 가능. 영어 multi-hop (MuSiQue) 은 OpenIE 기반 architecture 교체 필요하며 v0.18.0+ 연구 트랙."

### 9.3 릴리즈 스코프 (확정)

**Ship**
- `--local-bge` 경로 (transformers 직접 로드, no TEI/Ollama dependency) — `eval/run_all.py` + `run_tier1_benchmarks.py` 둘 다
- `--entity-linker` post-hoc DF-filtered phrase hub CLI flag
- `QueryDecomposer` Protocol + `LLMChainDecomposer` — **opt-in default-off** (Korean compound 쿼리 사용자용, MuSiQue 에서는 음수지만 다른 corpus 에선 positive 가능)
- 새 베이스라인 표 — CLAUDE.md, README.md, docs/comparison/

**Document (not code)**
- MuSiQue R@5 0.453 vs HippoRAG2 0.747 격차를 **정직한 "known gap"** 섹션으로 README.md/docs 에 포함. "OpenIE triple pipeline 없이는 -0.294 구조적" 명시
- 3-round ablation (decomposer, inline phrase, DF-filtered entity linker) 을 `docs/CONCEPTS.md` "measured negatives" 섹션에 기록 — 향후 session 이 같은 실수 반복 방지

**Defer to v0.18.0+**
- OpenIE triple extraction + triple-level embedding index + query-to-triple dense linking (HippoRAG2 본질 구현)
- EdgeKind 확장 + PPR `_EDGE_TYPE_WEIGHTS` (W-9 — triple 없으면 혜택 없음)

### 9.4 남은 작업

| ID | 작업 | 예상 |
|---|---|---:|
| R1-1 | KRRA/assort/X2BEE SQLite graph 를 H100 으로 전송 (scp) 또는 source 로 재인제스트 | 0.5d |
| R1-2 | 9개 custom 벤치 재측정 → 완전한 14-dataset baseline 확보 | 0.5d |
| R1-3 | CLAUDE.md 베이스라인 표 재작성 (FTS-only / Full pipeline 두 컬럼 병기) | 0.5d |
| R1-4 | README.md 공개 벤치 섹션 업데이트 + MuSiQue gap 정직 공개 | 0.5d |
| R1-5 | `CONCEPTS.md` "measured negatives" 섹션 추가 | 0.5d |
| R1-6 | `docs/comparison/synaptic_results.md` 에 "full pipeline Round 1" 섹션 | 0.5d |
| R1-7 | v0.17.0 release note 초안 | 0.5d |

**합계 ~3.5일**. 구현 코드는 이미 ship 단계, 주로 문서화 + 측정 완료.

---

## 8. 문서 이력

- 2026-04-18: 초안 작성. 4개 서브에이전트 교차검증 결과 반영. P0 실행 전 상태.
- 2026-04-18 (addendum): P0 Stage 1 (MuSiQue 500q, bge-m3 + bge-reranker-v2-m3) 완료. R@5 0.453 측정 → **Case B 확정**. §4.5 / §6 재작성. Stage 2/3 은 vLLM 공존 VRAM 부족으로 이월.
- 2026-04-18 (저녁): W-1~W-6 구현 완료 (커밋 `9e9dcf9` + `2eb2b3b`). Rule decomposer 가 MuSiQue 영어 chain 쿼리에 부적합 (4% split rate, 오분해)이라 W-5 (LLMChainDecomposer + Qwen3.5-27B) 를 W-7 이전으로 이동. W-7 측정 실행 중.
- 2026-04-18 (22:04): **W-7 negative** — LLMChainDecomposer MuSiQue 500q 측정 완료. R@5 0.453 → 0.405 (−10.6%), search 476s → 1820s. 원인: RRF fusion 이 서브쿼리 노이즈 seed 를 과대평가, reranker 는 원본만 봄. **방향 재설정**: W-8 을 typed relation sweep 에서 **query-to-phrase linking** (HippoRAG2 +12.5%p 진짜 기여분) 으로 재스코프.
- 2026-04-18 (22:59): **W-8a / W-8b 연속 negative** — inline `EnglishPhraseExtractor` (R@5 0.423, −6.6%, build 15.5× 느림) 및 post-hoc DF-filtered `EntityLinker` (R@5 0.435, −4%, build 1.6× 느림) 둘 다 baseline 못 이김. 결론: **MuSiQue 는 mechanism 추가 만으로는 절대 못 따라잡는 벤치.** HippoRAG2 추격은 **architecture-level 교체** (OpenIE triple + NV-Embed-v2 triple matching) 필요, v0.18.0+ 연구 트랙으로 분리.
- 2026-04-18 (23:30): **Round 1 완료 — 전략 회전**. 공개 벤치 5개 (HotPotQA-24 / Allganize RAG-ko 200q / RAG-Eval 300q / PublicHealthQA 77q / AutoRAG 720q) 전부 `--local-bge + --entity-linker` 로 재측정. 평균 MRR **+54% uplift** 보고. ~~v0.17.0 은 이 데이터로 리포지셔닝~~ (§9.1 참조 — 이 내러티브 다음 날 정정됨).
- 2026-04-19 (자정): **R1 내러티브 오판 발견 + 삼각검증**. FTS-only 베이스를 v0.14.4 에서 끌어온 실수. v0.16.0 엔진 flip 이후 FTS-only 는 이미 평균 0.837. 재측정으로 3-round 매트릭스 구축 (R1b FTS-only / R1a bge only / R1 bge+EL). **AutoRAG 에서 reranker 가 단독 −29% regression** 발견. 원인 격리 (`diagnose_autorag.py`) → cross-encoder blend weight 0.4 가 retrieval-style corpus 에서 과도함. Sweep (`sweep_rerank_blend.py`) 후 default **0.4 → 0.1** 변경. 5 벤치 평균 MRR 0.849 → 0.881 (+3.2%p), AutoRAG 0.642 → 0.766 회복. §9.3 참조.
