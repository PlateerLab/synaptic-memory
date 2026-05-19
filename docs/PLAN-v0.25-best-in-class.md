# PLAN v0.25 — 양호 → 최고

상태: draft · 목표: "여러 개 양호"를 "하나 압도적 + 외부 증거"로 전환.

## 0. 전제 — 정직한 현 위치

측정된 사실 (finreg, 정정 GT, 2026-05-19):
- multi-hop: FTS RAG 0% / dense RAG 32% / HippoRAG2 31% / synaptic 73%.
  **가장 가까운 경쟁자의 2.3배 — 진짜 우위.**
- single-hop: RAG와 동률(94%). 우위 없음.
- MuSiQue(영어 위키 multi-hop): R@5 0.45 vs HippoRAG2 0.75. **지고 있음.**

→ synaptic 의 우위는 **교차참조 구조가 있는 문서**(법령·규정·내규·기술표준·
계약)에 한정된다. 위키형 multi-hop 에는 우위가 없다. "최고"는 전방위 1등이
아니라 — **이 레인에서 명백한 1등 + 재현 가능한 외부 증거**다 (Path A).

핵심 긴장: synaptic 의 정체성("LLM-free indexing, relation-free graph")은
*비용 포지션*이다. 전방위 최고 GraphRAG 는 typed 엔티티 관계가 필요하고
그건 index-time LLM 을 부른다. v0.25 는 비용 포지션을 깨지 않고 — 레인을
좁혀 그 안에서 최고가 된다.

## 1. WS-2 — 경쟁자 head-to-head ✅ 완료

**측정 완료 (HippoRAG2 + dense RAG, 정정 GT, 2026-05-19)**:

| 시스템 | finreg multi-hop (120q) |
|---|---:|
| RAG (FTS) | 0% |
| RAG (dense, bge-m3) | 32% |
| HippoRAG2 (NeurIPS'24 graph+PPR) | 31% |
| synaptic-memory (9B agent) | **73%** |

- HippoRAG2 를 동일 finreg corpus(4,417 조문)·동일 120 multi-hop GT·strict
  채점으로 측정. OpenIE LLM 은 vLLM Qwen3.6-27B. 공정성 위해 한국어 임베더
  (bge-m3)를 어댑터로 주입(HippoRAG 기본은 영어 임베더뿐).
- **핵심 발견 — HippoRAG2 ≈ dense RAG (31% vs 32%).** LLM 이 추출한 엔티티
  그래프는 평범한 벡터 RAG 대비 *이득이 없다* — "제30조" 같은 정확한
  상호참조는 엔티티가 아니라 구조라서 퍼지 triple 로 안 잡힌다. synaptic 의
  `REFERENCES` 엣지만이 인용을 정확히 1-hop 화 → 73% (인덱싱 LLM 비용 0).
- **결론**: named 학술 경쟁자 대비 동일 조건 재현 가능 측정에서 73% vs
  31% (2.3배) — "교차참조 corpus 최고 GraphRAG" 가 외부 증거로 입증됨.
  synaptic 은 *더 작은* 9B agent 로 측정 (HippoRAG2 는 27B OpenIE).
- 재현: `examples/benchmark_vs_competitors/{finreg_hipporag,finreg_dense_rag}.py`.
  결과: `docs/REPORT-rag-vs-synaptic.md` §헤드라인, §B.3.
- 미수행: GraphRAG / LightRAG (HippoRAG2 1종으로 핵심 입증 — 추가는 후속).

## 2. WS-1 — typed 구조 관계 (온톨로지 깊이)

현재 `StructuralReferenceLinker` 는 `REFERENCES` 한 종류만 만든다. 법률
온톨로지는 "인용"만이 아니다:
- **준용 (MUTATIS)** — "제5조를 준용한다": B 의 규정이 A 에 *적용*된다.
  단순 인용과 의미가 다르다.
- 별표(ANNEX), 위임(법률→시행령) 등은 §4 참조.

설계: 링커가 해소된 인용을 **connective 윈도우로 분류**한다 — 토큰 주변에
"준용" 이 있으면 MUTATIS, 아니면 일반 REFERENCES. 이것은 v0.23
ReferenceLinker(connective 타이핑)의 부활인데 — **clean-target corpus 위에서는
작동한다**(v0.23 실패 원인은 노이즈 타깃이지 메커니즘이 아니었다).

- 정직한 가치 평가: typed 관계는 *어느 문서를 검색하는가*를 안 바꾼다
  (A→B 는 종류 무관 연결). 따라서 **retrieval hit-rate 는 안 오른다.**
  가치는 답변 품질·설명가능성·온톨로지 충실도 — answer-quality judge 로
  측정하지 hit-rate 로 측정하지 않는다.
- 프로파일이 connective→type 맵을 주입. EdgeKind 또는 엣지 metadata.

## 3. WS-0 — agent 루프 하드닝 (multi-hop 끌어올리기)

진단 이력 (정직하게): WS-0 초기 전제는 "83→95" 였으나, 측정 격차(100→88)를
파고든 결과 **cross-scope 회귀가 아니라 GT 버그**였음이 드러났다 — multi-hop
GT 생성기가 `「○○법」`·`법 ` 수식어 붙은 인용을 같은 법령 조문으로 잘못
해소했다. 생성기를 수정하고 GT를 전면 재생성, 3종을 정정 GT로 재측정했다
(§1). 현재 synaptic 9B 측정 73%, 미해결 32건은 단일 근본 원인 없음.

남은 WS-0: (a) 27B agent 재측정으로 헤드라인 강화 여부 판단, (b) 진입 조문
미검색 케이스부터. 다만 73% vs 31% 격차가 이미 확보돼 우선순위는 낮다.

## 4. 후속 — 정직한 약점 트랙 (v0.26+)

- **별표(ANNEX)** — 별표 노드가 그래프에 없다. corpus 수집 확장 필요.
- **위임(법률↔시행령)** — "대통령령으로 정한다"는 대상 조문을 명시하지
  않는다. clean 해소 불가 → LLM 또는 topic-correspondence 필요.
- **MuSiQue / 위키형 multi-hop** — synaptic 이 지는 영역. 닫으려면
  opt-in LLM triple-extraction 티어(Path B)가 필요. zero-cost 기본은
  유지하되 "max quality" 티어를 추가. 가장 큰 작업.
- **정의(DEFINES)** — 용어→정의조항. 용어가 노드 식별자가 아니라 어렵다.

## 5. 측정 규율

- WS-2 는 경쟁자와 동일 corpus·GT·judge. 하베스 병렬 실행.
- WS-1 은 answer-quality judge (hit-rate 아님).
- Phase 종료 시 1회 측정. 효과 없으면 ship 안 함 (WS-D 선례).

## 우선순위

1. **WS-2** ✅ 완료 — 외부 증거 확보 (synaptic 73% vs HippoRAG2 31%).
2. **WS-1** — 온톨로지 깊이 (answer quality).
3. **WS-0** — multi-hop 추가 개선 (격차 확보돼 우선순위 낮음).
4. §4 후속 트랙.
