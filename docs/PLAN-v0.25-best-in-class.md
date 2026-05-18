# PLAN v0.25 — 양호 → 최고

상태: draft · 목표: "여러 개 양호"를 "하나 압도적 + 외부 증거"로 전환.

## 0. 전제 — 정직한 현 위치

v0.24 에서 측정된 사실:
- multi-hop: vanilla RAG 0% vs synaptic 83% (finreg). **진짜 우위.**
- single-hop: RAG와 동률. 우위 없음.
- MuSiQue(영어 위키 multi-hop): R@5 0.45 vs HippoRAG2 0.75. **지고 있음.**

→ synaptic 의 우위는 **교차참조 구조가 있는 문서**(법령·규정·내규·기술표준·
계약)에 한정된다. 위키형 multi-hop 에는 우위가 없다. "최고"는 전방위 1등이
아니라 — **이 레인에서 명백한 1등 + 재현 가능한 외부 증거**다 (Path A).

핵심 긴장: synaptic 의 정체성("LLM-free indexing, relation-free graph")은
*비용 포지션*이다. 전방위 최고 GraphRAG 는 typed 엔티티 관계가 필요하고
그건 index-time LLM 을 부른다. v0.25 는 비용 포지션을 깨지 않고 — 레인을
좁혀 그 안에서 최고가 된다.

## 1. WS-2 — 경쟁자 head-to-head ✅ 완료

**측정 완료 (HippoRAG2, 2026-05-18)**:

| 시스템 | finreg multi-hop (120q) |
|---|---:|
| vanilla RAG | 0% |
| HippoRAG2 (NeurIPS'24 graph+PPR) | 25% |
| synaptic-memory | 83% |

- HippoRAG2 를 동일 finreg corpus(4,417 조문)·동일 120 GT·strict 채점으로
  측정. OpenIE LLM 은 동일 vLLM Qwen3.6-27B. 공정성 위해 한국어 임베더
  (bge-m3)를 어댑터로 주입(HippoRAG 기본은 영어 임베더뿐).
- HippoRAG2 는 LLM OpenIE 로 퍼지 엔티티 triple 을 추출 → "제30조" 같은
  정확한 상호참조를 깨끗한 엣지로 못 잡아 25%. synaptic 의 `REFERENCES`
  엣지는 인용을 정확히 1-hop 화 → 83%.
- **결론**: named 학술 경쟁자 대비 동일 조건 재현 가능 측정에서 83% vs
  25% — "교차참조 corpus 최고 GraphRAG" 가 외부 증거로 입증됨.
- 재현: `examples/benchmark_vs_competitors/finreg_hipporag.py`.
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

## 3. WS-0 — agent 루프 하드닝 (multi-hop 83 → 95)

WS-2 측정 전에 끌어올릴 수 있으면 입증이 더 강해진다. v0.24 미해결 20건:
agent-loop 7 / 능동추종 7 / 진입검색 6. 진입검색(A 미검색) 6건이 가장
근거 명확 — 진입 조문 검색 강화부터.

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

1. **WS-2** — 외부 증거. 신규 메커니즘 0, 최고 레버리지.
2. **WS-0** — 83→95, 입증 강화.
3. **WS-1** — 온톨로지 깊이 (answer quality).
4. §4 후속 트랙.
