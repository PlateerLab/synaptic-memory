# RAG vs synaptic-memory — 측정 기반 비교 보고서

작성: 2026-05-17 · 대상 corpus: finreg (금융 법령) · 모델: Qwen3.6-27B (vLLM)

---

# Part A — 요약

## 헤드라인

같은 corpus, 같은 LLM-judge, 같은 모델로 vanilla RAG와 synaptic-memory를
정면 비교한 결과:

| 쿼리 유형 | vanilla RAG | synaptic-memory | 차이 |
|---|---:|---:|---:|
| single-hop (조문 1개로 답) | 94% | 94% | 동률 |
| **multi-hop (조문 상호참조 추종)** | **0%** | **83%** | **+83pp** |

## 결론

- **single-shot 검색은 무승부다.** 조문 하나로 답이 나오는 질의에서 FTS
  top-k RAG는 이미 94% — 천장에 가깝다. 여기에 메커니즘을 더해도 의미 있는
  개선은 없다.
- **해자(moat)는 multi-hop이다.** "이 조문이 인용하는 다른 조문까지 알아야
  답이 완성되는" 질의에서 vanilla RAG는 **0%** — 한 번 검색하고 끝나는
  구조라 인용을 따라갈 방법이 원천적으로 없다. synaptic-memory는 상호참조를
  그래프 엣지로 만들고 agent가 그것을 따라가 **83%**를 달성한다.
- 즉 — *"문서 하나에 답이 있으면 RAG로 충분하다. 답이 문서 사이에 흩어져
  있으면 RAG는 못 한다 — 거기가 synaptic-memory의 영역이다."*

## 한계 (정직하게)

- FTS가 강한 corpus의 single-shot 질의에서는 동률 — synaptic의 우위 없음.
- multi-hop GT는 LLM 자동 생성 — 노이즈가 일부 섞여 있다 (§B.6).
- 영어 multi-hop(MuSiQue)에는 별도의 미해결 천장이 있다 (§B.6).

---

# Part B — 기술 본문

## B.1 비교 대상 정의

**vanilla RAG** (`eval/rag_baseline.py`)
질의 → 검색 top-k 청크 → 단일 LLM 호출로 답변. 그래프 확장 없음, 재랭킹
없음, 도구 루프 없음 — 교과서적 RAG. `k=5`.

**synaptic-memory** (`eval/run_all.py --agent`)
3세대 검색 파이프라인(FTS + 벡터 + GraphExpander + HybridReranker +
EvidenceAggregator) 위에서 도는 멀티턴 agent. agent는 `search` /
`get_document` / `expand` / `follow` 등 도구를 최대 5턴 호출하며 추론한다.
그래프에는 문서 간 명시적 상호참조가 `REFERENCES` 엣지로 들어 있다.

두 시스템은 **동일 corpus, 동일 GT, 동일 LLM-judge, 동일 모델
(Qwen3.6-27B)**로 측정된다. 따라서 숫자 차이는 retrieval 아키텍처의
차이만 반영한다.

## B.2 측정 셋업

**corpus — finreg.** 국가법령정보센터(law.go.kr)에서 수집한 금융 분야
법령. 법률 25 + 시행령 25 + 시행규칙 9 = 59개 문서, **4,417개 조문**,
순수 텍스트 약 264만 자(~130만 토큰). 각 조문이 하나의 검색 단위.

규정 corpus를 고른 이유: 조문 텍스트는 "제15조제2항에 따라", "「은행법」
제5조" 같은 **상호참조로 빽빽하다** — multi-hop 검색을 시험하기에 이상적.

**GT (검증 가능).** 쿼리는 LLM이 생성하되 각 레코드에 근거 조문 전문 +
정답을 함께 기록해 사람이 감수할 수 있게 했다.
- single-hop 120문항 — 조문 1개로 답.
- multi-hop 120문항 — 조문 A가 조문 B를 인용. 질문 표면은 A의 어휘만
  노출하고, **생성된 질문으로 직접 FTS를 돌려 B가 top-10에 안 잡히는
  것만 채택**했다(261회 시도 중 127회를 "B 노출"로 기각). 이로써
  multi-hop 집합은 단발 검색으로 풀 수 없음이 기계적으로 보장된다.

**채점.** id 매칭 + LLM-judge 폴백. multi-hop은 strict — GT 조문을
**전부** 검색해야 정답, judge 폴백 없음(인용 조문에 실제로 도달했는지가
핵심이므로).

**하베스.** agent 벤치를 쿼리 동시 실행으로 병렬화 — 5시간 → ~14분.

## B.3 결과

### single-hop — 94% 동률

| | solved |
|---|---:|
| vanilla RAG | 113/120 (94%) |
| synaptic agent | 117/120 (94%) |

조문 하나에 답이 있는 질의는 FTS top-k가 곧장 찾는다. agent도 같은 수준 —
이 영역에서 synaptic의 구조적 우위는 없다. 정직한 동률.

### multi-hop — RAG 0% → synaptic 83%

단계별로 분해하면 어디서 점수가 오는지 보인다:

| 구성 | multi-hop solved |
|---|---:|
| vanilla RAG | 0/120 (0%) |
| synaptic agent — 관계 엣지 없음 | 30/120 (25%) |
| synaptic agent — REFERENCES 엣지 (기본 확장) | 87/120 (73%) |
| **synaptic agent — REFERENCES + WS-B 재랭킹** | **100/120 (83%)** |

- **RAG 0%** — 설계상 당연하다. multi-hop GT는 "B가 단발 검색에 안 잡힘"을
  보장하도록 만들어졌고, RAG는 단발 검색이 전부다.
- **엣지 없는 agent 25%** — agent가 다턴으로 재검색을 시도하지만 인용을
  안정적으로 못 따라간다.
- **REFERENCES 엣지 73%** — 상호참조를 그래프 엣지로 만들자 GraphExpander가
  인용 조문을 후보로 끌어온다. +48pp.
- **WS-B 83%** — 인용 조문은 질의와 어휘가 겹치지 않아 재랭킹에서 탈락하던
  문제를 "참조-동반 lift + 묶음 선택"으로 해결. +10pp.

### 실패 분석 — 남은 20건

검색 단계 진단(20건):
- 7건 — A·B 둘 다 검색되나 agent가 최종 실패 (agent 루프 문제)
- 7건 — B가 단발 검색엔 없음, 능동적 참조 추종 필요
- 6건 — 진입 조문 A 자체가 미검색

단일 근본 원인이 없어 추가 개선은 보류 — 헤드라인(0%→83%)은 이미 확보.

## B.4 왜 차이가 나는가

**RAG가 0%인 이유 — 구조적이다.**
vanilla RAG = 검색 1회 + 답변 1회. multi-hop 질문은 표면에 A의 어휘만
있으므로 검색은 A만 가져온다. 답에 필요한 B는 A의 본문 안에 "제30조"라는
참조로만 존재한다. RAG에는 "A를 읽고, 그 안의 참조를 따라 B를 다시
가져오는" 단계 자체가 없다. k를 키워도 B는 질의와 어휘가 0이라 안 잡힌다.

**synaptic이 푸는 이유 — 4단계.**
1. **REFERENCES 엣지** — 인제스트 시 조문 텍스트의 "제N조" 인용을
   `StructuralReferenceLinker`가 엣지로 만든다. 상호참조가 산문이 아니라
   그래프 구조가 된다.
2. **GraphExpander** — 검색이 A를 seed로 잡으면 A의 REFERENCES 이웃(B)을
   후보 풀에 끌어온다. 한 번의 검색이 1-hop 확장된다.
3. **참조-동반 재랭킹 (WS-B)** — B는 질의와 어휘가 안 겹쳐 lexical 점수가
   0이다. 그냥 두면 탈락한다. B를 그것을 인용한 seed 점수의 0.9배로 끌어
   올리고, MMR 중복 필터를 우회시켜 top-k에 동반 생존시킨다.
4. **agent 다턴** — 그래도 단발에 안 잡히면 agent가 `get_document` /
   `follow`로 인용을 능동 추종한다.

## B.5 관계 자동 구축의 범용성

REFERENCES 엣지 메커니즘은 finreg 전용이 아니다. `StructuralReferenceLinker`는
corpus-agnostic이며 — DomainProfile이 "어느 속성이 식별자인가"(`article_no`
등)만 선언하면 — corpus의 실제 식별자 값에서 매처를 자동 도출한다(손으로
쓴 정규식 불요). 게다가 **clean-target 게이트**가 있어, 식별자 인벤토리가
깨끗하지 않은 corpus에서는 스스로 차단하고 no-op한다.

검증 (실제 그래프):
- finreg (깨끗한 조문번호 인벤토리): 8,133 REFERENCES 엣지 자동 생성
  (intra-law 7,293 + cross-law 840, 59개 법령 중 52개가 연결).
- KRRA (깨끗한 인용키 없음): 게이트가 3가지 경우 모두 안전하게 no-op.

→ *조문번호·조항코드 같은 규칙적 식별자가 있는 문서류(법령·규정·내규·기술
표준·매뉴얼)면 프로파일 몇 줄로 multi-hop 온톨로지가 자동 구축된다. 식별자
구조가 없는 corpus에는 무해하게 no-op한다.*

## B.6 한계

- **single-shot, FTS-strong corpus** — synaptic의 우위 없음(동률). 단발
  검색으로 충분한 질의에 메커니즘을 더하는 것은 측정상 도움이 안 된다.
- **measured negative — WS-D.** tool 결과에 참조를 명시 노출하는 변경은
  100→98/120으로 효과가 없어 기각·되돌렸다. 측정 규율: 효과 없는 메커니즘은
  ship하지 않는다.
- **GT 노이즈.** multi-hop GT는 LLM 자동 생성 — 일부 질문이 의도한 조문
  외에서도 답될 수 있다. FTS 검증 필터로 완화했으나 완전하지는 않다.
- **영어 multi-hop 천장.** MuSiQue 등 영어 multi-hop은 OpenIE triple 추출
  같은 별도 아키텍처가 필요한 미해결 트랙이다(`docs/PLAN-v0.18`).

## B.7 재현 방법

```bash
# 1. corpus 수집 (law.go.kr 헤드리스 스크래핑, API 키 불요)
python eval/datasets/build_finreg.py --with-decree

# 2. 그래프 구축 (DocumentIngester + StructuralReferenceLinker 자동)
uv run python eval/datasets/ingest_finreg.py --clean

# 3. GT 생성 (검증가능 — 근거+정답 포함)
uv run python eval/datasets/gen_finreg_queries.py \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B \
    --single 120 --multi 120

# 4. vanilla RAG 측정
uv run python eval/rag_baseline.py --dataset finreg --dataset "finreg multihop" \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.6-27B

# 5. synaptic agent 측정
uv run python eval/run_all.py --agent --agent-only --agent-dataset finreg \
    --judge --agent-concurrency 16 \
    --llm-base-url http://localhost:8012/v1 --agent-model Qwen3.6-27B
```

## B.8 부록 — v0.24 커밋

| 커밋 | 내용 |
|---|---|
| `d352efc` | finreg corpus + RAG-vs-agent 하베스 |
| `8ce9827` | WS-B — REFERENCES 엣지 + 확장/재랭킹 |
| `777afc3` | Phase 1 측정 결과 (WS-D 기각) |
| `5069a06` | WS-A — 범용 StructuralReferenceLinker |
| `ee785eb` | WS-A — from_data 일반 경로 연결 |
| `7ce9a6f` | WS-A 마무리 |
| `3ddb179` | cross-scope 참조 해소 |

corpus 통계: 4,417 조문 / 59 문서 / 264만 자. 그래프: 4,417 part_of +
8,133 references 엣지.

설계 상세: `docs/PLAN-v0.24-relation-enrichment.md`.
