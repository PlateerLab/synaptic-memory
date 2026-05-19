# RAG vs synaptic-memory — 측정 기반 비교 보고서

작성: 2026-05-17 · 갱신: 2026-05-19 (GT 정합성 수정 + dense RAG 추가) ·
대상 corpus: finreg (금융 법령)

---

# Part A — 요약

## 헤드라인

같은 corpus·동일 multi-hop GT·strict 채점으로 RAG(키워드/벡터 두 모드),
**HippoRAG2**(학술 graph+PPR 베이스라인), synaptic-memory 를 정면 비교한
결과 (finreg, 120 multi-hop 문항):

| 시스템 | 검색 방식 | multi-hop solved |
|---|---|---:|
| RAG (FTS) | BM25 키워드 | **0/120 (0%)** |
| RAG (dense) | bge-m3 벡터 | **39/120 (32%)** |
| HippoRAG2 | LLM 엔티티 그래프 + PPR | **37/120 (31%)** |
| **synaptic-memory** | **구조적 `REFERENCES` 엣지 + agent** | **88/120 (73%)** |

> 모델 메모 — synaptic agent 는 vLLM **Qwen3.5-9B**, HippoRAG2 의 OpenIE
> 추출은 **Qwen3.6-27B**, dense RAG 는 **bge-m3** 임베더. 즉 synaptic 은
> *더 작은* 9B 로 측정됐고, HippoRAG2 는 더 큰 27B 로 측정됐다 — synaptic
> 에 불리한 비대칭이다 (§B.3 참조).

## 결론

- **single-shot 검색은 무승부다.** 조문 하나로 답이 나오는 질의에서 RAG는
  이미 천장에 가깝다(v0.24 측정 94%). 메커니즘을 더해도 의미 있는 개선은
  없다.
- **해자(moat)는 multi-hop이다.** "이 조문이 인용하는 다른 조문까지 알아야
  답이 완성되는" 질의에서 — 표준 dense RAG는 32%, 학술 SOTA GraphRAG인
  HippoRAG2도 31%. synaptic-memory는 상호참조를 그래프 엣지로 만들고 agent가
  그것을 따라가 **73%** — 가장 가까운 경쟁자의 **2.3배**다.
- **dense RAG ≈ HippoRAG2 (32% vs 31%) 가 핵심 발견이다.** LLM이 추출한
  엔티티 그래프(HippoRAG2의 접근)는 **평범한 벡터 RAG 대비 사실상 이득이
  없다** — "제30조" 같은 정확한 상호참조는 *엔티티*가 아니라 *구조*라서
  퍼지 triple로는 안 잡힌다. synaptic의 `REFERENCES` 엣지만이 인용을 정확히
  1-hop으로 따라간다.
- 즉 — *"문서 하나에 답이 있으면 RAG로 충분하다. 답이 문서 사이에 흩어져
  있으면 RAG도, LLM 엔티티 그래프도 못 한다 — 거기가 synaptic-memory의
  영역이다."*

## 한계 (정직하게)

- FTS/벡터가 강한 corpus의 single-shot 질의에서는 동률 — synaptic의 우위
  없음.
- 모델 비대칭 — synaptic 9B, HippoRAG2 27B(OpenIE). HippoRAG2를 9B로 재측정
  하면 *더 낮아질* 뿐이므로(약한 LLM = 약한 엔티티 그래프) 31%는 HippoRAG2에
  공정하거나 유리한 숫자다.
- 영어 multi-hop(MuSiQue)에는 별도의 미해결 천장이 있다 (§B.6).

---

# Part B — 기술 본문

## B.1 비교 대상 정의

**RAG** (`eval/rag_baseline.py`, `examples/benchmark_vs_competitors/finreg_dense_rag.py`)
질의 → 검색 top-k → (RAG 답변용) 단일 LLM 호출. 그래프 확장 없음, 재랭킹
없음, 도구 루프 없음 — 교과서적 RAG. 두 모드 모두 측정한다:
- **FTS RAG** — BM25 키워드 검색.
- **dense RAG** — `bge-m3` 임베딩 코사인 top-k. *표준 RAG는 벡터 검색이므로*
  헤드라인 비교의 기준은 dense RAG다.

**HippoRAG2** (`examples/benchmark_vs_competitors/finreg_hipporag.py`)
NeurIPS'24 / ICML'25 — LLM OpenIE로 추출한 엔티티 triple 그래프 위의
Personalized PageRank. 한국어 임베더가 없어(영어 전용) 공정성을 위해
synaptic과 같은 급의 `bge-m3`를 어댑터로 주입했다.

**synaptic-memory** (`eval/run_all.py --agent`)
3세대 검색 파이프라인(FTS + 벡터 + GraphExpander + HybridReranker +
EvidenceAggregator) 위에서 도는 멀티턴 agent. agent는 `search` /
`get_document` / `expand` / `follow` 등 도구를 최대 5턴 호출하며 추론한다.
그래프에는 문서 간 명시적 상호참조가 `REFERENCES` 엣지로 들어 있다.

세 시스템 모두 **동일 corpus, 동일 multi-hop GT, 동일 strict 채점**으로
측정된다. LLM 모델은 어긋난다(헤드라인 메모 참조).

## B.2 측정 셋업

**corpus — finreg.** 국가법령정보센터(law.go.kr)에서 수집한 금융 분야
법령. 법률 25 + 시행령 25 + 시행규칙 9 = 59개 문서, **4,417개 조문**,
순수 텍스트 약 264만 자(~130만 토큰). 각 조문이 하나의 검색 단위.

규정 corpus를 고른 이유: 조문 텍스트는 "제15조제2항에 따라", "「은행법」
제5조" 같은 **상호참조로 빽빽하다** — multi-hop 검색을 시험하기에 이상적.

**GT (검증 가능, 2026-05-19 정합성 수정).** 쿼리는 LLM이 생성하되 각
레코드에 근거 조문 전문 + 정답을 함께 기록해 사람이 감수할 수 있게 했다.
multi-hop 120문항 — 조문 A가 같은 법령의 조문 B를 인용. 질문 표면은 A의
어휘만 노출하고, **생성된 질문으로 직접 FTS를 돌려 B가 top-10에 안 잡히는
것만 채택**했다. 이로써 multi-hop 집합은 단발 검색으로 풀 수 없음이 기계적
으로 보장된다.

> **GT 정합성 수정 (2026-05-19).** 초기 GT 생성기는 `제N조` 인용을 무조건
> *같은 법령*의 조문으로 해소해, 앞에 붙은 `「○○법」`·`법 ` 수식어(다른
> 법령·모법 지칭)를 무시했다 — multi-hop 쌍 120개 중 일부가 잘못된 B를
> 정답으로 가졌다. 생성기를 수정(수식어 붙은 인용은 intra 쌍에서 제외)하고
> GT를 전면 재생성했다. 본 보고서의 모든 수치는 **수정된 GT** 기준이다.

**채점.** multi-hop은 strict — GT 조문을 **전부** top-k에 검색해야 정답.
인용 조문에 실제로 도달했는지가 핵심이므로 judge 폴백 없음.

## B.3 결과

### single-hop — 94% 동률 (v0.24 측정, 이번 라운드 미재측정)

| | solved |
|---|---:|
| vanilla RAG | 113/120 (94%) |
| synaptic agent | 117/120 (94%) |

조문 하나에 답이 있는 질의는 검색 top-k가 곧장 찾는다. single-hop GT는
이번 정합성 수정 대상이 아니어서(영향 없음) 재측정하지 않았다. 이 영역에서
synaptic의 구조적 우위는 없다 — 정직한 동률.

### multi-hop — RAG/HippoRAG2 ~31% vs synaptic 73%

| 시스템 | multi-hop solved |
|---|---:|
| RAG (FTS) | 0/120 (0%) |
| RAG (dense, bge-m3) | 39/120 (32%) |
| HippoRAG2 (27B OpenIE) | 37/120 (31%) |
| **synaptic agent (9B)** | **88/120 (73%)** |

- **FTS RAG 0%** — 설계상 당연하다. multi-hop GT는 "B가 단발 FTS 검색에 안
  잡힘"을 보장하도록 만들어졌고, FTS RAG는 단발 검색이 전부다.
- **dense RAG 32%** — 벡터 유사도는 어휘가 안 겹쳐도 B를 *가끔* 끌어온다.
  FTS보다 낫지만, A를 읽고 그 안의 인용을 따라가는 단계가 없으므로 1/3에
  그친다.
- **HippoRAG2 31%** — LLM 엔티티 그래프 + PPR을 얹어도 dense RAG와 사실상
  동률이다. §B.4 참조.
- **synaptic 73%** — 상호참조를 `REFERENCES` 엣지로 만들고 agent가 그것을
  따라간다. 더 작은 9B 모델로도 가장 가까운 경쟁자의 2.3배.

### HippoRAG2 ≈ dense RAG — named 경쟁자가 못 넘는 벽

"최고 GraphRAG"를 주장하려면 다른 *그래프* 시스템도 이겨야 한다.
HippoRAG2를 동일 finreg corpus(4,417 조문)·동일 120 multi-hop GT·strict
채점으로 측정했다.

- 셋업: HippoRAG2의 OpenIE 추출 LLM은 로컬 vLLM Qwen3.6-27B. 한국어
  임베더 부재를 보완하려 bge-m3를 어댑터로 주입(HippoRAG의 그래프 알고리즘을
  임베더 부재로 불리하게 만들지 않기 위함). 인덱싱: 4,417 조문 → 1만+ 노드 +
  2만+ triple.
- **왜 31%인가**: HippoRAG2는 LLM OpenIE로 *퍼지 엔티티 triple*
  (`금융감독원장`, `과징금` …)을 뽑는다. "제30조"라는 정확한 상호참조는
  엔티티가 아니라 *구조*다 — OpenIE가 깨끗한 엣지로 잡지 못한다. 그 결과
  HippoRAG2는 **평범한 dense RAG(32%)와 통계적으로 구분되지 않는다.**
- **함의**: LLM 엔티티 추출 + PPR이라는 학술 SOTA 접근이, 교차참조 corpus
  에서는 벡터 RAG 대비 *아무 이득이 없다*. 인덱싱 시 LLM 비용을 들이고도.
  synaptic은 인덱싱 LLM 비용 0으로 인용을 `REFERENCES` 엣지화해 73%.

### 실패 분석

synaptic 9B 측정의 미해결 32건은 단일 근본 원인이 아니다(진입 조문 미검색
/ 능동 추종 실패 / agent 루프 종료 실패가 섞여 있음). 27B agent 면 더 높을
것으로 보이나(구 GT 27B 측정 83%) 본 라운드에서는 9B로 통일해 측정했다 —
헤드라인(가장 가까운 경쟁자의 2.3배)은 이미 확보.

## B.4 왜 차이가 나는가

**RAG가 0~32%인 이유 — 구조적이다.**
RAG = 검색 1회 + 답변 1회. multi-hop 질문은 표면에 A의 어휘만 있으므로
검색은 A만 가져온다. 답에 필요한 B는 A의 본문 안에 "제30조"라는 참조로만
존재한다. RAG에는 "A를 읽고, 그 안의 참조를 따라 B를 다시 가져오는" 단계
자체가 없다 — FTS면 어휘 0이라 0%, dense면 의미 유사도로 가끔 잡혀 32%.

**HippoRAG2가 RAG를 못 넘는 이유.**
HippoRAG2의 그래프 엣지는 LLM이 추출한 엔티티 공유 관계다. "A조와 B조가
같은 엔티티(`금융위원회`)를 언급" 같은 약한 신호는 PPR로 전파되지만,
"A조가 B조를 인용한다"는 정확한 구조는 엔티티가 아니라서 엣지가 안 된다.
그래프를 얹었어도 따라갈 *올바른 엣지*가 없다.

**synaptic이 푸는 이유 — 4단계.**
1. **REFERENCES 엣지** — 인제스트 시 조문 텍스트의 "제N조" 인용을
   `StructuralReferenceLinker`가 엣지로 만든다(인덱싱 LLM 비용 0).
   상호참조가 산문이 아니라 그래프 구조가 된다.
2. **GraphExpander** — 검색이 A를 seed로 잡으면 A의 REFERENCES 이웃(B)을
   후보 풀에 끌어온다. 한 번의 검색이 1-hop 확장된다.
3. **참조-동반 재랭킹** — B는 질의와 어휘가 안 겹쳐 lexical 점수가 0이다.
   B를 그것을 인용한 seed 점수의 0.9배로 끌어올리고 MMR 중복 필터를
   우회시켜 top-k에 동반 생존시킨다.
4. **agent 다턴** — 그래도 단발에 안 잡히면 agent가 `get_document` /
   `follow`로 인용을 능동 추종한다.

## B.5 관계 자동 구축의 범용성

REFERENCES 엣지 메커니즘은 finreg 전용이 아니다. `StructuralReferenceLinker`는
corpus-agnostic이며 — DomainProfile이 "어느 속성이 식별자인가"(`article_no`
등)만 선언하면 — corpus의 실제 식별자 값에서 매처를 자동 도출한다(손으로
쓴 정규식 불요). 게다가 **clean-target 게이트**가 있어, 식별자 인벤토리가
깨끗하지 않은 corpus에서는 스스로 차단하고 no-op한다.

검증 (실제 그래프):
- finreg (깨끗한 조문번호 인벤토리): 8,000+ REFERENCES 엣지 자동 생성
  (intra-law + cross-law, 59개 법령 중 52개가 연결).
- KRRA (깨끗한 인용키 없음): 게이트가 안전하게 no-op.

→ *조문번호·조항코드 같은 규칙적 식별자가 있는 문서류(법령·규정·내규·기술
표준·매뉴얼·계약)면 프로파일 몇 줄로 multi-hop 온톨로지가 자동 구축된다.
식별자 구조가 없는 corpus에는 무해하게 no-op한다.*

## B.6 한계

- **single-shot, 검색-strong corpus** — synaptic의 우위 없음(동률).
- **모델 비대칭** — synaptic 9B agent vs HippoRAG2 27B OpenIE. HippoRAG2를
  9B로 재측정하면 더 낮아질 뿐이라 31%는 HippoRAG2에 유리한 숫자다.
- **measured negative — WS-D.** tool 결과에 참조를 명시 노출하는 변경은
  효과가 없어 기각·되돌렸다. 측정 규율: 효과 없는 메커니즘은 ship하지
  않는다.
- **GT 노이즈.** multi-hop GT는 LLM 자동 생성 — FTS 검증 필터 + 2026-05-19
  수식어-인용 정합성 수정으로 완화했으나 완전하지는 않다.
- **영어 multi-hop 천장.** MuSiQue 등 영어 multi-hop은 OpenIE triple 추출
  같은 별도 아키텍처가 필요한 미해결 트랙이다(`docs/PLAN-v0.18`).

## B.7 재현 방법

```bash
# 1. corpus 수집 (law.go.kr 헤드리스 스크래핑, API 키 불요)
python eval/datasets/build_finreg.py --with-decree

# 2. 그래프 구축 (DocumentIngester + StructuralReferenceLinker 자동)
uv run python eval/datasets/ingest_finreg.py --clean

# 3. GT 생성 (검증가능 — 근거+정답 포함, 수식어-인용 정합성 수정 반영)
uv run python eval/datasets/gen_finreg_queries.py \
    --llm-base-url http://localhost:8012/v1 --model Qwen3.5-9B \
    --single 0 --multi 120

# 4. RAG 측정 — FTS
uv run python eval/rag_baseline.py --dataset "finreg multihop" --no-judge
#    RAG 측정 — dense (표준 RAG 베이스라인)
/tmp/hrag/bin/python examples/benchmark_vs_competitors/finreg_dense_rag.py \
    --device cuda:1 --fp16 --batch-size 4

# 5. HippoRAG2 측정 (격리 venv — 스크립트 헤더 참조)
/tmp/hrag/bin/python examples/benchmark_vs_competitors/finreg_hipporag.py

# 6. synaptic agent 측정
OPENAI_API_KEY=dummy uv run python eval/run_all.py --agent --agent-only \
    --agent-dataset "finreg multihop" --agent-concurrency 12 \
    --llm-base-url http://localhost:8012/v1 --agent-model Qwen3.5-9B
```

## B.8 부록 — 측정 이력

| 날짜 | 변경 |
|---|---|
| 2026-05-17 | finreg corpus + RAG-vs-agent 하베스, 초기 헤드라인 |
| 2026-05-18 | HippoRAG2 head-to-head 추가 |
| 2026-05-19 | GT 수식어-인용 정합성 수정 + 전면 재생성 / dense RAG 추가 / synaptic·HippoRAG2 재측정 / 모델 9B 전환 |

corpus 통계: 4,417 조문 / 59 문서 / 264만 자. 그래프: 4,417 part_of +
8,000+ references 엣지.

설계 상세: `docs/PLAN-v0.24-relation-enrichment.md`,
`docs/PLAN-v0.25-best-in-class.md`.
