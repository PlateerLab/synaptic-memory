# Synaptic Memory vs. Existing Agent Memory Systems

> 기존 시스템들은 **"저장하고 검색하는 데이터베이스"**에 머물러 있다.
> Synaptic Memory는 **"경험하고, 학습하고, 잊고, 구조화하는 뇌"**이자,
> **아무 데이터를 넣으면 그래프를 자동 구축해 LLM이 탐색하는 범용 기질**이다.

---

## 1. 현황: 에이전트 메모리는 지금 어디에 있는가

2024-2025년, LLM 에이전트 메모리 분야가 폭발적으로 성장했다. NeurIPS, ICLR에서 다수의 논문이 발표되었고, Mem0, Zep, Letta 같은 프로덕션 시스템도 등장했다.

하지만 2026년 초 주요 서베이들이 공통으로 지적하는 문제가 있다:

> "how to consolidate without catastrophic loss, how to retrieve by cause rather than similarity, how to reflect without entrenching errors, and how to forget safely"
> — [Memory for Autonomous LLM Agents (2026)](https://arxiv.org/html/2603.07670)

저장과 검색은 발전했지만, **학습, 망각, 구조화**는 아직 미해결이다. 그리고 하나 더 — 대부분의 시스템이 **"에이전트가 대화하면서 축적한 메모리"** 만 다루고, **임의의 기업 데이터(CSV, PDF, DB 덤프)**를 그래프로 흡수해 같은 검색 파이프라인에서 다루는 통합된 기질(substrate)은 드물다.

---

## 2. 주요 시스템 개요

| 시스템 | 출처 | 핵심 아이디어 |
|--------|------|-------------|
| [HippoRAG](https://arxiv.org/abs/2405.14831) | NeurIPS 2024 | 해마 이론 기반 KG + Personalized PageRank spreading activation |
| [Zep/Graphiti](https://arxiv.org/abs/2501.13956) | 2025 | Temporal KG — 시간 기반 유효성 추적, bi-temporal model |
| [Mem0](https://arxiv.org/abs/2504.19413) | 2025 | Extract→Update 파이프라인 + 그래프 변형(Mem0g) |
| [A-MEM](https://arxiv.org/abs/2502.12110) | NeurIPS 2025 | Zettelkasten 방식 자기조직화, 동적 인덱싱/링킹 |
| [MAGMA](https://arxiv.org/abs/2601.03236) | 2026 | 4개 직교 그래프 (semantic/temporal/causal/entity) + intent-aware retrieval |
| [Letta/MemGPT](https://github.com/letta-ai/letta) | 2023- | OS 메타포 — core/archival/recall 3계층 메모리 관리 |

---

## 3. 여섯 가지 구조적 단점

### 3-1. 학습하지 않는다 (No Behavioral Learning)

거의 모든 시스템이 "저장 → 검색"에 그친다. 사용 패턴으로부터 자동으로 배우는 메커니즘이 없다.

**HippoRAG**: KG가 offline에서 구축되면 고정. 같은 지식을 100번 참조해도 1번 참조한 것과 가중치가 동일. [논문](https://arxiv.org/html/2405.14831v1)에서 "The KG is static offline; online, only the personalization vector for PPR changes"라고 명시.

**Zep/Graphiti**: temporal 유효성(언제 생겼고 언제 만료되는지)만 추적. "이 지식이 얼마나 유용했는지"는 기록하지 않는다. 성공/실패 피드백 루프가 없다.

**Mem0**: LLM에게 "이 메모리를 업데이트해야 하나?"를 매번 판단시킨다. conflict detection과 update resolve가 전부 LLM 호출. 비용이 비싸고 (메모리 1개 처리에 [1350.9초](https://www.alphaxiv.org/overview/2601.02553v1)), LLM의 판단 정확도에 전적으로 의존한다.

**A-MEM**: 메모리를 추가할 때 LLM이 note를 생성하고, 태깅하고, 기존 note와 링킹한다. "agentic"하지만 매 메모리마다 여러 번의 LLM 호출이 필요. [NeurIPS 리뷰](https://openreview.net/forum?id=FiM0M8gcct)에서도 비용 문제가 지적됨.

**MAGMA**: causal graph로 인과관계를 추적하지만, 이 인과 추론 자체가 LLM에 의존한다. [논문](https://arxiv.org/html/2601.03236v1)이 "susceptible to extraction errors and hallucinations"이라고 인정.

#### Synaptic Memory의 접근

Hebbian learning은 LLM 호출 없이 작동한다:

```python
await graph.reinforce([decision_id, outcome_id], success=True)
# → decision.success_count += 1
# → edge(decision ↔ outcome).weight += 0.1

await graph.reinforce([bad_decision_id, failure_id], success=False)
# → edge weight -= 0.15 (실패는 더 강하게 학습)
```

순수 Python 연산. 비용 0, 지연 ~0ms. 에이전트가 결과만 보고하면 그래프가 스스로 조정된다.

---

### 3-2. 잊지 못한다 (No Principled Forgetting)

> "Memory management is a critical yet underexplored aspect... most systems lack principled mechanisms for memory decay, consolidation, or selective forgetting."
> — [Memory in the Age of AI Agents (2025)](https://arxiv.org/abs/2512.13564)

**HippoRAG**: 노드/엣지가 무한 증가. KG가 커질수록 PageRank 계산 비용이 선형 이상으로 증가. [후속 연구](https://arxiv.org/abs/2602.01965)에서 "high-degree hub nodes"로의 semantic drift 문제가 보고됨.

**Zep**: temporal invalidation으로 시간이 지난 관계를 무효화하지만, "자주 참조되는 오래된 지식"과 "안 쓰이는 최근 지식"을 구분하지 못한다. 시간만 보고 유용성은 안 본다.

**Mem0**: [논문](https://arxiv.org/abs/2504.19413)이 직접 인정 — "consolidation is not fully automated; duplicate and semantically similar memories may accumulate, and staleness and conflicting memories are only weakly managed by recency or manual LRU rules."

**A-MEM**: Zettelkasten 방법론의 근본적 한계 — 본래 "버리지 않는" 시스템. 메모리가 무한히 쌓이는 구조.

**Letta/MemGPT**: 오래된 대화를 요약해서 archival로 이관. 하지만 요약 시 정보 손실이 불가피하고, [context limit 초과 에러](https://github.com/letta-ai/letta/issues/957)가 빈번히 보고됨.

#### Synaptic Memory의 접근

뇌의 수면 중 기억 통합을 모방한 4단계 consolidation:

```
L0 (Raw, 72h TTL)
  → 72시간 내 3회 이상 접근하면 L1 승격
  → 안 쓰이면 삭제

L1 (Sprint, 90d TTL)
  → 10회 이상 접근하면 L2 승격

L2 (Monthly, 365d TTL)
  → 성공 10회 + 성공률 80% 이상이면 L3 승격

L3 (Permanent)
  → 검증된 핵심 지식. 영구 보존.
```

추가로 `decay_vitality(factor=0.95)`로 전체 노드의 vitality를 주기적으로 감쇠하고, `prune_edges(weight_below=0.1)`로 약해진 연결을 정리한다. 쓰이는 지식만 살아남는 자연선택.

---

### 3-3. 검색이 의도를 모른다 (Intent-Blind Retrieval)

같은 "배포"라는 쿼리라도:
- "배포할 때 지켜야 할 규칙이 뭐지?" → RULE 노드가 필요
- "배포가 실패한 적 있나?" → failure_count > 0인 OUTCOME + 원인 DECISION이 필요
- "이 배포 결정의 결과가 어땠지?" → decision→outcome→lesson 체인이 필요

대부분의 시스템이 이 구분을 못 한다.

**HippoRAG**: Personalized PageRank의 transition probability가 인덱싱 시점에 고정. [후속 연구](https://arxiv.org/abs/2602.01965)가 "Static Graph Fallacy"라고 명명 — 쿼리의 의도에 관계없이 같은 경로로 activation이 퍼진다. hub 노드로 drift하는 문제.

**Zep**: semantic + keyword + graph 3중 검색. 나쁘진 않지만, "왜 검색하는지"에 따라 전략을 바꾸지는 않는다.

**Mem0**: vector similarity 기반 검색 → LLM rerank. 관계 구조를 활용한 검색이 아니라 텍스트 유사도 + 후처리. [Mem0g 논문](https://arxiv.org/abs/2504.19413)에서 "fragmentation of multi-evidence cases: retrieval may omit critical context spread across disparate graph nodes" 문제 인정.

**MAGMA**: 4개 그래프에서 intent에 따라 다른 view를 선택하는 구조. 가장 진보적이지만, 4개 그래프를 동시에 유지·동기화하는 복잡도가 크다. [논문](https://arxiv.org/html/2601.03236v1)이 "additional storage and engineering complexity"를 인정.

#### Synaptic Memory의 접근

`intent` 파라미터 하나로 6가지 검색 전략을 전환:

```python
# 같은 쿼리, 다른 intent, 다른 전략
await graph.agent_search("배포", intent="past_failures")
# → failure_count > 0 필터 → 원인 Decision 역추적 → Lesson 수집

await graph.agent_search("배포", intent="related_rules")
# → RULE/LESSON 타입 필터 → 그래프 이웃 확장

await graph.agent_search("배포", intent="reasoning_chain")
# → Decision → Outcome → Lesson multi-hop 순회
```

Intent는 `similar_decisions | past_failures | related_rules | reasoning_chain | context_explore | general` 6종. 각각 5축 resonance 가중치(relevance / importance / recency / vitality / context)와 spreading activation 감쇠율이 다르다. `past_failures`는 importance(성공률)에 0.25를, `context_explore`는 spread factor에 0.75를 준다. 하나의 그래프에서 쿼리 시점에 전략만 바꾸므로 MAGMA 같은 multi-graph 유지 비용이 없다.

---

### 3-4. LLM에 과도하게 의존한다 (LLM-Heavy Indexing & Operations)

메모리의 핵심 연산(추출, 업데이트, 조직화, 충돌 해결)에 LLM을 사용하면 세 가지 문제가 생긴다:

1. **비용**: 문서 1개 인덱싱에 수천~수만 토큰 소모
2. **지연**: 실시간 에이전트 루프에서 병목, 초기 구축에 며칠
3. **hallucination 전파**: LLM이 잘못 추출한 관계가 메모리에 영구 저장

| 시스템 | LLM 호출 시점 | 대표적 문제 |
|--------|-------------|-----------|
| **Mem0** | 추출 + conflict detection + update | 메모리 1개 처리에 1350.9초 ([SimpleMem 논문](https://www.alphaxiv.org/overview/2601.02553v1) 비교) |
| **A-MEM** | note 생성 + 태깅 + 링킹 + 재조직 | 메모리 추가마다 3-5회 LLM 호출 |
| **Zep/Graphiti** | entity extraction + relation generation + conflict resolution | structured output 필수. 작은 모델에서 [스키마 오류 빈발](https://github.com/getzep/graphiti) |
| **MAGMA** | causal inference + entity extraction | "susceptible to extraction errors and hallucinations" ([논문](https://arxiv.org/html/2601.03236v1)) |
| **Letta/MemGPT** | 메모리 관리 자체가 LLM 함수 호출 | context limit 초과 시 요약도 LLM → 이중 비용 |
| **HippoRAG** | OpenIE triple 추출 (NER + RE) | 인덱싱 시 문서당 수회 LLM 호출, domain tuning 요구 |

#### Synaptic Memory의 접근 — LLM-free indexing

**인덱싱 단계에서 LLM 호출 0회**가 목표다. v0.12의 EvidenceSearch는 "relation-free graph" 설계로, 엣지를 LLM이 추출한 (s, r, o) 트리플이 아니라 **구조적·통계적 신호**만으로 만든다:

| 연산 | 구현 | LLM 필요 |
|------|------|---------|
| Category ↔ Document ↔ Chunk 계층 | 파일 시스템/DB 스키마 그대로 반영 | ✗ |
| Chunk-next 엣지 | 슬라이딩 윈도우 위치 | ✗ |
| MENTIONS 엣지 | DF 필터 phrase hub (entity linker) | ✗ |
| BM25 lexical seed | SQLite FTS5 + Kiwi 형태소(한국어) | ✗ |
| Vector seed | usearch HNSW (BYO embedder) | ✗ |
| PRF (Pseudo-Relevance Feedback) | top-k 임베딩 평균 → 2차 검색 | ✗ |
| PPR graph discovery | Personalized PageRank BFS | ✗ |
| Hybrid reranking | lexical + semantic + graph + structural + authority + temporal + MaxP | ✗ |
| Cross-encoder reranker | BYO protocol (TEI bge-reranker-v2-m3 등) | ✗* |
| Hebbian learning | `weight += 0.1` / `weight -= 0.15` | ✗ |
| Memory consolidation | access_count, success_rate 기반 규칙 | ✗ |
| Ontology validation | 타입 계층 순회 + 속성 검사 | ✗ |
| Intent-based search | intent별 전략 dispatch + 필터 | ✗ |

*Cross-encoder는 로컬 모델. LLM API 호출 아님.

LLM이 관여하는 지점은 **사용 시점**뿐이다:
- 에이전트가 `search` / `deep_search` / `filter_nodes` 등 MCP 도구를 호출
- LLM이 결과를 읽고 판단·합성

즉, 지식의 **관리**는 규칙 기반이고, 지식의 **사용**만 LLM이 한다. 이 분리가 비용·지연·hallucination 문제를 동시에 해결한다.

또한 임베더/리랭커는 **torch 의존성 0** 원칙으로 BYO(Bring-Your-Own)다. Ollama, TEI, OpenAI API 등 HTTP endpoint만 있으면 주입 가능. 덕분에 `pip install synaptic-memory` 코어가 수 MB에 머무른다.

---

### 3-5. 지식에 구조가 없다 (No Schema / Ontology)

"이 Decision의 근거가 뭐였지?", "이 Outcome은 어떤 Decision에서 나왔지?", "ToolCall은 Session의 일부인가?" — 이런 질문에 답하려면 지식의 **스키마**가 필요하다.

**HippoRAG**: `(entity, relation, entity)` 트리플만 존재. "Stanford is_located_in California" 수준. 타입 계층이나 속성 제약 없음.

**Zep**: episode → entity → community 3계층으로 고정. 커스텀 타입을 정의할 수 없다. "incident"나 "api_endpoint" 같은 도메인 특화 타입을 추가하려면 코드를 수정해야 한다.

**Mem0**: key-value (기본) 또는 entity-relation (Mem0g). 스키마가 없어서 어떤 entity든 어떤 relation이든 만들 수 있다. 자유도가 높지만, "resulted_in은 Decision에서 Outcome으로만 연결 가능"같은 **제약을 걸 수 없다**. 잘못된 관계가 만들어져도 검증할 방법이 없다.

**A-MEM**: Zettelkasten note (title, content, tags, links). 사실상 linked document. "이 note는 Decision 타입이고 rationale이 필수 속성"이라는 구조적 정의 불가.

**MAGMA**: 4개 그래프로 관점을 분리한 건 진보적이지만, **그래프 안에서의 타입 계층이 없다**. semantic graph에서 "Decision"과 "Outcome"은 그냥 다른 entity일 뿐, "Decision이 Outcome을 만든다"는 구조적 관계가 정의되지 않는다.

**Letta/MemGPT**: core/archival/recall 3계층은 **저장 위치**의 구분이지 지식의 **의미적 구조**가 아니다.

#### Synaptic Memory의 접근

두 가지 레이어의 스키마 시스템을 갖는다.

**(a) OntologyRegistry** — 코드에서 타입 계층을 정의:

```python
ontology.register_type(TypeDef(name="knowledge"))
ontology.register_type(TypeDef(name="decision", parent="knowledge",
    properties=[PropertyDef(name="rationale", required=True)]))

ontology.register_constraint(RelationConstraint(
    edge_kind="resulted_in",
    domain_types=["decision"],
    range_types=["outcome"],
))

ontology.validate_edge("resulted_in", "concept", "outcome")
# → ["source type 'concept' not in allowed domains ['decision']"]
```

상속도 지원: `technical_decision` 은 `decision` 의 `rationale` 을 자동 상속.

**(b) DomainProfile (TOML)** — 도메인 지식을 코드 바깥에 선언:

```toml
# profiles/hr.toml
[ontology]
entity_types = ["employee", "department", "project"]
ontology_hints = { employee_id = "employee", dept_code = "department" }

[lexical]
stopwords = ["주식회사", "님"]
synonyms = { "퇴사" = ["이직", "퇴직"], "월급" = ["급여", "연봉"] }

[ingestion]
chunk_size = 512
chunk_overlap = 64
```

`from_data("./hr_data/", profile="profiles/hr.toml")` 한 줄이면 위 설정이 인덱싱·검색 전체 파이프라인에 적용된다. 같은 라이브러리로 법률·의료·전자상거래 도메인을 각각 TOML 한 장으로 전환. Profile은 `profile_generator` 3-tier(rule → classifier → LLM)로 자동 생성도 가능.

이로써 "코어 코드에 도메인 종속 로직을 두지 않는다"는 원칙이 실제로 작동한다.

---

### 3-6. 비정형 + 정형 데이터를 함께 다루지 못한다 (Document-only or KV-only)

대부분의 에이전트 메모리는 **텍스트 문서** 또는 **대화 히스토리**만 받는다. 하지만 실제 기업에서 가장 중요한 지식은:
- **비정형**: 정책 문서, 매뉴얼, 회의록, PDF, 마크다운
- **정형**: 제품 DB, 직원 테이블, 주문 이력, 스펙 시트, CSV/Parquet

두 종류가 **하나의 그래프** 안에 섞여 있어야 "이 직원이 담당한 프로젝트와 관련된 회의록은?" 같은 질의가 가능하다.

**HippoRAG / Mem0 / A-MEM / MAGMA**: 텍스트만 다룬다. 테이블 행을 넣으려면 사용자가 직접 자연어로 변환해야 한다.
**Zep**: 에피소드 = 대화 메시지. 기업 DB ingestion 경로 없음.
**Letta/MemGPT**: core/archival 전부 텍스트 블록.

#### Synaptic Memory의 접근

`TableIngester` + `DBIngester` 가 CSV/Parquet/PostgreSQL 테이블을 **typed property 노드**로 전환한다. 각 행은 ontology type 을 갖는 노드가 되고, 외래키는 그래프 엣지가 된다. 그리고 3개의 정형 전용 MCP 도구가 LLM에 노출된다:

```python
# filter_nodes — WHERE 절 대체
filter_nodes(node_type="order", conditions={"status": "shipped", "total_gte": 100})

# aggregate_nodes — GROUP BY 대체
aggregate_nodes(node_type="order", group_by="customer_id",
                agg="sum", field="total")

# join_related — JOIN 대체
join_related(from_type="customer", to_type="order",
             via_edge="placed", to_conditions={"status": "pending"})
```

같은 `SynapticGraph` 인스턴스 안에서 문서 FTS 검색과 테이블 필터·집계·조인이 공존한다. 에이전트는 intent에 따라 도구를 선택만 하면 된다. 이는 v0.12에서 도입된 차별화 포인트로, 다른 6개 시스템 어느 것도 같은 그래프에서 제공하지 않는다.

---

## 4. v0.12 — Easy API와 벤치마크

### 4-1. `from_data()` — 2줄로 시작

```python
from synaptic import SynapticGraph

graph = await SynapticGraph.from_data("./my_data/")   # CSV/JSONL/PDF/MD 자동 감지
result = await graph.search("내 질문")
```

내부적으로는:
1. 파일 형식 감지 (`.csv` → TableIngester, `.pdf/.md/.txt` → DocumentIngester, DB URL → DBIngester)
2. 자동 DomainProfile 생성 (rule → classifier → optional LLM)
3. Category → Document → Chunk 계층 생성 + MENTIONS 엣지
4. BM25 인덱스(FTS5 + Kiwi) 및 HNSW 벡터 인덱스 구축
5. 검색 준비 완료

코어 의존성은 여전히 **0개**다. 백엔드·임베더·한국어 분석은 전부 optional extras (`pip install synaptic-memory[sqlite,korean,vector,embedding]`).

### 4-2. EvidenceSearch 3세대 파이프라인

```
Step 0: 쿼리 임베딩 (BYO embedder)
Step 1: QueryAnchorExtractor (카테고리/엔티티/키워드)
Step 2a: FTS seed — BM25 + Kiwi 형태소 + title 3x boost
Step 2b: Vector seed — usearch HNSW, cascade
Step 2c: Vector PRF — top-3 임베딩 평균 → 2차 검색
Step 3:  GraphExpander — 1-hop (category siblings, chunk-next, MENTIONS)
Step 3b: PPR graph discovery
Step 4:  HybridReranker — lexical + semantic + graph + structural + authority + temporal + MaxP
Step 4b: Cross-encoder reranker (BYO, TEI/Ollama)
Step 5:  EvidenceAggregator — MMR + per-doc cap + category coverage
```

v0.11 대비 알고리즘 개선:
- **Phase 1** (MaxP document aggregation + Vector PRF): Hard-set MRR +21%
- **Phase 2** (usearch HNSW): 벡터 검색 레이턴시 **11,000ms → 1ms (100×)**
- **Phase 3** (Cross-encoder bge-reranker-v2-m3): Hard-set MRR +22%
- **Phase 4** (PPR graph discovery): 크로스 문서 hit rate +15%
- **Kiwi 형태소**: 한국어 조사 분리 (한글 비율 50%+ 자동 감지)

### 4-3. 벤치마크

9개 데이터셋 자동 평가 (`uv run python eval/run_all.py`). v0.12 기준:

| 데이터셋 | 언어 | Corpus | MRR | Hit | 비고 |
|---------|------|--------|:---:|:---:|------|
| KRRA Easy (20q) | KO | 19,720 | **0.975** | 20/20 | FTS + Kiwi + embed |
| KRRA Hard (15q) | KO | 19,720 | **0.933** | 15/15 | + embed + cross-encoder |
| KRRA Hard Multi-turn | KO | 19,720 | — | 12/15 (80%) | GPT-4o-mini agent |
| assort Easy (15q) | KO | 13,909 | **0.889** | 14/15 | 정형 CSV |
| HotPotQA-24 | EN | 226 | **0.727** | 24/24 | multi-hop |
| Allganize RAG-ko | KO | 200 | **0.621** | 180/200 | 기업 문서 |
| Allganize RAG-Eval | KO | 300 | **0.615** | 264/300 | 금융/의료/법률 |
| AutoRAG | KO | 720 | **0.592** | 98/114 | 기업 검색 |
| X2BEE Easy (20q) | EN | 19,843 | **1.000** | 20/20 | DB→온톨로지 (FTS) |
| X2BEE Hard Multi-turn | EN/KO | 19,843 | — | 8/19 (42%) | structured tools agent |

모든 수치는 **로컬 실행** 기준. 인덱싱 단계 LLM 호출 0회. 벤치마크 실행 자체에 드는 API 비용은 multi-turn agent 평가에 쓰이는 GPT-4o-mini 호출뿐이다.

---

## 5. 시스템별 강점과 약점

앞 섹션들은 Synaptic의 설계 원칙을 부각하기 위해 **"Synaptic이 해결했다고 주장하는 축"** 을 기준으로 정리했다. 균형을 위해 이 섹션에서는 각 시스템을 **자체 논문/문서가 내세우는 강점과 함께** 서술하고, Synaptic 의 한계도 솔직히 나열한다.

> 중요한 주의: 아래 서술은 **각자의 논문·공식 문서·코드 공개 범위에 근거한 질적 비교**다. 공통 데이터셋에서 동일 조건으로 head-to-head 측정한 결과가 아니다. HotPotQA 정도를 제외하면 평가 프로토콜·코퍼스·임베더가 달라서 숫자 비교에는 한계가 있다.

### HippoRAG (NeurIPS 2024)
**강점**: PPR 기반 spreading activation은 multi-hop QA에서 강력하다 — MuSiQue·2WikiMultiHopQA에서 경쟁력 있는 성능. 단일 retrieval 스텝으로 다중 홉을 근사한다는 점이 이론적으로 우아하며, 해마 이론에서 출발한 설계가 후속 연구에 영감을 주었다.
**약점**: OpenIE 기반 KG 구축에 LLM 비용이 들고, 구축 이후 KG가 고정되어 사용 패턴으로 학습하지 않는다. 타입 시스템이 없고, 망각 메커니즘도 없어 스케일에서 hub-drift 문제가 보고됨.

### Zep / Graphiti (2025)
**강점**: bi-temporal(valid_at / invalid_at) 모델은 "언제 참이었는가"를 추적하는 가장 엄밀한 구현 중 하나. 상업 제품으로 운영 경험이 축적되어 있고, Neo4j 백엔드 기반으로 성숙한 그래프 쿼리 생태계를 활용할 수 있음. 관계 스키마를 structured output 으로 강제.
**약점**: entity/relation 추출이 LLM에 의존 → 작은 모델에서 스키마 오류 보고. episode→entity→community 3계층이 고정이라 도메인 특화 타입 추가는 코드 수정 필요. 사용 빈도·성공률 기반 학습은 없음.

### Mem0 / Mem0g (2025)
**강점**: 단순하고 직관적인 extract-update 파이프라인으로 프로덕션 도입 문턱이 낮음. OpenAI·Anthropic 등 주요 에이전트 스택에 통합. retrieval 단계의 token 효율은 논문에서 강조되는 장점.
**약점**: 논문에서 "consolidation is not fully automated" 라고 직접 인정. LLM에 conflict detection 을 의존해 메모리 1개 처리 지연이 비교 실험에서 1350.9초로 보고됨. 스키마가 없어 관계 제약을 걸 수 없음.

### A-MEM (NeurIPS 2025)
**강점**: Zettelkasten 기반 자기조직화로 메모리가 스스로 링크·태깅되는 구조가 신선함. LoCoMo 등 긴 대화 벤치마크에서 눈에 띄는 개선. "agentic memory" 개념을 구체화한 첫 시스템 중 하나.
**약점**: 메모리 추가마다 LLM을 여러 번 호출해 비용·지연이 큼. Zettelkasten 은 본래 "버리지 않는" 방법론이라 망각이 없고, 규모가 커질수록 링크 그래프가 무거워짐.

### MAGMA (2026)
**강점**: semantic/temporal/causal/entity 4개 직교 그래프 분리는 이론적으로 가장 포괄적인 아키텍처. intent에 따라 다른 view를 선택하는 retrieval 은 Synaptic 의 intent 개념과 같은 방향성을 공유.
**약점**: 4개 그래프를 동시에 유지·동기화하는 엔지니어링 비용이 큼. causal 추출이 LLM 의존이라 논문이 직접 "susceptible to extraction errors and hallucinations" 인정. 참조 구현·벤치마크가 제한적.

### Letta / MemGPT (2023–)
**강점**: "OS 메타포" 가 직관적이고 개발자 교육 비용이 낮음. 성숙한 오픈소스 제품으로 문서·예제·커뮤니티가 풍부. function calling 기반 메모리 관리는 기존 LLM 에이전트 루프와 잘 맞음.
**약점**: core/archival/recall 은 저장 위치 계층이지 의미적 스키마가 아님. 요약 기반 archival 은 정보 손실이 불가피하고, context limit 초과 버그가 공개 이슈에서 반복 보고됨. 그래프 구조가 없어 다중 홉 추론에 약함.

### Synaptic Memory (이 프로젝트)
**강점**
- 인덱싱 단계 LLM 호출 0회. OpenIE / relation extraction 없이 구조적·통계적 신호만으로 그래프 구축.
- Hebbian learning + 4단계 consolidation + TTL decay 로 사용 패턴 기반 생명주기가 규칙 기반으로 작동.
- 6개 intent × 5축 resonance 가중치로 검색 전략을 쿼리 시점에 전환.
- OntologyRegistry(코드) + DomainProfile TOML(설정) 이라는 2-layer 스키마로 범용화.
- 정형(CSV/DB) + 비정형(PDF/MD) 을 같은 그래프에 올리고 `filter_nodes / aggregate_nodes / join_related` 도구로 에이전트에 노출.
- 코어 의존성 0, torch-free, BYO embedder/reranker (Ollama·TEI·API).
- `from_data()` 2줄 API, 29개 MCP 도구, 9개 데이터셋 자동 평가 스크립트.

**알려진 약점 및 미검증 영역**
- **프로덕션 성숙도 낮음**. 단일 저자 오픈소스이며 대규모 운영 사례 없음. Zep·Letta·Mem0 는 상업·커뮤니티 경험이 훨씬 많음.
- **공정한 head-to-head 벤치마크 부재**. 섹션 4의 숫자는 자체 재현 환경 기준. HippoRAG / Mem0 / Zep 과 동일 코퍼스·프로토콜에서 비교한 공개 결과는 아직 없음. 현재 HotPotQA 0.727 이 공통 비교 가능한 유일한 지점이며, 이것도 corpus 샘플링이 다르다.
- **Hebbian / consolidation 의 실제 효과 미입증**. 설계는 합리적이지만 "이 메커니즘이 장기 운영에서 retrieval 품질을 유의미하게 개선하는가"는 정량화된 증거가 없다. 내부 시나리오 테스트만 존재.
- **한국어 편중**. Kiwi 형태소, KRRA/assort/Allganize 등 평가 다수가 한국어. 영어 벤치마크는 HotPotQA / X2BEE 정도로 폭이 좁음.
- **Kuzu 백엔드 아카이브 상태**. 2025-10 Apple 인수로 upstream 동결(0.11.3 고정). 현재 동작하지만 장기 지원 불확실. LadybugDB fork 는 검증 결과 drop-in 불가. 당분간 SQLite + Postgres 가 안전한 선택.
- **인프라 부담은 사용자에게 전가**. torch-free 를 달성한 대가로 embedder/cross-encoder 서빙(Ollama, TEI 등) 을 사용자가 직접 운영해야 함. 완제품 SaaS 가 아님.
- **통합·생태계 부족**. LangChain·LlamaIndex 등 주류 프레임워크 어댑터 미비. MCP 외 경로로는 수동 통합 필요.
- **벤치마크의 자기평가 편향**. `eval/run_all.py` 는 자체 스크립트이며, 공개 리더보드(예: BEIR, MTEB, KILT) 등재 없음. 숫자는 참조 지표이지 공정한 비교가 아님.

---

## 6. 포지셔닝 — 어떤 경우에 Synaptic 이 맞고, 어떤 경우에 아닌가

Synaptic Memory 는 "가장 나은 에이전트 메모리" 를 주장하지 않는다. 대신 **설계 선택의 조합이 특정 시나리오에 잘 맞는다** 는 정도가 정직한 표현이다.

**설계 원칙의 계보:**
- HippoRAG 의 spreading activation 이 multi-hop retrieval에 효과적이라는 관찰에서 출발하되, "읽기 전용 KG" 를 넘어 쓰기·학습·망각을 추가하려 함.
- Zep 의 temporal KG 가 시간 축 유효성을 엄밀히 추적한다는 관찰에서 출발하되, "시간만으로는 유용성을 못 판단한다" 는 문제에 사용 패턴 기반 생명주기로 접근.
- MAGMA 의 intent-aware retrieval 이 맞는 방향이라고 판단하되, 4-graph 유지 비용이 실전에서 부담이라는 가정에서 단일 그래프 + intent dispatch 로 단순화.
- 시맨틱 웹 온톨로지의 타입 엄격성을 가져오되, OWL/RDF 의 복잡도 없이 Python dataclass + TOML 로 경량화.
- GraphRAG 계열이 인덱싱에 LLM 을 쓰는 흐름에 반대로, BM25 + HNSW + MENTIONS phrase hub 로 LLM-free indexing 에 집중.

**Synaptic 을 고를 만한 경우**
- 기업 내부 데이터(문서 + DB) 를 **로컬에서** 인덱싱해야 하고, 인덱싱 비용·프라이버시·재현성이 중요한 경우
- 한국어 데이터가 비중 있고, Kiwi 형태소 기반 FTS 의 품질이 필요한 경우
- `from_data()` 수준의 간단한 API 로 빠르게 PoC 를 만들고, 나중에 DomainProfile 로 튜닝할 여지를 원하는 경우
- MCP 기반 에이전트 툴 사용이 중심이고, 29개 도구를 에이전트에 그대로 노출하고 싶은 경우
- 정형 테이블(filter/aggregate/join) 과 비정형 문서 검색을 같은 그래프에서 다루고 싶은 경우
- embedder/reranker 를 **본인이 골라서** 운영하는 것이 부담이 아니라 장점인 경우

**Synaptic 을 고르면 안 되는 경우**
- 즉시 프로덕션에 투입 가능한 "배터리 포함" 상업 제품이 필요하다면 → Zep, Letta, Mem0 의 managed 서비스가 더 안전
- 영어 위주 대규모 공개 벤치마크 경쟁이 목표라면 → 자체 벤치마크의 대표성 한계 때문에 HippoRAG·GraphRAG 의 공개 결과를 직접 참고하는 편이 빠름
- embedder/cross-encoder 를 별도로 운영할 여력이 없다면 → API 기반의 단순한 메모리 계층(Mem0 등)이 총비용이 낮을 수 있음
- LangChain / LlamaIndex 생태계 안에서만 작업한다면 → 현재 어댑터가 없어 통합 비용이 추가됨
- "검증된 대규모 운영 사례"가 도입 조건이라면 → 현 단계 Synaptic 은 해당 요건을 만족하지 않음

핵심 주장은 단 하나: **인덱싱에서 LLM 을 뺄 수 있다면, 메모리의 관리 비용과 hallucination 위험을 동시에 줄일 수 있다**. Synaptic 은 그 가설 위에 만들어진 참조 구현이며, 이 가설이 당신의 시나리오와 맞는지가 결국 도입 여부를 결정한다.

---

## References

- [HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs](https://arxiv.org/abs/2405.14831) — NeurIPS 2024
- [Breaking the Static Graph: Context-Aware Traversal for Robust RAG](https://arxiv.org/abs/2602.01965) — 2026
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) — 2025
- [Graphiti: Knowledge Graph Memory (Neo4j)](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) — 2025
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) — NeurIPS 2025
- [MAGMA: A Multi-Graph based Agentic Memory Architecture](https://arxiv.org/abs/2601.03236) — 2026
- [Letta/MemGPT Documentation](https://docs.letta.com/concepts/memgpt/)
- [Memory in the Age of AI Agents: A Survey](https://arxiv.org/abs/2512.13564) — 2025
- [Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers](https://arxiv.org/html/2603.07670) — 2026
- [Anatomy of Agentic Memory: Taxonomy and Empirical Analysis](https://arxiv.org/abs/2602.19320) — 2026
- [ICLR 2026 Workshop: MemAgents](https://openreview.net/pdf?id=U51WxL382H)
- [Agent Memory Paper List (Comprehensive Survey)](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
