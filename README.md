# Synaptic Memory

LLM/멀티에이전트를 위한 뇌 기반 지식 그래프.

에이전트가 운영 중에 만들어내는 모든 데이터 — tool call, 의사결정, 결과, 학습 — 를 Graph DB에 온톨로지로 구축하고, 나중에 에이전트가 스스로 검색·추론할 수 있게 하는 라이브러리 + MCP 서버.

---

## Why — 이 프로젝트가 풀려는 문제

LLM 에이전트는 **기억하지 못한다.** 매번 같은 실수를 반복하고, 과거의 성공 패턴을 활용하지 못하고, 팀의 축적된 지식에 접근하지 못한다.

기존 RAG는 "문서를 잘게 잘라서 벡터로 검색"하는 데 그친다. 하지만 에이전트에게 필요한 건 문서 검색이 아니라 **경험의 구조화**다:

- "지난번 이런 상황에서 어떤 결정을 했고, 결과가 어땠지?"
- "이 패턴이 실패했던 적이 있나? 왜?"
- "이 도구를 쓸 때 지켜야 할 규칙이 뭐지?"

Synaptic Memory는 이 문제를 **뇌의 작동 방식**에서 답을 가져온다.

---

## Design Philosophy — 뇌에서 빌려온 네 가지 원리

### 1. Spreading Activation — 연상 검색

뇌는 "deploy"라는 단어를 들으면 CI/CD, rollback, 장애, 모니터링이 함께 활성화된다. 키워드 매칭이 아니라 **연결된 개념이 함께 떠오르는 것**.

Synaptic Memory의 검색도 동일하게 작동한다:

```
"배포" 검색
  → FTS 매칭: [CI/CD 파이프라인, 배포 자동화]
  → 이웃 활성화: [롤백 전략, 카나리 배포, 장애 대응 규칙]
  → Resonance 정렬: 최근성 × 중요도 × 성공률 × 맥락 친화도
```

텍스트가 아니라 **지식의 그래프 구조**를 따라 탐색한다. 이것이 RAG와의 근본적 차이.

### 2. Hebbian Learning — "함께 성공한 것은 더 강하게 연결된다"

뇌의 시냅스는 함께 발화하면 강화된다 (Hebb's Rule). Synaptic Memory는 이를 그대로 구현한다:

```
에이전트가 [PostgreSQL 선택] + [벡터 검색 구현]을 함께 사용 → 성공
  → 두 노드 사이 edge weight += 0.1
  → 다음 검색 시 하나를 찾으면 다른 하나도 함께 활성화

에이전트가 [테스트 스킵] + [프로덕션 배포]를 함께 사용 → 실패
  → edge weight -= 0.15 (실패는 더 강하게 학습)
  → 다음에 "테스트 스킵"을 검색하면 실패 경험이 먼저 뜬다
```

에이전트가 명시적으로 "이건 나쁜 패턴이야"라고 태깅할 필요 없다. **사용하고 결과를 기록하면 그래프가 스스로 학습**한다. Adaptive learning rate로 초기에는 빠르게, 성숙하면 안정적으로 학습한다.

### 3. Memory Consolidation — 중요한 기억만 남기기

사람은 잠을 자는 동안 단기기억을 장기기억으로 전환한다. 자주 떠올린 기억은 강화되고, 안 쓰는 기억은 사라진다.

```
L0 (Raw, 72h)      ← 에이전트가 만든 모든 기록. 72시간 후 미접근 시 삭제.
  ↓ 3회 이상 접근
L1 (Sprint, 90d)    ← 반복 참조된 지식. 90일간 유지.
  ↓ 10회 이상 접근
L2 (Monthly, 365d)  ← 검증된 지식. 1년간 유지.
  ↓ 성공률 80%+
L3 (Permanent)      ← 조직의 핵심 지식. 영구 보존.
  ↓ 성공률 60% 미만
L2 (강등)           ← 더 이상 유효하지 않은 지식은 강등.
```

이걸 안 하면? 에이전트가 만들어내는 데이터가 무한히 쌓여서 검색 품질이 떨어진다. **쓰이는 지식만 살아남는 자연선택**.

### 4. Ontology — 지식에 구조를 부여하기

플랫한 key-value 저장소에서는 "이 결정의 근거가 뭐였지?"를 물을 수 없다. 온톨로지는 **지식의 스키마**를 정의한다:

```
Decision --[resulted_in]--> Outcome --[learned_from]<-- Lesson
    |
    +--[depends_on]--> Context
    +--[part_of]--> Session
```

타입 계층으로 "tool_call은 agent_activity의 하위 타입"이라는 관계를 표현하고, "resulted_in 엣지는 Decision에서 Outcome으로만 연결 가능"이라는 제약을 건다. 이 구조 덕분에 에이전트는 단순 텍스트 매칭이 아니라 **의미적 관계를 따라 추론**할 수 있다.

---

## How It Works — 에이전트의 하루

전체 흐름을 하나의 시나리오로:

```
1. 에이전트 세션 시작
   → Session 노드 생성

2. 에이전트가 "DB 마이그레이션" 작업을 받음
   → agent_find_similar("DB 마이그레이션", intent="similar_decisions")
   → 과거 Decision 3건 + Outcome 3건 + Lesson 1건 반환
   → "지난번 zero-downtime 마이그레이션 성공했었네. 그때 사용한 전략은..."

3. 에이전트가 결정을 내림: "Blue-green 마이그레이션 사용"
   → Decision 노드 생성 (rationale, alternatives 포함)
   → 참조한 지식 노드에 depends_on 엣지

4. 에이전트가 도구를 실행
   → ToolCall 노드 생성 (tool_name, params, result, duration)
   → Session에 part_of + followed_by 체인

5. 결과 확인: 성공
   → Outcome 노드 생성 (success=true)
   → Decision --[resulted_in]--> Outcome 엣지
   → Hebbian reinforcement: 관련 노드 간 연결 강화

6. Consolidation (주기적)
   → 72시간 동안 안 쓴 L0 노드 삭제
   → 자주 참조된 노드 L1→L2 승격
   → 성공률 80%+ 노드 L3 (영구) 승격
   → 성공률 60% 미만 L3 노드 → L2 강등
   → edge weight < 0.1인 약한 연결 정리
```

이 사이클이 반복되면서 **에이전트의 경험이 그래프에 축적**되고, 시간이 지날수록 검색 품질이 올라간다.

---

## Architecture

```
SynapticGraph (Facade)
  │
  ├── OntologyRegistry ─── 타입 계층 + 속성 상속 + 제약 검증
  ├── ActivityTracker ──── 세션/tool call/decision/outcome 캡처
  ├── AgentSearch ──────── 6가지 intent 기반 검색 전략
  ├── HybridSearch ─────── FTS + fuzzy + vector → synonym → LLM rewrite
  ├── ResonanceScorer ──── 5축 (relevance × importance × recency × vitality × context)
  ├── HebbianEngine ────── co-activation 강화/약화 (adaptive rate)
  ├── ConsolidationCascade  L0→L3 생명주기 + L3 강등
  ├── EmbeddingProvider ── 자동 벡터 생성 (vLLM/llama.cpp/Ollama)
  ├── NodeCache (LRU)
  └── Exporters (Markdown, JSON)
       │
  StorageBackend (Protocol — 20개 메서드)
       │
  ┌────┼──────────┬───────────────┬──────────────┐
  │    │          │               │              │
Memory SQLite  PostgreSQL     Neo4j       CompositeBackend
(dev)  (FTS5)  (pgvector)   (Cypher)    (Neo4j+Qdrant+MinIO)
```

**핵심 설계 결정:**

- **Protocol-based** — 코어가 백엔드를 모른다. SQLite든 Neo4j든 같은 API. 백엔드 교체 시 코드 변경 0.
- **Zero core deps** — 코어는 순수 Python. `pip install synaptic-memory`에 외부 의존성 없음.
- **CompositeBackend** — Neo4j(그래프+FTS) + Qdrant(벡터 ANN) + MinIO(blob)를 하나의 StorageBackend로 통합. 용도별 라우팅.
- **Auto-embedding** — EmbeddingProvider를 주입하면 `add()`/`search()` 시 자동으로 벡터 생성. vLLM, llama.cpp, Ollama, TEI 등 OpenAI-compatible 엔드포인트 호환.

---

## 5-axis Resonance Scoring

검색 결과는 단순 텍스트 유사도가 아니라 5개 축의 가중합으로 정렬된다:

```
Score = 0.35 × relevance     검색 매칭 점수 [0,1]
      + 0.20 × importance    (success - failure) / access_count [0,1]
      + 0.15 × recency       exp(-0.05 × days_since_update) [0,1]
      + 0.10 × vitality      주기적 decay ×0.95 [0,1]
      + 0.20 × context       현재 세션 태그와의 Jaccard 유사도 [0,1]
```

Intent별로 가중치가 다르다. `past_failures`는 importance(성공률)에 0.35를, `context_explore`는 context(태그 친화도)에 0.40을 준다. 에이전트가 "왜 이걸 찾고 있는지"에 따라 **같은 쿼리라도 다른 결과**가 나온다.

---

## Install

```bash
pip install synaptic-memory                      # 코어 (zero deps)
pip install synaptic-memory[sqlite]              # + SQLite
pip install synaptic-memory[neo4j]               # + Neo4j
pip install synaptic-memory[neo4j,embedding]     # + Neo4j + auto-embedding
pip install synaptic-memory[scale]               # Neo4j + Qdrant + MinIO + embedding
pip install synaptic-memory[mcp]                 # + MCP server
pip install synaptic-memory[all]                 # 전부
```

## Quick Start

```python
from synaptic.backends.memory import MemoryBackend
from synaptic import SynapticGraph, ActivityTracker, NodeKind, build_agent_ontology

async def main():
    backend = MemoryBackend()
    await backend.connect()

    graph = SynapticGraph(backend, ontology=build_agent_ontology())
    tracker = ActivityTracker(graph)

    # 세션 시작
    session = await tracker.start_session(agent_id="my-agent")

    # 과거 경험 검색 (intent 자동 추론)
    result = await graph.agent_search("DB 마이그레이션 실패")
    # → intent="past_failures" 자동 선택

    # 결정 기록
    decision = await tracker.record_decision(
        session.id,
        title="PostgreSQL 선택",
        rationale="벡터 검색 + ACID 필요",
        alternatives=["MongoDB", "SQLite"],
    )

    # 결과 기록 → 자동 Hebbian learning
    await tracker.record_outcome(
        decision.id,
        title="마이그레이션 성공",
        content="Zero downtime, 벡터 검색 정상 작동",
        success=True,
    )

    await backend.close()
```

## Auto-Embedding (vLLM / llama.cpp / Ollama)

EmbeddingProvider를 주입하면 모든 노드가 자동 임베딩 + 벡터 검색 활성화:

```python
from synaptic import SynapticGraph, OpenAIEmbeddingProvider

# vLLM, llama.cpp, Ollama, TEI — 어디든 동일한 인터페이스
embedder = OpenAIEmbeddingProvider(
    "http://gpu-server:8080/v1",   # OpenAI-compatible 엔드포인트
    model="BAAI/bge-m3",
)

graph = SynapticGraph(backend, embedder=embedder)

# 자동: title+content → 벡터 생성 → Qdrant 저장
await graph.add("배포 전략", "Blue-green 배포로 zero downtime 달성")

# 자동: 쿼리 → 벡터 생성 → FTS + fuzzy + vector 동시 검색
result = await graph.search("배포 방식")
```

## Scale: CompositeBackend

Neo4j(그래프) + Qdrant(벡터) + MinIO(blob)를 하나의 StorageBackend로:

```python
from synaptic.backends.composite import CompositeBackend
from synaptic.backends.neo4j import Neo4jBackend
from synaptic.backends.qdrant import QdrantBackend
from synaptic.backends.minio_store import MinIOBackend

composite = CompositeBackend(
    graph=Neo4jBackend("bolt://localhost:7687"),
    vector=QdrantBackend("http://localhost:6333"),
    blob=MinIOBackend("localhost:9000", access_key="minio", secret_key="secret"),
)
await composite.connect()
graph = SynapticGraph(composite, embedder=embedder)

# 내부 라우팅:
# - embedding → Qdrant에 자동 저장
# - content > 100KB → MinIO에 자동 offload
# - 나머지 → Neo4j (그래프 + FTS)
# - search_vector → Qdrant ANN → Neo4j batch get
# - graph traversal → Neo4j Cypher native
```

## Ontology

```python
from synaptic import OntologyRegistry, TypeDef, PropertyDef, build_agent_ontology

ontology = build_agent_ontology()

# 커스텀 타입 추가 (상속 지원)
ontology.register_type(TypeDef(
    name="incident",
    parent="agent_activity",
    description="Production incident",
    properties=[
        PropertyDef(name="severity", value_type="str", required=True),
        PropertyDef(name="resolved", value_type="bool"),
    ],
))

# 계층/상속/검증
ontology.is_a("incident", "agent_activity")              # True
ontology.infer_properties("incident")                     # parent 속성 포함
ontology.validate_node("incident", {})                    # ["Missing 'severity'"]
ontology.validate_edge("resulted_in", "concept", "outcome")  # ["source not in domains"]

graph = SynapticGraph(backend, ontology=ontology)
# → graph.add(), graph.link() 시 자동 검증
```

### 기본 온톨로지

```
knowledge                          agent_activity
  ├── concept                        ├── session (agent_id, status)
  ├── entity                         ├── tool_call (tool_name*, success, duration_ms)
  ├── lesson                         ├── observation
  ├── decision (rationale*)          ├── reasoning
  ├── rule                           └── outcome (success*, impact)
  └── artifact
```

## Backends

| Backend | 그래프 순회 | 벡터 검색 | 스케일 | 용도 |
|---------|-----------|----------|-------|------|
| `MemoryBackend` | Python BFS | cosine | ~10K | 테스트, 프로토타이핑 |
| `SQLiteBackend` | CTE 재귀 | ✗ | ~100K | 임베디드, 단일 프로세스 |
| `PostgreSQLBackend` | CTE 재귀 | pgvector HNSW | ~1M | 프로덕션, 벡터 검색 |
| `Neo4jBackend` | Cypher native | ✗ (Qdrant 위임) | ~10B | 대규모 그래프 |
| `QdrantBackend` | ✗ | HNSW + 양자화 | ~10B | 벡터 전용 (ANN) |
| `MinIOBackend` | ✗ | ✗ | ~10TB | blob 저장 (S3 호환) |
| `CompositeBackend` | Neo4j | Qdrant | ∞ | **통합 라우터** |

## MCP Server — 16 Tools

```bash
synaptic-mcp                                              # stdio (Claude Code)
synaptic-mcp --db ./knowledge.db                         # SQLite
synaptic-mcp --embed-url http://localhost:8080/v1        # + auto-embedding
synaptic-mcp --embed-url http://localhost:8080/v1 --embed-model BAAI/bge-m3
```

**Knowledge** (7) — `knowledge_search`, `knowledge_add`, `knowledge_link`, `knowledge_reinforce`, `knowledge_stats`, `knowledge_export`, `knowledge_consolidate`

**Agent Workflow** (4) — `agent_start_session`, `agent_log_action`, `agent_record_decision`, `agent_record_outcome`

**Semantic Search** (3) — `agent_find_similar`, `agent_get_reasoning_chain`, `agent_explore_context`

**Ontology** (2) — `ontology_define_type`, `ontology_query_schema`

## Data Model

### Node Types (15)

| Category | Types |
|----------|-------|
| Knowledge | concept, entity, lesson, decision, rule, artifact, agent, task, sprint |
| Agent Activity | tool_call, observation, reasoning, outcome, session, type_def |

### Edge Types (12)

| Category | Types |
|----------|-------|
| Knowledge | related, caused, learned_from, depends_on, produced, contradicts, supersedes |
| Ontology & Activity | is_a, invoked, resulted_in, part_of, followed_by |

### Consolidation Levels

| Level | TTL | Promotion | Demotion |
|-------|-----|-----------|----------|
| L0 Raw | 72h | 3+ accesses → L1 | |
| L1 Sprint | 90d | 10+ accesses → L2 | |
| L2 Monthly | 365d | 10+ successes + 80%+ rate → L3 | |
| L3 Permanent | ∞ | 영구 보존 | 성공률 60% 미만 → L2 |

## Dev

```bash
uv sync --extra dev --extra sqlite --extra neo4j --extra qdrant --extra minio
uv run pytest -v                              # 185+ unit tests
uv run pytest -m neo4j                        # Neo4j integration (docker compose up neo4j)
uv run pytest -m qdrant                       # Qdrant integration (docker start qdrant)
uv run pytest -m composite                    # Full stack (Neo4j + Qdrant + MinIO)
uv run ruff check --fix && uv run ruff format
uv run pyright                                # strict mode
```

```bash
# 개발 인프라
docker compose up neo4j        # Neo4j (bolt://localhost:7687)
docker start qdrant            # Qdrant (http://localhost:6333)
# MinIO는 서버에서 직접 실행 (localhost:9000)
```

## License

MIT
