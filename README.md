# Synaptic Memory

LLM/멀티에이전트를 위한 뇌 기반 지식 그래프.

에이전트가 운영 중에 만들어내는 모든 데이터 — tool call, 의사결정, 결과, 학습 — 를 **자동으로 온톨로지에 구축**하고, 나중에 에이전트가 스스로 검색·추론할 수 있게 하는 라이브러리 + MCP 서버.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why — 이 프로젝트가 풀려는 문제

LLM 에이전트는 **기억하지 못한다.** 매번 같은 실수를 반복하고, 과거의 성공 패턴을 활용하지 못하고, 팀의 축적된 지식에 접근하지 못한다.

기존 RAG는 "문서를 잘게 잘라서 벡터로 검색"하는 데 그친다. 하지만 에이전트에게 필요한 건 문서 검색이 아니라 **경험의 구조화**다:

- "지난번 이런 상황에서 어떤 결정을 했고, 결과가 어땠지?"
- "이 패턴이 실패했던 적이 있나? 왜?"
- "이 도구를 쓸 때 지켜야 할 규칙이 뭐지?"

Synaptic Memory는 이 문제를 **뇌의 작동 방식**에서 답을 가져온다.

---

## Differentiators — 왜 Synaptic Memory인가

| | Synaptic Memory | Cognee | Mem0 | LightRAG |
|---|---|---|---|---|
| **에이전트 경험 학습** | ✅ Hebbian co-activation | ❌ | ❌ | ❌ |
| **메모리 정리 (4단계)** | ✅ L0→L1→L2→L3 | ❌ | △ | ❌ |
| **온톨로지 자동 구축** | ✅ 규칙 + LLM + 임베딩 | △ LLM만 | ❌ | △ LLM만 |
| **다축 랭킹** | ✅ relevance×importance×recency×vitality×context | ❌ | ❌ | ❌ |
| **Zero-dep 코어** | ✅ 순수 Python | ❌ | ❌ | ❌ |
| **MCP 서버** | ✅ 16 tools | ❌ | ❌ | ❌ |
| **한국어 최적화** | ✅ FTS + synonym 튜닝 | ❌ | ❌ | ❌ |

### 벤치마크 (FTS only, embedding 없이)

| 데이터셋 | Corpus | MRR | nDCG@10 | R@10 |
|----------|--------|-----|---------|------|
| Allganize RAG-Eval (금융/의료/법률) | 300 | **0.793** | 0.810 | 0.870 |
| HotPotQA-24 (multi-hop, Cognee 비교) | 226 | **0.754** | 0.636 | 0.729 |
| AutoRAGRetrieval (엔터프라이즈) | 720 | **0.639** | 0.677 | 0.800 |
| KLUE-MRC (한국어 QA) | 500 | **0.607** | 0.643 | 0.760 |

---

## Design Philosophy — 뇌에서 빌려온 네 가지 원리

### 1. Spreading Activation — 연상 검색

뇌는 "deploy"라는 단어를 들으면 CI/CD, rollback, 장애, 모니터링이 함께 활성화된다.

```
"배포" 검색
  → FTS 매칭: [CI/CD 파이프라인, 배포 자동화]
  → 이웃 활성화: [롤백 전략, 카나리 배포, 장애 대응 규칙]
  → Resonance 정렬: relevance × importance × recency × vitality × context
```

### 2. Hebbian Learning — "함께 성공한 것은 더 강하게 연결된다"

```
에이전트가 [PostgreSQL 선택] + [벡터 검색 구현]을 함께 사용 → 성공
  → edge weight += 0.1 → 다음 검색 시 함께 활성화

에이전트가 [테스트 스킵] + [프로덕션 배포]를 함께 사용 → 실패
  → edge weight -= 0.15 → 실패 경험이 먼저 뜬다
```

### 3. Memory Consolidation — 중요한 기억만 남기기

```
L0 (Raw, 72h)    ← 모든 기록. 72시간 후 미접근 시 삭제.
L1 (Sprint, 90d)  ← 3회+ 접근. 90일 유지.
L2 (Monthly, 365d) ← 10회+ 접근. 1년 유지.
L3 (Permanent)    ← 성공률 80%+. 영구 보존. (60% 미만 시 강등)
```

### 4. Auto-Ontology — LLM이 잘 찾을 수 있는 구조로 자동 적재

**"나중에 이 지식을 언제 찾게 될까?"** 를 예측하여 메타데이터를 자동 생성한다:

```python
await graph.add("결제 장애 사후 분석", "PG사 API 타임아웃...")

# LLM이 자동 생성:
# kind: LESSON
# tags: ["결제", "PG사", "타임아웃", "서킷브레이커"]
# search_keywords: ["결제 실패 원인", "PG사 장애 대응", "API 타임아웃 해결"]
# search_scenarios: ["결제 시스템 장애 발생 시 과거 사례 검색"]
# 기존 노드와 관계: --[LEARNED_FROM]--> "배포 결정"
```

3단계 자동 구축:

| 모드 | 설정 | 비용 | 특징 |
|------|------|------|------|
| **규칙 기반** | `RuleBasedClassifier()` | 무료 | 키워드 매칭, zero-dep |
| **+ 임베딩** | `+ RuleBasedRelationDetector()` + embedder | 로컬 무료 | cosine similarity 자동 연결 |
| **+ LLM** | `LLMClassifier()` + `LLMRelationDetector()` | 로컬/API | 검색 키워드 예측, 의미적 관계 추출 |

---

## Install

```bash
pip install synaptic-memory                      # 코어 (zero deps)
pip install synaptic-memory[embedding]           # + auto-embedding (Ollama/vLLM)
pip install synaptic-memory[sqlite]              # + SQLite FTS5
pip install synaptic-memory[scale]               # Neo4j + Qdrant + MinIO + embedding
pip install synaptic-memory[mcp]                 # + MCP server
pip install synaptic-memory[all]                 # 전부
```

## Quick Start

### 기본 — zero-dep

```python
from synaptic.backends.memory import MemoryBackend
from synaptic import SynapticGraph, ActivityTracker, build_agent_ontology

async def main():
    backend = MemoryBackend()
    await backend.connect()

    graph = SynapticGraph(backend, ontology=build_agent_ontology())
    tracker = ActivityTracker(graph)

    # 과거 경험 검색 (intent 자동 추론)
    result = await graph.agent_search("DB 마이그레이션 실패")

    # 결정 기록
    session = await tracker.start_session(agent_id="my-agent")
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
        content="Zero downtime 달성",
        success=True,
    )
    await backend.close()
```

### 자동 온톨로지 — 규칙 기반 (무료)

```python
from synaptic import SynapticGraph, RuleBasedClassifier, RuleBasedRelationDetector

graph = SynapticGraph(
    backend,
    classifier=RuleBasedClassifier(),
    relation_detector=RuleBasedRelationDetector(),
)

# kind, tags 지정 없이 넣기만 하면 자동 분류 + 자동 관계
await graph.add("환불 정책", "7일 이내 환불 가능...")  # → kind=RULE 자동
```

### 자동 온톨로지 — LLM 기반 (최고 품질)

```python
from synaptic import (
    SynapticGraph, OllamaLLMProvider, OllamaEmbeddingProvider,
    LLMClassifier, LLMRelationDetector,
    RuleBasedClassifier, RuleBasedRelationDetector,
)

llm = OllamaLLMProvider(model="qwen3:0.6b")

graph = SynapticGraph(
    backend,
    classifier=LLMClassifier(llm, fallback=RuleBasedClassifier()),
    relation_detector=LLMRelationDetector(llm, fallback=RuleBasedRelationDetector()),
    embedder=OllamaEmbeddingProvider(model="qwen3-embedding:0.6b"),
)

# LLM이 kind 분류 + tags + 검색 키워드 + 검색 시나리오 자동 생성
# embedding에 search_keywords 포함 → 벡터 검색 정확도 향상
# 기존 노드와 의미적 관계 자동 탐지 (DEPENDS_ON, LEARNED_FROM 등)
node = await graph.add("결제 장애 사후 분석", "PG사 API 타임아웃...")
```

### Auto-Embedding (vLLM / llama.cpp / Ollama)

```python
from synaptic import SynapticGraph, OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider(
    "http://gpu-server:8080/v1",
    model="BAAI/bge-m3",
)
graph = SynapticGraph(backend, embedder=embedder)

# 자동: title+content → 벡터 생성 → 저장
await graph.add("배포 전략", "Blue-green 배포로 zero downtime 달성")
# 자동: 쿼리 → 벡터 생성 → FTS + vector 동시 검색
result = await graph.search("배포 방식")
```

### Scale: CompositeBackend

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
# - embedding → Qdrant, content > 100KB → MinIO, 나머지 → Neo4j
```

---

## Architecture

```
SynapticGraph (Facade)
  │
  ├── Auto-Ontology ───── RuleBasedClassifier / LLMClassifier
  │                       RuleBasedRelationDetector / LLMRelationDetector
  ├── OntologyRegistry ── 타입 계층 + 속성 상속 + 제약 검증
  ├── ActivityTracker ─── 세션/tool call/decision/outcome 캡처
  ├── AgentSearch ──────── 6가지 intent 기반 검색 전략
  ├── HybridSearch ─────── FTS + vector → synonym → LLM rewrite
  ├── ResonanceScorer ──── 5축 공명 (relevance × importance × recency × vitality × context)
  ├── HebbianEngine ────── co-activation 강화/약화
  ├── ConsolidationCascade  L0→L3 생명주기
  ├── EmbeddingProvider ── 자동 벡터 생성 (Ollama/vLLM/OpenAI)
  ├── LLMProvider ──────── 온톨로지 구축용 LLM (Ollama/OpenAI)
  └── Exporters ─────────── Markdown, JSON
       │
  StorageBackend (Protocol)
       │
  ┌────┼──────────┬───────────────┬──────────────┐
  │    │          │               │              │
Memory SQLite  PostgreSQL     Neo4j       CompositeBackend
(dev)  (FTS5)  (pgvector)   (Cypher)    (Neo4j+Qdrant+MinIO)
```

---

## 5-axis Resonance Scoring

```
Score = 0.55 × relevance     검색 매칭 점수 [0,1]
      + 0.15 × importance    (success - failure) / access_count [0,1]
      + 0.20 × recency       exp(-0.05 × days_since_update) [0,1]
      + 0.10 × vitality      주기적 decay ×0.95 [0,1]
      + (context weight) × context  세션 태그 Jaccard 유사도 [0,1]
```

Intent별로 가중치가 다르다. `past_failures`는 importance에 높은 비중, `context_explore`는 context에 높은 비중. **같은 쿼리라도 의도에 따라 다른 결과**.

---

## Ontology

```python
from synaptic import OntologyRegistry, TypeDef, PropertyDef, build_agent_ontology

ontology = build_agent_ontology()

# 커스텀 타입 추가
ontology.register_type(TypeDef(
    name="incident",
    parent="agent_activity",
    description="Production incident",
    properties=[
        PropertyDef(name="severity", value_type="str", required=True),
    ],
))

graph = SynapticGraph(backend, ontology=ontology)
# → graph.add(), graph.link() 시 자동 검증
```

### 기본 온톨로지

```
knowledge                          agent_activity
  ├── concept                        ├── session
  ├── entity                         ├── tool_call
  ├── lesson                         ├── observation
  ├── decision                       ├── reasoning
  ├── rule                           └── outcome
  └── artifact
```

## Backends

| Backend | 그래프 순회 | 벡터 검색 | 스케일 | 용도 |
|---------|-----------|----------|-------|------|
| `MemoryBackend` | Python BFS | cosine | ~10K | 테스트 |
| `SQLiteBackend` | CTE 재귀 | ✗ | ~100K | 임베디드 |
| `PostgreSQLBackend` | CTE 재귀 | pgvector HNSW | ~1M | 프로덕션 |
| `Neo4jBackend` | Cypher native | ✗ | ~10B | 대규모 그래프 |
| `QdrantBackend` | ✗ | HNSW + 양자화 | ~10B | 벡터 전용 |
| `MinIOBackend` | ✗ | ✗ | ~10TB | blob (S3 호환) |
| `CompositeBackend` | Neo4j | Qdrant | ∞ | **통합 라우터** |

## MCP Server — 16 Tools

```bash
synaptic-mcp                                              # stdio (Claude Code)
synaptic-mcp --db ./knowledge.db                         # SQLite
synaptic-mcp --embed-url http://localhost:8080/v1        # + auto-embedding
```

**Knowledge** (7) — `knowledge_search`, `knowledge_add`, `knowledge_link`, `knowledge_reinforce`, `knowledge_stats`, `knowledge_export`, `knowledge_consolidate`

**Agent Workflow** (4) — `agent_start_session`, `agent_log_action`, `agent_record_decision`, `agent_record_outcome`

**Semantic Search** (3) — `agent_find_similar`, `agent_get_reasoning_chain`, `agent_explore_context`

**Ontology** (2) — `ontology_define_type`, `ontology_query_schema`

## Dev

```bash
uv sync --extra dev --extra sqlite --extra neo4j --extra qdrant --extra minio
uv run pytest -v                              # 266+ tests
uv run pytest tests/benchmark/ -v -s          # 벤치마크 (8개 데이터셋 + ablation)
uv run ruff check --fix && uv run ruff format
```

## License

MIT
