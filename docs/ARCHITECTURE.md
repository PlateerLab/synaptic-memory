# Synaptic Memory — Architecture

> 기준: v0.17.1 (2026-04-19). 개념 배경은 [CONCEPTS.md](CONCEPTS.md),
> 로드맵과 미해결 질문은 [PLAN-v0.18-architecture.md](PLAN-v0.18-architecture.md)
> 를 참조.

## 개요

Synaptic Memory는 **3세대 GraphRAG** 라이브러리입니다. 아무 데이터(문서,
CSV, SQL DB, PDF/DOCX/…)를 넣으면 **LLM 호출 없이** 관계형 그래프를 구축하고,
하이브리드 파이프라인 + 36개 MCP 도구 + 멀티턴 에이전트 루프로 LLM이 탐색
하게 합니다.

설계 원칙:

1. **코드는 데이터와 도구만** — 판단 로직은 전부 LLM에 위임
2. **인덱싱 LLM 비용 0원** — 관계는 FK/카테고리/청크순서 등 *구조*에서 추출
3. **BYO embedder / reranker** — torch 직접 의존 0
4. **범용** — 도메인 특화는 `DomainProfile` TOML로 주입

---

## 상위 구조

```
                 ┌───────────────────────────────────────────┐
                 │           SynapticGraph (Facade)          │
                 │  from_data() · from_database() · chat()   │
                 │  search() · agent_search() · add_document │
                 │  add_table() · sync_from_database()       │
                 │  backfill() · maintain() · reinforce()    │
                 └──┬─────────┬─────────┬────────┬───────────┘
                    │         │         │        │
       ┌────────────▼──┐  ┌───▼─────┐  │   ┌────▼────────────┐
       │ Ingest layer  │  │Evidence │  │   │ Agent layer     │
       │ ──────────── │  │ Search  │  │   │ ──────────────  │
       │ document_    │  │ (v0.12~ │  │   │ agent_loop      │
       │ ingester     │  │  v0.17) │  │   │ agent_tools     │
       │ table_       │  │         │  │   │ agent_tools_v2  │
       │ ingester     │  └────┬────┘  │   │ agent_tools_    │
       │ db_ingester  │       │       │   │   structured    │
       │ + CDC sync   │       │       │   │ search_session  │
       │ profile_gen  │       │       │   │ mcp.server (36) │
       └──────┬───────┘       │       │   └──────┬──────────┘
              │               │       │          │
              │    ┌──────────▼───────▼──────────▼──────────┐
              │    │     Domain layer (models + ontology)    │
              │    │  Node / Edge / NodeKind / EdgeKind      │
              │    │  DomainProfile (TOML injection)         │
              │    └─────────────────────┬───────────────────┘
              │                          │
              │    ┌─────────────────────▼───────────────────┐
              │    │  StorageBackend (Protocol)              │
              └────▶  MemoryBackend · SqliteGraphBackend     │
                   │  KuzuBackend · PostgreSQLBackend        │
                   │  CompositeBackend (+ Qdrant / MinIO)    │
                   └─────────────────────────────────────────┘

     주변 기능 (유지 중): HebbianEngine · ConsolidationCascade
     — reinforce() / consolidate() / maintain() 경로에서만 사용 (§7)
```

Facade가 라이프사이클(ingest → search → maintain)을 조립합니다. 각 레이어는
protocol로 느슨하게 결합되어 있어 백엔드, 임베더, 리랭커, 청커, 분류기를
자유롭게 교체할 수 있습니다 ([`protocols.py`](../src/synaptic/protocols.py)).

---

## 1. 데이터 모델

[`src/synaptic/models.py`](../src/synaptic/models.py)

### 1.1 Node

| 필드 | 타입 | 의미 |
|------|------|------|
| `id` | `str` (16 hex) | 기본 UUID. `source_url=`로 deterministic ID 활성화 가능 (CDC 필수). |
| `kind` | `NodeKind` | 노드 타입 (아래 표) |
| `title` | `str` | 표제 (FTS에서 3× 가중치) |
| `content` | `str` | 본문 |
| `tags` | `list[str]` | 태그 |
| `properties` | `dict[str,str]` | 행 데이터 / 카테고리 힌트 / `_table_name` 등 메타 |
| `embedding` | `list[float]` | 선택. BYO embedder가 채움 |
| `level` | `ConsolidationLevel` | L0_RAW → L3_PERMANENT (§7) |
| `vitality` / `access_count` / `success_count` / `failure_count` | — | 레거시 resonance/hebbian 필드 |
| `source` | `str` | 출처 (예: `sprint:xxx`, 파일 경로) |
| `created_at` / `updated_at` | `float` | epoch |

### 1.2 주요 NodeKind

| kind | 역할 | 생성 위치 |
|------|------|-----------|
| `CONCEPT` | 카테고리 / 분류 / 타입 정의 | `document_ingester` (카테고리), `ontology` |
| `ENTITY` | 정형 행 / 문서 본체 (기본값) | `table_ingester`, `db_ingester`, `document_ingester` |
| `CHUNK` | 문서 조각 | `document_ingester` |
| `RULE` / `DECISION` / `OBSERVATION` / `LESSON` | 에이전트가 학습/기록하는 지식 타입 | `add()` 직접 호출 |
| `COMMUNITY` | 커뮤니티 요약 (선택) | `extensions/community.py` |
| `TOOL_CALL` / `REASONING` / `OUTCOME` / `SESSION` | 에이전트 활동 로그 | `activity.py` |

전체 목록은 `NodeKind` enum에 있음 — 도메인 특화 kind는 `DomainProfile.ontology_hints`
로 카테고리에 매핑.

### 1.3 EdgeKind (자동 생성되는 관계)

| EdgeKind | 방향 | 자동 생성 시점 |
|----------|------|----------------|
| `PART_OF` | Document → Category | `document_ingester` |
| `CONTAINS` | Document → Chunk | `document_ingester` |
| `NEXT_CHUNK` | Chunk → Chunk | `document_ingester` |
| `RELATED` | Entity → Entity | `table_ingester` / `db_ingester` (FK) |
| `MENTIONS` | Entity → Source | `entity_linker` (선택) |
| `EXTRACTED_FROM`, `IS_A`, `CAUSED`, `RESULTED_IN`, … | 필요 시 | `add()` 또는 수동 |

**모두 인제스트 시점에 LLM 없이 구조적으로 생성.** PPR은 kind별 가중치
(`CAUSED` 1.0, `PART_OF` 0.7, … `NEXT_CHUNK` 0.3)로 전파 세기를 조절합니다.

---

## 2. 인제스트 레이어

[`src/synaptic/extensions/`](../src/synaptic/extensions/)

### 2.1 엔트리 포인트 — `SynapticGraph.from_data()`

```python
graph = await SynapticGraph.from_data("./my_data/")
```

내부 단계:

1. **포맷 감지**: 확장자(CSV/JSONL/PDF/DOCX/PPTX/XLSX/HWP/TXT/MD)별 로더
   선택 ([`doc_loader.py`](../src/synaptic/extensions/doc_loader.py))
2. **DomainProfile 생성**: 프로파일이 없으면 `profile_generator`가 3-tier
   (rule → classifier → LLM)로 자동 작성
3. **Ingest 디스패치**:
   - 텍스트 파일 → `document_ingester` → Category(CONCEPT) → Document(ENTITY) → Chunk(CHUNK)
   - 표 형식 → `table_ingester` → 컬럼 타입 추정 + row-per-ENTITY + FK 엣지
   - DB → `db_ingester` (PostgreSQL / MySQL / SQLite / Oracle / MSSQL 5종 자동)
4. **Index 구축**: SQLite FTS5, usearch HNSW, chunk-entity 링크
5. **엔티티 추출** (선택): `phrase_extractor`(한국어/영어) + `entity_linker`
   (DF 필터 phrase hub + MENTIONS)

### 2.2 CDC — Live DB 동기화

[`extensions/cdc/`](../src/synaptic/extensions/cdc/)

프로덕션 DB와 연결할 때 매번 전체 재빌드 대신 변경분만 반영.

```python
# 첫 호출: full load + sync state 기록 (deterministic ID)
graph = await SynapticGraph.from_database(dsn, mode="cdc")

# 이후 호출: added / updated / deleted 만 반영 (X2BEE 19,843행 → 6초)
await graph.sync_from_database(dsn)
```

- 전략: `updated_at`류 컬럼 있으면 timestamp watermark, 없으면 행 content
  blake2b hash fallback
- 삭제 감지: TEMP TABLE + LEFT JOIN (메모리 부담 0)
- FK 재계산: row의 FK가 바뀌면 `RELATED` 엣지 자동 rewire
- 정합성 보장: `mode="cdc"`와 `mode="full"`이 동일 top-k 반환
  (regression test로 잠금)

dialects: SQLite · PostgreSQL · MySQL/MariaDB. Oracle/MSSQL은 legacy full-reload만.

### 2.3 DomainProfile

[`extensions/domain_profile.py`](../src/synaptic/extensions/domain_profile.py)

도메인 특화를 **코드 없이 TOML로**:

```toml
name = "fashion_ecommerce"
locale = "ko"
stopwords_extra = ["상품", "제품"]

[ontology_hints]
"상품" = "ENTITY"
"리뷰" = "OBSERVATION"

[authority_by_kind]
RULE = 10
DECISION = 7

[table_query_hints]        # v0.17.1 — 정형 corpus FTS augmentation
"sizes" = ["사이즈", "size"]
"sales_partners" = ["판매 파트너", "partner"]
```

없어도 동작합니다. `profile_generator`가 자동 생성.

---

## 3. 검색 파이프라인 — EvidenceSearch (3세대)

[`extensions/evidence_search.py`](../src/synaptic/extensions/evidence_search.py)

```
Query
  │
  ├─ Step 0  Query embedding (BYO embedder, 선택)
  ├─ Step 1  QueryAnchorExtractor → keywords / entities / categories
  │
  ├─ Step 2a FTS seed      (BM25 + Kiwi 형태소 + title 3× boost)     ← SQLite FTS5
  ├─ Step 2b Vector seed   (usearch HNSW, cascade)
  ├─ Step 2c Vector PRF    (top-3 임베딩 평균 → 2차 검색)
  │
  ├─ Step 3  PPR graph propagation  (seeds × edge-kind weight, damping 0.85)
  ├─ Step 3b GraphExpander          (1-hop: category siblings · doc→chunk ·
  │                                  chunk_next · MENTIONS · RELATED)
  │
  ├─ Step 4  HybridReranker (4축 융합)
  │            0.45 · lexical  +  0.25 · semantic
  │          + 0.20 · graph    +  0.10 · structural
  │          (reason prior: seed 1.00 / doc_chunk 0.70 / chunk_next 0.55 /
  │           related 0.50 / entity_mention 0.50 / ppr 0.45 / sibling 0.40)
  │
  ├─ Step 4b Cross-encoder reranker (BYO TEI/Ollama, top-20만)
  │          — v0.17.1: `_table_name` 노드는 skip (structured rows 에선
  │            cross-encoder가 FTS 랭킹을 오히려 망가뜨림)
  │          — adaptive blend: effective = base · min(1, std/3)
  │            (variance-gated; AutoRAG std≈0.3 → near-0, PubMedQA std≈4 → full)
  │
  └─ Step 5  EvidenceAggregator (MaxP + MMR + per-doc cap + category coverage)
             — v0.17.1: kind-aware split. `_table_name` 노드는 MMR/cap 우회
```

### 3.1 왜 FTS5 + HNSW 두 시드인가

FTS와 vector는 실패 모드가 직교합니다:

- FTS 강점: 정확한 키워드, 한국어 조사 분리 (Kiwi), 짧은 정형 쿼리
- Vector 강점: 패러프레이즈, 다국어, 의역
- PRF가 둘 사이의 어휘 간극을 메움

### 3.2 왜 PPR + GraphExpander 둘 다인가

- **PPR** = 수치 기반 전파 (damping=0.85, 2-hop)
- **GraphExpander** = 명시적 의미 경로 (category sibling, doc→chunk, …)

양쪽이 낸 `reason` 태그가 reranker의 graph score 항에 prior로 들어갑니다.

### 3.3 v0.17.1 핵심 개선

측정 기반으로 다음 3가지가 **최초로 Full pipeline 평균 > FTS-only 평균**
(0.647 vs 0.615, +5.2 %)을 만들어냄:

1. **Kind-aware aggregator** — 정형 행은 MMR/cap 우회
2. **Cross-encoder skip on `_table_name`** — bge-reranker-v2-m3가 structured
   에선 uniform logit을 내서 FTS를 망가뜨리는 현상 회피
3. **Adaptive blend (std/3)** — 리랭커 discriminative power로 블렌딩 자동 조정
4. **`table_query_hints` augmentation** — assort Easy +0.096, Conv +0.047

자세한 배경: [PLAN-v0.18-architecture.md §1](PLAN-v0.18-architecture.md).

---

## 4. 에이전트 루프 — 측정상 가장 강한 모드

[`src/synaptic/agent_loop.py`](../src/synaptic/agent_loop.py) ·
[`agent_tools.py`](../src/synaptic/agent_tools.py) ·
[`agent_tools_v2.py`](../src/synaptic/agent_tools_v2.py) ·
[`agent_tools_structured.py`](../src/synaptic/agent_tools_structured.py)

### 4.1 왜 agent loop가 주인공인가

v0.17.1 측정 (Qwen3.5-27B vLLM, 5 턴, LLM-judge):

| 벤치 | Single-shot MRR | Agent solved |
|------|----------------:|-------------:|
| assort Hard | **0.000** | **30/33 (91 %)** |
| X2BEE Hard | 0.379 | **19/19 (100 %)** |
| KRRA Hard | 0.583 | 30/39 (77 %) |
| **평균 (6 벤치)** | ~0.30 | **140/172 = 81.4 %** |

Single-shot 파이프라인이 못 푸는 hard/conversational 질의를, 에이전트가
도구를 갈아끼며 2~5턴에 풉니다. Synaptic의 진짜 narrative는
"single-shot 보통, agent 모드 평균 81 %".

### 4.2 구조

```
User query
  │
  ▼
graph.chat(query, llm_client, model, max_turns=5)
  │
  ├─ build_graph_context()        ← 카테고리·테이블·FK·도구 힌트 주입
  │   (search_session.py)
  │
  ├─ LLM (OpenAI-compatible client) 이 tool_calls 결정
  │
  ├─ Tool dispatch
  │     knowledge_search · deep_search · compare_search · expand · follow
  │     filter_nodes · aggregate_nodes · join_related · get_document ·
  │     count · list_categories · search_exact · session_info · …
  │
  ├─ SearchSession (턴 간 상태)
  │     seen_node_ids · queries_tried · categories_explored · facts
  │     budget_tool_calls (무한 루프 방지)
  │
  └─ 최종 답변 생성
```

### 4.3 36 MCP 도구

[`src/synaptic/mcp/server.py`](../src/synaptic/mcp/server.py)

| 분류 | 수 | 대표 |
|------|:-:|------|
| Knowledge CRUD | 8 | search, add, link, reinforce, stats, export, consolidate, backfill |
| Ingest / CDC | 6 | add_document, add_table, add_chunks, ingest_path, remove, sync_from_database |
| Agent workflow | 4 | start_session, log_action, record_decision, record_outcome |
| Semantic search | 3 | find_similar, get_reasoning_chain, explore_context |
| Ontology | 2 | define_type, query_schema |
| Agent v1 | 8 | search, expand, get_document, list_categories, count, search_exact, follow, session_info |
| Agent v2 | 2 | deep_search, compare_search |
| Structured | 3 | filter_nodes, aggregate_nodes, join_related |

`synaptic-mcp --db knowledge.db [--embed-url ...] [--source-dsn ...]`로 기동.

---

## 5. 백엔드

[`src/synaptic/backends/`](../src/synaptic/backends/)

| 백엔드 | 벡터 검색 | 그래프 순회 | 규모 | 의존성 |
|--------|-----------|-------------|------|--------|
| `MemoryBackend` | cosine (numpy) | BFS | ~1만 | 없음 |
| `SqliteGraphBackend` ★ | **usearch HNSW** | recursive CTE | ~10만 | `aiosqlite` |
| `KuzuBackend` | HNSW | native Cypher | ~1천만 | `kuzu` |
| `PostgreSQLBackend` | pgvector HNSW | recursive CTE | ~100만 | `asyncpg` + `pgvector` |
| `CompositeBackend` | Qdrant 등 조합 | — | 무제한 | 조합 |
| `QdrantBackend` (선택) | Qdrant | — | 벡터 전용 | `qdrant-client` |
| `MinIOStore` (선택) | — | — | 오브젝트 스토어 | `minio` |

★ = 기본 추천. `pip install synaptic-memory[sqlite]` 한 줄.

모두 동일한 `StorageBackend` Protocol을 구현:
`save_node`, `save_edge`, `search_fts`, `search_vector`, `get_neighbors`, …
([`protocols.py`](../src/synaptic/protocols.py)).

### 5.1 왜 SQLite가 기본인가

1. 설치 제로 — 그래프 DB 서버 띄울 필요 없음
2. FTS5가 강력 (BM25 · 한국어 · title boost · substring)
3. usearch HNSW가 100× 빠름 (11s → 1ms, v0.12 측정)
4. Synaptic의 쿼리 패턴은 "시드 → 1-hop" — 복잡한 Cypher 불필요

더 큰 규모가 필요하면 Kuzu / PostgreSQL / Composite로 교체.

---

## 6. 확장 지점 (BYO)

| 역할 | Protocol | 예시 구현 |
|------|----------|-----------|
| Embedder | `EmbeddingProvider` | Ollama, TEI, OpenAI, BGE-M3, HyDE |
| Cross-encoder | `CrossReranker` | TEI `bge-reranker-v2-m3`, Ollama, ColBERT |
| LLM reranker | `LLMReranker` | v0.17.0+ |
| Classifier | `KindClassifier` | rule → ontology → LLM chain |
| Query rewriter | `QueryRewriter` | 선택 |
| Query decomposer | `QueryDecomposer` | v0.17+ (multi-hop) |
| Relation detector | `RelationDetector` | LLM 버전 선택 |
| Chunker | `DocumentChunker` | BYO (document_ingester에 주입) |
| Tag extractor | `TagExtractor` | LLM / regex |

전부 `protocols.py` 에 정의. torch 직접 의존 0 — 외부 HTTP 엔드포인트로
주입하는 것이 기본 경로입니다.

---

## 7. 유지 기능 — Hebbian / Consolidation / Resonance

[`hebbian.py`](../src/synaptic/hebbian.py) ·
[`consolidation.py`](../src/synaptic/consolidation.py) ·
[`resonance.py`](../src/synaptic/resonance.py)

초기 설계(v0.5)의 뇌-신경 메타포에서 온 세 메커니즘은 **주 검색 경로
밖에서 보조 역할**로 유지됩니다.

### 7.1 Hebbian Learning

`graph.reinforce([node_ids], success=True/False)`:

- 노드 쌍 간 `weight += 0.1` (성공) / `-= 0.15` (실패)
- `weight` 범위 [-2.0, 5.0] (anti-resonance 지원)
- 에이전트가 `record_outcome` 도구로 학습 신호를 쏠 때 사용

### 7.2 Memory Consolidation (L0 → L3)

`graph.consolidate()` / `maintain()`:

```
L0_RAW     (72h TTL)  ── 72h 내 3회 접근 → L1
L1_SPRINT  (90d TTL)  ── 10회 접근 → L2
L2_MONTHLY (365d TTL) ── 성공 10회 + 성공률 80%+ → L3
L3_PERMANENT          영구 보존
```

에이전트 활동 로그 / 학습된 지식의 수명 관리에만 사용. 인제스트된 문서/
테이블 데이터는 L0에서 시작해 접근 패턴에 따라 승격되지만, 실전에서는
대부분 consolidate를 돌리지 않고 사용합니다.

### 7.3 4축 Resonance Scoring

`ActivatedNode.resonance` 필드는 유지되지만, 기본 검색 경로에서는
EvidenceSearch의 aggregated score로 재사용됩니다. 원래 공식
(`0.40·relevance + 0.25·importance + 0.20·recency + 0.15·vitality`)은
legacy `HybridSearch` ([`search.py`](../src/synaptic/search.py))에서만 적용.

v0.15부터 `search(engine='evidence')`가 기본. `HybridSearch`는 backward-compat
경로로 남아 있습니다.

---

## 8. 데이터 흐름 예시

### 8.1 `from_data()` → `chat()`

```
./my_data/
  ├─ contracts/*.pdf       ─┐
  ├─ products.csv           ├── from_data()
  └─ sales_db.sqlite       ─┘       │
                                     ▼
       doc_loader → document_ingester  ──┐
       table_ingester                    ├──▶  Nodes + Edges
       db_ingester (CDC-aware)          ─┘     (CONCEPT/ENTITY/CHUNK)
                                               (PART_OF/CONTAINS/
                                                NEXT_CHUNK/RELATED)
                                                    │
                                                    ▼
                                       SqliteGraphBackend
                                       (FTS5 + usearch HNSW)
                                                    │
                                                    ▼
                                          graph.chat("...", ...)
                                                    │
                                    build_graph_context()  →  System prompt
                                                    │
                                    LLM ◀─────────┐ │
                                      │           │ ▼
                                      └─ tool_call ─▶  Tool dispatch
                                                         (36 tools)
                                                         │
                                                    SearchSession
                                                         │
                                                         ▼
                                                  Final answer
```

### 8.2 Live DB 동기화

```
PostgreSQL (X2BEE prod)
      │
      ▼ from_database(dsn, mode="cdc")    ← full load + state 기록 (35s)
knowledge.db + syn_cdc_state

      ▼ sync_from_database(dsn)            ← 이후 호출 (6s, 19,843 rows)
      │
   timestamp watermark (updated_at) 또는 hash fallback
   ──▶  added / updated / deleted 분리
   ──▶  RELATED 엣지 rewire (FK 변경 시)
   ──▶  검색 품질 정합성 (full과 동일 top-k)
```

---

## 9. 평가와 관측

[`eval/run_all.py`](../eval/run_all.py) 기반 14-벤치 회귀 테스트.
베이스라인 + 스냅샷 메타는 [`eval/baselines/qa_latest.json`](../eval/baselines/qa_latest.json)
에 hash 인라인 — stale 의심 시 즉시 재측정 가능.

FTS-only 모드(`--quick`)를 CI/일상 회귀에, Full pipeline 모드
(`--quick --local-bge`)를 릴리스 전에 사용합니다. Agent 모드는
Qwen3.5-27B vLLM 환경에서 별도 측정.

현재 미해결 과제 (v0.18 트랙):

- AutoRAG −0.100 regression (cross-encoder가 retrieval-style에 구조적 해로움)
- KRRA Conv −23pp agent regression (한국어 conversational reasoning)
- MuSiQue R@5 0.453 vs HippoRAG2 0.747 (triple extraction 필요)

자세한 진단: [PLAN-v0.18-architecture.md](PLAN-v0.18-architecture.md).

---

## 참조

- 개념 설명 ([CONCEPTS.md](CONCEPTS.md))
- 튜토리얼 ([TUTORIAL.md](TUTORIAL.md), [TUTORIAL.en.md](TUTORIAL.en.md))
- 가이드 ([GUIDE.md](GUIDE.md))
- 로드맵 ([ROADMAP.md](ROADMAP.md))
- 타 라이브러리 비교 ([COMPARISON.md](COMPARISON.md))
- v0.18 설계 질문 ([PLAN-v0.18-architecture.md](PLAN-v0.18-architecture.md))
- 최신 릴리스 ([RELEASE_NOTES_v0.16.0.md](RELEASE_NOTES_v0.16.0.md), [CHANGELOG.md](../CHANGELOG.md))
