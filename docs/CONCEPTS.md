# Synaptic Memory — 핵심 개념

이 문서는 Synaptic Memory가 어떻게, 왜 그렇게 만들어졌는지를 설명합니다.
[GUIDE.md](GUIDE.md)가 "무엇이냐"라면 이건 "어떻게·왜"입니다.

---

## 1. 3세대 GraphRAG란 무엇인가

GraphRAG는 지금까지 3번 크게 발전했습니다.

### 1세대 — GraphRAG (Microsoft, 2024 초)

```
문서 → LLM으로 엔티티·관계 추출 → 그래프 → 커뮤니티 요약 → 검색
              └ $$$$ ──┘
```

**장점**: 의미 있는 관계를 LLM이 직접 추출하니 품질이 높음
**단점**:
- 인덱싱에 LLM 호출 다량 → 10만 토큰당 수십 달러
- 새 문서가 들어오면 재요약 필요
- 관계 추출이 불안정 (동일 문서를 여러 번 돌리면 다른 결과)

### 2세대 — LightRAG / LazyGraphRAG (2024 말)

```
문서 → 간단한 엔티티만 추출 → 검색 시점에 LLM 호출
```

**개선점**: 인덱싱 비용을 뒤로 미뤄서 총 비용 감소
**한계**: 여전히 LLM 기반 그래프 구조. 쿼리 때 비용이 쌓임.

### 3세대 — Relation-Free Graph + Hybrid Retrieval (Synaptic Memory가 채택)

**핵심 발상**: 관계(edge)를 LLM으로 뽑지 말고, 구조적으로 이미 있는 것을
쓰자. 그러면 LLM 비용 0원.

```
문서/테이블 → 자동 그래프 구축 (FK, 카테고리, 청크 순서) → 하이브리드 검색
              └ LLM 0회 ──┘
```

**구조적 그래프 = LLM 없이 얻을 수 있는 관계**:
- **외래 키** (테이블 → 테이블)
- **카테고리** (문서 → 카테고리)
- **청크 순서** (청크 → 다음 청크)
- **문서 포함** (문서 → 청크)

이 4가지 관계만으로도 대부분의 질의를 풀 수 있다는 게 **LinearRAG, Practical
GraphRAG** 같은 최근 연구의 결론입니다. "shallow expansion is enough"라고
불립니다.

**부족한 부분은 하이브리드 검색으로 메움**:
- BM25 (키워드 매칭)
- 벡터 임베딩 (의미 유사)
- PPR (그래프 전파)
- Cross-encoder (정밀 재정렬)

이 네 가지를 합쳐 점수를 내면 LLM 없이도 1세대 GraphRAG에 필적하는 품질이
나옵니다.

---

## 2. 그래프 구조 심화

### 2-1. 노드 종류 (NodeKind)

| NodeKind | 용도 | 예시 |
|----------|------|------|
| `CONCEPT` | 카테고리 / 분류 | "규정 및 지침", "ESG" |
| `DOCUMENT` | 문서 전체 | "2024년 경영 계획서" |
| `CHUNK` | 문서 조각 | Doc의 512토큰 단위 |
| `ENTITY` | 정형 데이터 행 / 추출 엔티티 | `products:12800000` |
| `RULE` / `DECISION` / `OBSERVATION` | 지식 타입 (선택) | 초기 설계의 잔재 |

실제로는 **문서 그래프 (CONCEPT → DOCUMENT → CHUNK)** 와
**정형 그래프 (ENTITY)** 가 병렬로 존재할 수 있습니다. 한 그래프에 섞어
저장해도 됩니다.

### 2-2. 엣지 종류 (EdgeKind)

| EdgeKind | 방향 | 의미 | 자동 생성 시점 |
|----------|------|------|---------------|
| `PART_OF` | Document → Category | 문서가 카테고리에 속함 | DocumentIngester |
| `CONTAINS` | Document → Chunk | 문서가 청크를 포함 | DocumentIngester |
| `NEXT_CHUNK` | Chunk → Chunk | 다음 청크 (순서) | DocumentIngester |
| `RELATED` | Entity → Entity | FK 관계 | TableIngester / DbIngester |
| `MENTIONS` | Entity → Source | 엔티티가 언급됨 | EntityLinker (선택) |

모두 **인제스트 시점에 자동 생성**됩니다. LLM이 관여하지 않습니다.

### 2-3. 엣지 가중치

PPR (PersonalizedPageRank)이 엣지 가중치를 활용합니다:

```python
_EDGE_WEIGHTS: dict[EdgeKind, float] = {
    EdgeKind.CAUSED: 1.0,       # 강한 인과
    EdgeKind.RESULTED_IN: 1.0,
    EdgeKind.PART_OF: 0.7,      # 중간
    EdgeKind.CONTAINS: 0.6,
    EdgeKind.MENTIONS: 0.5,
    EdgeKind.RELATED: 0.4,      # 약함 (노이즈 방지)
    EdgeKind.NEXT_CHUNK: 0.3,   # 최소 (청크→청크 노이즈 방지)
}
```

같은 노드라도 어떤 엣지로 도달했느냐에 따라 점수가 다릅니다.

---

## 3. 검색 파이프라인 단계별 해부

[`src/synaptic/extensions/evidence_search.py`](../src/synaptic/extensions/evidence_search.py)

쿼리 1회에 대해 실제로 일어나는 일:

### Step 0 — 쿼리 임베딩 (선택)
```python
query_embedding = await embedder.embed(query)  # BYO
```
`embed_url`이 지정됐을 때만 실행. 벡터 검색의 기반.

### Step 1 — 쿼리 분석 (QueryAnchorExtractor)
```python
anchors = QueryAnchors(
    query="말 복지 향상 프로그램",
    keywords=["말", "복지", "향상"],
    entities=[],
    categories=["복지 및 교육"],   # ← 카테고리 추정
    category_node_ids=["cat_welfare_id"],
)
```

DomainProfile의 `ontology_hints`를 이용해 쿼리가 어느 카테고리를 건드리는지
추정합니다. 카테고리 매칭이 강력한 단서가 됩니다.

### Step 2a — FTS 시드
```python
fts_hits = await backend.search_fts(query, limit=30)
```
- BM25 점수
- 한국어면 Kiwi 형태소 분석 적용
- 제목은 3배 가중치
- LIKE fallback (한국어 부분 매칭)

### Step 2b — 벡터 시드
```python
vec_hits = await backend.search_vector(query_embedding, limit=30)
```
- usearch HNSW 인덱스 (수 ms)
- cascade: 임베딩 있는 노드만

### Step 2c — Vector PRF (Pseudo Relevance Feedback)
```python
# top 3 결과의 임베딩 평균을 새 쿼리로 2차 검색
refined_emb = mean(top3.embedding)
vec_hits_2 = await backend.search_vector(refined_emb, limit=30)
```

쿼리와 데이터의 어휘 차이를 극복하는 기법. 첫 검색이 약해도 두 번째가
보강합니다.

### Step 3 — PPR 그래프 전파
```python
ppr_scores = await personalized_pagerank(
    backend,
    seeds={nid: fts_score for nid, fts_score in fts_hits},
    damping=0.85,
    top_k=k * 3,
)
```

시드 노드에서 엣지 타입 가중치로 2-hop 전파. 직접 매칭 안 된 노드도
그래프 거리가 가까우면 후보에 포함됩니다.

### Step 4 — GraphExpander (1-hop 확장)
```python
expanded = await expander.expand(anchors=anchors, seed_nodes=all_seeds)
```

5가지 경로로 이웃을 추가:
1. **Category siblings**: 같은 카테고리의 다른 문서
2. **Document chunks**: 문서의 자식 청크
3. **Chunk neighbours**: 인접 청크 (NEXT_CHUNK)
4. **Entity mentions**: 엔티티 허브 → 언급 문서
5. **RELATED** (v0.13+): 정형 데이터의 FK 관련 노드

각 노드에 `reason` 태그가 붙어서 이후 reranker가 활용합니다.

### Step 5 — HybridReranker (4축 점수 융합)
```python
final_score = (
    0.45 * lexical_score    # BM25 정규화
    + 0.25 * semantic_score  # 벡터 유사도
    + 0.20 * graph_score     # expansion reason 기반
    + 0.10 * structural_score  # 카테고리/종류/authority/temporal
)
```

`reason` prior 예시:
```python
"seed": 1.00,           # FTS 직접 매칭
"document_chunk": 0.70, # 같은 문서 내
"chunk_next": 0.55,     # 다음 청크
"related": 0.50,        # FK 관련 (v0.13+)
"entity_mention": 0.50,
"ppr_discovery": 0.45,
"category_sibling": 0.40,
```

### Step 5b — Cross-encoder reranker (선택)
```python
scores = await reranker.rerank(query, [ev.content for ev in top20])
```

쿼리-문서 쌍을 직접 점수화. bge-reranker-v2-m3를 TEI로 배포한 경우 호출.
top 20개만 재정렬해서 비용을 제한합니다.

### Step 6 — EvidenceAggregator (MMR + per-doc cap)
```python
diverse = mmr(candidates, lambda_=0.7)  # 관련성 vs 다양성
```

같은 문서에서 5개 이상 안 뽑고, 카테고리별로 최소 1개는 확보하는 등
다양성 보장. 단일 문서가 결과를 도배하는 걸 방지합니다.

---

## 4. 멀티턴 에이전트 설계

### 4-1. 도구 중심 설계

코드가 "판단"하지 않습니다. LLM이 도구를 선택합니다:

```
Agent System Prompt + Graph Metadata
  ↓
LLM (GPT-4o-mini / Claude)
  ↓
도구 호출 (JSON tool_calls)
  ↓
도구 실행 (synaptic-memory 내부)
  ↓
결과 → 다시 LLM
  ↓
(반복)
  ↓
최종 답변
```

### 4-2. Graph Metadata 주입

`build_graph_context()`가 시스템 프롬프트 앞에 자동으로 붙습니다:

```
[Graph metadata]
Categories (10): 규정 및 지침(235), 운영계획(315), ESG(198), ...
Total nodes: 19,720

[Structured data — tables and columns for filter/aggregate/join]
Table: pr_goods_base (1008 rows)
  goods_no: e.g. G00001, G00002
  goods_nm: e.g. iPhone 15 Pro, Shin Ramyun
  sales_prc: e.g. 1600000.0, 5000.0

[Foreign key relationships]
  pr_goods_sold_hist.goods_no → pr_goods_base
  pr_goods_user_feedback.goods_no → pr_goods_base

[Graph composition — match tool to data type]
- Structured rows: 19843 → use filter_nodes/aggregate_nodes/join_related
```

LLM은 이 메타데이터를 보고:
- 어떤 카테고리가 있는지 (검색에 활용)
- 어떤 테이블이 있는지 (filter/aggregate 대상)
- FK가 어떻게 연결됐는지 (join_related 경로)
- **데이터 유형이 뭔지** (문서 vs 정형 → 도구 선택)

### 4-3. 도구 선택 전략 (시스템 프롬프트)

```
## Tool selection (pick the RIGHT one first time)
- Text question → deep_search
- Price/date/attribute filter → filter_nodes
- "how many per X" / TOP N → aggregate_nodes
- "find related records" → join_related
- Find by name/text → filter_nodes with op="contains"

## Fallback when search returns 0 results
1. Try filter_nodes with op="contains" on text columns
2. Try shorter/individual keywords
3. Try translated terms (Korean ↔ English)
```

이 프롬프트가 **코드를 대체**합니다. 새 도메인이 추가돼도 프롬프트만
수정하면 됩니다.

### 4-4. SearchSession: 턴 간 상태 관리

```python
@dataclass
class SearchSession:
    session_id: str
    budget_tool_calls: int     # 호출 예산 (무한 루프 방지)
    seen_node_ids: set[str]    # 이미 본 노드 (중복 방지)
    queries_tried: list[str]   # 시도한 쿼리
    categories_explored: set   # 탐색한 카테고리
    expanded_nodes: set        # 확장한 노드 (v0.13+)
    facts: dict                # 스크래치 패드
```

- 같은 청크를 두 번 읽지 않음
- 이미 시도한 쿼리를 재시도하지 않음
- 예산 초과 시 도구가 `budget_exceeded` 힌트 반환 → LLM이 답변 생성 모드 전환

---

## 5. "왜 SQLite인가?"

처음 보면 의아합니다. GraphRAG 라이브러리들은 보통 Neo4j, Kuzu, Neptune
같은 그래프 DB를 쓰는데 왜 SQLite?

### 5-1. 설치 제로

`pip install synaptic-memory[sqlite]` 한 줄. 그래프 DB 서버 실행 필요 없음.

### 5-2. FTS5가 강력함

SQLite의 FTS5는 BM25 지원, 한국어 토큰화, substring 매칭까지 지원합니다.
별도 검색 엔진(Elasticsearch 등)이 필요 없습니다.

### 5-3. usearch HNSW로 벡터도 커버

벡터 검색은 usearch (100x 빠른 HNSW 인덱스)를 SQLite와 병합해서 처리.
10만 노드 규모까지 단일 `knowledge.db` 파일 하나로 해결됩니다.

### 5-4. 그래프 쿼리가 단순

Synaptic Memory의 쿼리는 모두 "시드 → 1-hop 확장" 패턴입니다. 복잡한
Cypher MATCH 패턴이 필요 없습니다. `SELECT * FROM syn_edges WHERE source_id=?
AND kind=?` 면 충분합니다.

### 5-5. 더 큰 규모가 필요할 땐?

백엔드 교체 가능:
- Kuzu (임베디드 Cypher, ~1천만)
- PostgreSQL + pgvector (~100만)
- Qdrant (벡터 전용, 무제한)

`StorageBackend` 프로토콜을 구현한 백엔드면 어떤 것도 substituable합니다.

---

## 6. DomainProfile: 도메인 특화를 코드 없이

모든 도메인마다 설정을 코드로 짜면 유지보수가 지옥입니다. 대신 TOML로:

```toml
# my_profile.toml
name = "fashion_ecommerce"
locale = "ko"

stopwords_extra = ["상품", "제품", "rows"]

[ontology_hints]
"상품" = "ENTITY"
"리뷰" = "OBSERVATION"
"브랜드" = "CONCEPT"

[authority_by_kind]
RULE = 10
DECISION = 7
OBSERVATION = 3
```

- `stopwords_extra`: 해당 도메인에서 노이즈인 단어
- `ontology_hints`: 카테고리 → NodeKind 매핑
- `authority_by_kind`: 랭킹에 영향 (권위 있는 문서 가산점)

**없어도 동작**합니다. ProfileGenerator가 자동 생성합니다 (3-tier: rule →
classifier → LLM).

---

## 7. "LLM 없이 인덱싱한다"의 실제 의미

오해: "그럼 모든 게 키워드 매칭이야?"
아닙니다. **LLM을 "인덱스를 만드는 데"만 안 쓴다**는 뜻입니다. 검색할 때는
LLM이 도구를 호출하면서 판단합니다.

### 인덱스 시점에 하는 일 (LLM 0회)
- 텍스트 파싱·청킹
- BM25 인덱스 구축
- 임베딩 생성 (BYO, 주입된 모델)
- 엔티티 후보 추출 (빈도 + 길이 + 형태소)
- 카테고리 분류 (rule-based)
- 그래프 엣지 구축 (FK, 순서, 포함)

### 검색 시점에 쓰는 LLM (사용자 비용)
- 에이전트가 도구 선택
- 최종 답변 생성

사용자가 `query()` 1회 호출하면 LLM 호출 수는 **에이전트 턴 수 + 답변 1회**
입니다. 보통 1~5회.

---

## 8. 확장 지점 (Extension Points)

Synaptic Memory는 레고 블록처럼 조합 가능합니다:

### 백엔드 교체
```python
class MyBackend:
    async def save_node(self, node): ...
    async def save_edge(self, edge): ...
    # StorageBackend Protocol 구현
```

### 임베더 교체
```python
class MyEmbedder:
    async def embed(self, text: str) -> list[float]: ...
```

### 리랭커 교체
```python
class MyReranker:
    async def rerank(self, query, docs) -> list[float]: ...
```

### 도구 추가
```python
async def my_tool(backend, session, **kwargs) -> ToolResult:
    ...
```

MCP 서버에 바로 등록 가능.

### DomainProfile
TOML로 도메인 특화.

### Classifier
`ClassifierChain(rule, ontology, llm)`로 3-tier 분류 커스터마이즈.

---

## 9. 라이브러리가 피하는 것들

- **torch / transformers 직접 의존**: BYO로 해결, 설치 가볍게
- **LLM으로 인덱싱**: 비용 폭발 방지
- **도메인 하드코딩**: DomainProfile로 주입
- **판단 로직을 코드에**: LLM 프롬프트로 옮김
- **복잡한 설정**: `from_data()` 한 줄로 충분하게
- **Neo4j/벡터 DB 필수**: 기본값 SQLite

---

## 10. CDC: 라이브 데이터베이스 동기화

> 프로덕션 DB와 연결할 때 매번 전체 재빌드 대신 변경분만 그래프에 반영합니다.

### 왜 필요한가

`from_database()`는 한 줄로 그래프를 만들어 주지만, 호출할 때마다 모든 행을
다시 읽고 다시 인제스트합니다. 19,843행짜리 X2BEE 프로덕션 DB로 35초.
100만 행이면 한 시간 단위가 됩니다. 라이브 DB라면 매 시간 또는 분 단위로
동기화해야 하는데, 이걸 풀로드로 처리할 수 없습니다.

CDC(Change Data Capture)는 이전 동기화 이후 **변경된 행만** 읽어 그래프에
반영합니다.

### 가장 큰 블로커: 노드 ID

기본 동작에서 `Node.id`는 `uuid4().hex[:16]` — 매 인제스트마다 새 ID가
생성됩니다. 같은 행을 두 번 읽으면 그래프에 두 개의 노드가 생기고 검색
중복이 발생합니다.

CDC는 이걸 **deterministic ID**로 해결합니다:

```python
deterministic_row_id(source_url, table, primary_key)
  = blake2b("{normalized_url}::{table}::{canonical_pk}", digest_size=8)
  → 16 hex chars (기존 UUID 너비와 동일)
```

같은 (source_url, table, pk) 조합은 항상 같은 노드 ID를 만들어 내고,
SQLite 백엔드의 `ON CONFLICT(id) DO UPDATE SET`이 즉시 upsert로 동작합니다.

### 두 개의 bookkeeping 테이블

CDC 상태는 그래프 SQLite 파일 안에 저장돼서 그래프 파일 하나로 자체
완결됩니다:

```sql
CREATE TABLE syn_cdc_state (
    source_url, table_name, strategy, change_col,
    last_sync_at, last_watermark, primary_key_col,
    row_count, schema_fingerprint,
    PRIMARY KEY (source_url, table_name)
);

CREATE TABLE syn_cdc_pk_index (
    source_url, table_name, pk,
    node_id, row_hash, fk_edges,  -- fk_edges는 JSON
    PRIMARY KEY (source_url, table_name, pk)
);
```

`syn_cdc_state`는 테이블별 워터마크와 전략을, `syn_cdc_pk_index`는 모든
소스 행의 (pk → node_id) 매핑과 FK 스냅샷을 가지고 있습니다.

### 두 개의 변경 감지 전략

| 전략 | 조건 | 비용 |
|---|---|---|
| **timestamp** (선호) | `updated_at` 같은 단조 증가 컬럼 존재 | O(변경 행 수) |
| **hash** (fallback) | 단조 컬럼 없음 | O(전체 행 수) — 매번 풀스캔 + blake2b |

자동 선택입니다. `detect_change_column()`이 컬럼 이름을 보고
`updated_at`, `modified_at`, `mtime` 등을 찾아내면 timestamp 전략, 못
찾으면 hash 전략으로 떨어집니다. 사용자가 손댈 필요 없습니다.

### 삭제 감지 — TEMP TABLE + LEFT JOIN

매 동기화마다 소스의 모든 PK를 읽어와서 transient `cdc_current_pks`
TEMP TABLE에 bulk insert한 다음 한 번의 LEFT JOIN으로 끝냅니다:

```sql
SELECT idx.pk, idx.node_id
  FROM syn_cdc_pk_index AS idx
  LEFT JOIN cdc_current_pks AS cur ON cur.pk = idx.pk
 WHERE idx.source_url = ? AND idx.table_name = ?
   AND cur.pk IS NULL
```

Python 쪽에서 set diff를 만들지 않으니 1M행짜리 테이블도 메모리가 평탄.
삭제된 노드는 `graph.remove()`로 처리되고 `ON DELETE CASCADE`가 엣지를
자동 정리합니다.

### FK 엣지 재계산

`syn_cdc_pk_index.fk_edges`에 각 행의 FK 값 스냅샷을 JSON으로 저장합니다:

```json
{"category_id": "C1", "vendor_id": "V42"}
```

행이 업데이트되면 prior 스냅샷과 새 FK 값을 diff해서 바뀐 컬럼만 옛
RELATED 엣지를 삭제하고 새 엣지를 만듭니다. 신규 엣지는
`TableIngester.ingest`가 이미 idempotent (UNIQUE source/target/kind)로
삽입하므로 별도 처리 없이 동작합니다.

### PK 없는 테이블은 명시적으로 skip

소스 스키마에 진짜 PRIMARY KEY가 없는 테이블 (AWS DMS 검증 테이블,
임시 로그 테이블 등)은 CDC 모드에서 skip합니다. 이유:

`columns[0]`로 fallback할 경우 그 컬럼이 unique가 아니면 (예: `TASK_NAME`
1개 distinct value) 모든 행이 동일한 deterministic ID로 collapse돼서
46개 행 중 45개가 사라지고 매 동기화마다 churn이 발생합니다.
`TableSchema.has_explicit_pk = False`인 테이블은 `SyncResult.tables`에
명시적 에러 항목으로 들어갑니다 — 이런 테이블은 `mode="full"`로만
인제스트하라는 신호입니다.

### API

```python
# 첫 번째 호출 — deterministic ID로 풀로드 + sync state 시드
graph = await SynapticGraph.from_database(
    "postgresql://user:pass@host/db",
    db="knowledge.db",
    mode="cdc",
)

# N번째 호출 — 변경분만
result = await graph.sync_from_database(
    "postgresql://user:pass@host/db"
)
print(result.added, result.updated, result.deleted)

# 자동 모드 — prior state 있으면 cdc, 없으면 full
graph = await SynapticGraph.from_database(dsn, mode="auto")
```

### 검색 품질에 미치는 영향

**없습니다**. CDC는 노드 ID 생성 방식만 바꿀 뿐 검색 알고리즘 (BM25,
Vector, PRF, PPR, Reranker, MMR)을 전혀 건드리지 않습니다.
`tests/test_cdc_search_regression.py`가 동일 데이터를 `mode="full"`과
`mode="cdc"`로 빌드해서 top-k가 일치하는지 매 PR마다 검증합니다.

X2BEE 프로덕션 검증 결과 (19,843행 PostgreSQL):

| 지표 | 값 |
|---|---|
| Initial CDC load | 51초 |
| Full reload baseline | 35초 |
| **Idempotent re-sync** | **6초** (~6× 빠름) |
| Search top-1 일치 | 4/4 ✓ |

### 지원되는 dialects

| DB | CDC 모드 | Notes |
|---|---|---|
| SQLite | ✅ | 1차 타깃 |
| PostgreSQL | ✅ | asyncpg 필요 |
| MySQL/MariaDB | ✅ | aiomysql 필요 |
| Oracle | legacy full-reload만 | Phase 6 follow-up |
| MSSQL | legacy full-reload만 | Phase 6 follow-up |

dialect별 placeholder 차이 (`?` vs `$1` vs `%s`)는 `_translate_placeholders`가
한 번에 처리합니다. 신규 dialect 추가는 row_reader / pk_reader 두 함수만
구현하면 됩니다.

---

## 11. Backfill: 기존 그래프를 in-place로 복구

> v0.14.x 시리즈에서 발견한 "silent failure" 패턴의 회수 경로.

### 왜 필요한가

v0.14.x 시리즈는 "기능은 코드에 있는데 wiring이 빠져서 silent하게
죽어 있던" 버그들을 여러 건 고쳤습니다:

- **v0.14.0 초기**: MCP 서버의 `_ensure_graph()`가 `ChunkEntityIndex`는
  wire했지만 `PhraseExtractor`를 빼먹어서 문서 간 phrase hub 다리가
  아예 안 만들어짐. v0.14.3에서 한 줄 fix.
- **v0.14.0 전체 기간**: 사용자가 embedder 없이 인제스트 → 나중에
  `--embed-url`로 다시 띄워도 기존 노드는 `Node.embedding=[]` 상태
  그대로. HNSW 인덱스는 비어 있고 vector 검색이 부분적으로 죽음.

두 경우 모두 **신규 인제스트만** 고쳐졌고, 이미 들어 있는 데이터는
재인제스트 말고는 복구할 방법이 없었습니다. 실전에서는 재인제스트가
비싸거나 (수십만 문서) 불가능합니다 (소스 파일이 더 이상 없음). 그래서
in-place 복구 도구가 필요했습니다.

### `graph.backfill()` — 두 가지 복구 경로

```python
result = await graph.backfill(
    embeddings=True,     # 빈 embedding 채우기
    phrases=True,        # phrase hub 누락분 재생성
    batch_size=64,
    max_nodes=None,      # None = 전체, int = 처음 N개만
)
print(result.embeddings_filled, result.phrases_linked)
```

**Pass 1 — Embedding backfill** (`embeddings=True`):
모든 노드를 walk하고 `node.embedding == []`인 것만 모아 `embedder.embed_batch()`에
batch_size씩 넘김. 성공한 결과를 `backend.update_node()`로 저장.
이미 임베딩 있는 노드는 건너뜀 (멱등성).

**Pass 2 — Phrase hub backfill** (`phrases=True`):
텍스트를 가진 노드 중 outgoing CONTAINS 엣지가 **하나도 없는** 노드만
선별 → `phrase_extractor.extract_and_link()` 재실행 → 결과로 나온
phrase hub 노드에 `CONTAINS` 엣지 생성 → `ChunkEntityIndex`에 등록.
Phrase hub 노드 자신 (태그 `_phrase`)는 건너뜀 — hub of hubs 방지.

두 pass 모두:

- **Best-effort** — 단일 행 실패는 `BackfillResult.errors`에 append만
  되고 나머지는 계속 진행. 한 노드의 임베딩 실패가 전체를 abort하지
  않음.
- **Idempotent** — 두 번 돌려도 두 번째는 `embeddings_filled=0`,
  `phrases_linked=0`. 건강한 노드는 스킵.
- **Bounded** — `max_nodes` 파라미터로 점진 처리 가능. 100만 노드
  그래프를 한 번에 처리할 수 없을 때 유용.

### `BackfillResult` — 투명한 리포트

```python
@dataclass(slots=True)
class BackfillResult:
    scanned: int = 0               # 총 inspect한 노드 수
    embeddings_filled: int = 0     # 새로 임베딩 채운 노드 수
    phrases_linked: int = 0        # 새로 만든 CONTAINS 엣지 수
    skipped_no_text: int = 0       # title/content 없어서 embed 불가
    elapsed_ms: float = 0.0        # 벽시계 시간
    errors: list[str] = []         # per-node 에러 메시지
```

### Wiring 필요 조건

`backfill()`은 그래프가 이미 필요한 컴포넌트를 wire하고 있어야 동작:

- **Embedding backfill**: `SynapticGraph(embedder=...)` 필요.
  없으면 **no-op** (에러 아님). `graph.backfill(embeddings=True)`는
  `scanned=0`을 반환하고 조용히 넘어감.
- **Phrase backfill**: `SynapticGraph(phrase_extractor=...)` 필요.
  없으면 **no-op**.

즉 백필 도구는 "없는 의존성을 상상으로 만들어내지" 않습니다. 사용자가
먼저 누락된 wiring을 고치고 (`--embed-url` 추가, 신규 그래프
생성자에서 `PhraseExtractor()` 전달) 그 다음 backfill을 호출하는
순서입니다.

### MCP tool

```json
// MCP 도구 호출 예시
{
  "tool": "knowledge_backfill",
  "scope": "all",          // "all" | "embeddings" | "phrases"
  "batch_size": 64,
  "max_nodes": null        // 전체 처리
}
```

응답:
```json
{
  "success": true,
  "scope": "all",
  "scanned": 19843,
  "embeddings_filled": 19843,
  "phrases_linked": 5612,
  "skipped_no_text": 0,
  "elapsed_ms": 42350.2,
  "errors": []
}
```

### Phrase backfill은 검색 품질을 극적으로 개선함

Phrase hub가 없는 그래프는 `GraphExpander`와 `PersonalizedPageRank`의
cross-document 탐색이 **사실상 dead path**입니다. 문서 간 연결이
없으니 PPR이 walk할 엣지가 없고, 1-hop 확장도 같은 문서 내부만
맴돕니다. 결과는 "FTS over disjoint files" — 키워드 매칭이 안 되는
의미 기반 질의에 0건 응답.

Backfill을 실행하면 `CONTAINS` 엣지가 대량 생성되면서 `ChunkEntityIndex`가
채워지고, 다음 검색부터 PPR이 실제 그래프를 돌기 시작합니다. 이 효과는
벤치마크 수치로도 바로 드러납니다 (특히 multi-hop KRRA Hard / assort Hard).

### 한계

- **재-인제스트의 완전한 대체는 아님**. 원본 소스의 스키마가 바뀌었거나
  새 컬럼이 추가됐다면 backfill은 그걸 감지하지 못합니다. 그때는 CDC
  (`sync_from_database()`)가 올바른 도구.
- **Embedder / reranker 모델을 바꾸면 재-embed 필요**. Backfill은
  "현재 wire된 embedder로 다시 돌려" 동작이기 때문에 임베더 전환 시
  `embeddings=True`로 다시 돌려야 기존 벡터가 새 모델 벡터로 교체됩니다.
  (단, 현재 구현은 `node.embedding is []`만 체크하므로 강제 재-embed를
  원하면 수동으로 비워야 함 — P3에서 `force=True` 옵션 예정.)
- **Phrase hub 품질은 extractor에 의존**. Korean vs English vs mixed
  locale에서 extractor 선택이 중요 — `create_phrase_extractor(profile)`
  경로 사용 권장.

### 설계 원칙

Backfill은 **"새 기능이 아니라 복구 도구"**입니다. v0.14.x 시리즈에서
발견한 교훈 — "feature가 wiring 누락으로 silent하게 죽어 있으면 안
된다" — 의 후속 조치. 이상적으로는 애초에 wiring이 잘 되어 있어서
backfill을 쓸 일이 없는 게 맞지만, 한번 발생한 과거를 되돌리는 경로는
제공해야 사용자가 재인제스트의 비용/불가능성에 묶이지 않습니다.

향후 발견될 새로운 silent-failure 패턴도 같은 방식으로 backfill 도구의
한 pass로 추가될 수 있습니다. 예: `fingerprint_backfill=True` (스키마
fingerprint 재계산), `category_backfill=True` (카테고리 라벨 재추출) 등.

---

## 12. 한계와 향후 방향

### 현재 한계
- **멀티홉 질의 불안정**: GPT-4o-mini 같은 작은 모델은 2-3홉부터 오판
- **패러프레이즈**: 데이터에 없는 개념어는 임베딩으로도 한계
- **평가 방식**: ID 매칭 기반 GT는 집계 쿼리에 부정확 (LLM-as-Judge 보완)
- **CDC Oracle/MSSQL**: 아직 legacy full-reload만 (Phase 6 follow-up)
- **CDC PK 없는 테이블**: 안전을 위해 skip — 실 데이터 테이블이라면
  보통 PK가 있으므로 큰 제약은 아님

### 로드맵 (ROADMAP.md)
- CDC 기반 인크리멘털 인덱싱
- Doc2Query++ 쿼리 확장
- Multi-agent 협업 탐색
- 평가 GT 자동 확장 도구

---

## 13. Measured negatives — 시도했지만 ship하지 않은 것들

v0.17.0 개발 중 MuSiQue 와 공개 벤치 5종에서 여러 메커니즘을 측정했고, **4개 접근이 품질을 악화시킨다는 증거로 확정됐다.** 향후 세션/기여자가 같은 실수를 반복하지 않도록 측정치와 함께 기록한다.

### 13.1 LLM query decomposer — MuSiQue R@5 −10.6%

가설: Multi-hop 쿼리를 LLM 으로 서브쿼리로 분해하고 (`LLMChainDecomposer`, `OpenAILLMProvider` 백엔드) 각 서브에 대해 FTS seed 를 뽑아 RRF(k=60) 로 융합하면 bridge document 가 seed pool 에 들어올 것.

측정 (MuSiQue-Ans dev 500q, bge-m3 + bge-reranker-v2-m3):
- Baseline: MRR 0.729 / R@5 0.453 / Search 476s
- With LLMChainDecomposer: MRR 0.696 / R@5 **0.405** / Search 1820s

원인: RRF 가 원본 쿼리와 서브쿼리 랭크를 동등하게 취급 (`1/(60+rank)`). 서브쿼리가 끌어온 FTS seed 대부분이 topic-drift 노이즈 (UHF 영화의 "film UHF" 서브가 UHF 방송 일반 문서를 대량 인입). Reranker 는 여전히 원본 쿼리로 스코어하므로 bridge doc 이 top-N 에서 밀림.

**현재 상태**: `QueryDecomposer` Protocol + `LLMChainDecomposer` 는 commit `2eb2b3b` 에 남아 있다 — **opt-in default-off**. Compound 쿼리 ("A 와 B 비교") 가 주 사용 패턴인 한국어 corpus 에선 positive 일 가능성 있음. Chain-reasoning 벤치에선 enable 하지 말 것.

### 13.2 Inline phrase hub (DF filter 없이) — MuSiQue R@5 −6.6%, build 15× 느림

가설: `PhraseExtractor` 를 `SynapticGraph(phrase_extractor=...)` 에 인라인으로 붙이면 shared-phrase 노드가 생겨서 문서 간 bridging edge 가 생기고, PPR 이 multi-hop doc 을 자연히 집는다.

측정:
- Baseline: build 99s, R@5 0.453
- With inline phrase_extractor: build **1534s** (15.5× slower), R@5 **0.423**

원인: `EnglishPhraseExtractor` 에 DF filter 가 없어서 "American" (샘플 1k docs 에서 130개), "United", "She" 같은 generic phrase 가 super-hub 노드로 만들어진다. PPR random walk 이 super-hub 에 teleport 된 뒤 무관한 문서로 흩어짐. **노이즈 필터링 없는 phrase hub 은 cross-document bridging 이 아니라 cross-document poisoning.**

**현재 상태**: Inline 경로는 그대로 남지만 (다른 locale/corpus 에선 유용 가능) v0.17.0 default 는 **post-hoc DF-filtered `EntityLinker`** 사용. 벤치 스크립트의 `--phrase-extractor` flag 는 진단용.

### 13.3 DF-filtered EntityLinker — 공개 벤치에서 neutral (±1%)

가설: `EntityLinker` 는 post-hoc 패스에서 `min_df=2, max_df_ratio=0.02` DF filter 를 걸어 super-hub 를 제거한다. Build 시간 15× → 1.6× 로 회복. MuSiQue 에서 R@5 0.423 → 0.435 (inline 대비 +2.8%, baseline 대비 여전히 −4%).

공개 5 벤치 교차 측정에서 EntityLinker ON/OFF 차이는 모두 ±1% 미만:
- HotPotQA-24: 1.000 → 0.979 (−2%, 24 쿼리 중 1개 차이)
- Allganize RAG-ko: 0.972 → 0.967 (−0.5%)
- Allganize RAG-Eval: 0.925 → 0.924 (neutral)
- PublicHealthQA: 0.706 → 0.706 (neutral)
- AutoRAG: 0.642 → 0.638 (neutral)

**현재 상태**: `--entity-linker` flag 로 opt-in, default-off. 대규모 corpus 에서 사용자 corpus profile 튜닝 뒤 재측정 시 positive 가능성 남김. Release scope 에서 "phrase hub 이 도움이 되는 corpus 에서만 opt-in" 으로 문서화.

### 13.4 `rerank_blend=0.4` (pre-v0.17.0 default) — AutoRAG −29%

가설: Cross-encoder rerank blend 0.4 (40% cross + 60% hybrid) 가 `bge-reranker-v2-m3` 의 paraphrase 잡는 능력을 최대화한다.

측정 (AutoRAG 114q, component isolation via `diagnose_autorag.py`):
- FTS-only: MRR 0.906 / Hit 114/114
- Embedder only (no reranker): 0.879 / 114/114 (**near-neutral**)
- Reranker only: **0.641 / 81/114** (**−29%, 33 쿼리가 top-10 밖으로**)
- Embedder + reranker: 0.642 / 80/114

Blend sweep (5 벤치 × 3 blend, `sweep_rerank_blend.py`):

| Bench | b=0.1 | b=0.2 | b=0.4 (구) | FTS-only |
|---|---:|---:|---:|---:|
| HotPotQA-24 | 0.979 | 1.000 | 1.000 | 0.875 |
| Allganize RAG-ko | **0.982** | 0.981 | 0.972 | 0.947 |
| Allganize RAG-Eval | **0.946** | 0.935 | 0.925 | 0.911 |
| PublicHealthQA | **0.734** | 0.719 | 0.706 | 0.547 |
| AutoRAG | **0.766** | 0.708 | 0.642 | 0.906 |
| **평균** | **0.881** | 0.869 | 0.849 | 0.837 |

원인: `bge-reranker-v2-m3` 는 long-form paraphrase 데이터로 튜닝됐다. Retrieval-style corpus (AutoRAG — 짧은 팩트 쿼리, 좁은 gold 셋) 에선 FTS 랭킹이 이미 최적이고 cross-encoder 의 sentence-level 재스코어가 정답을 떨어뜨린다. 0.4 blend 에서 이 파괴력이 최대.

**현재 상태**: commit `7472dc0` 에서 default 0.1 로 변경. 5 벤치 평균 +3.2pp. Retrieval-style corpus 는 여전히 FTS-only 가 ceiling (AutoRAG 0.906 vs full-pipeline 0.766). 사용자 가이드에 "corpus 유형별 reranker opt-in/opt-out" 권고.

### 13.5 교훈 한줄 요약

> **"Mechanism 추가 = 품질 개선" 이라는 전제는 corpus 유형에 따라 부합하지 않는다.**
> FTS 랭킹이 이미 near-optimal 인 corpus 에서 bridging / reranking / decomposition
> 모두 정답을 top-K 밖으로 밀어낸다. 하나를 켜기 전에 FTS-only 와 교차측정할 것.
> v0.14.4 시점 베이스라인이 4 major release 뒤에도 최신인 척 비교하면 false uplift
> narrative 가 나온다 — **항상 current-code FTS-only 를 재측정한 뒤 비교할 것.**

---

## 참고 문헌

- **GraphRAG** (Microsoft, 2024): Edge et al., "From Local to Global"
- **LightRAG** (2024): Han et al., "Simple and Fast Retrieval-Augmented Generation"
- **LazyGraphRAG** (Microsoft, 2024): Delayed LLM invocation pattern
- **LinearRAG** (2025): "Practical Graph Retrieval" — 1-hop is enough
- **HippoRAG2**: MaxP document aggregation
- **usearch**: Efficient HNSW for embedded vector search
- **BM25**: Robertson & Walker, "Okapi BM25"
- **PPR**: Haveliwala, "Topic-sensitive PageRank"

---

## 다음으로

- 실전 예제를 돌려보고 싶다면 → [TUTORIAL.md](TUTORIAL.md)
- 친절한 전체 소개가 필요하다면 → [GUIDE.md](GUIDE.md)
- 다른 라이브러리와 비교는 → [COMPARISON.md](COMPARISON.md)
