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

## 10. 한계와 향후 방향

### 현재 한계
- **인크리멘털 업데이트 없음**: 새 데이터가 오면 전체 재빌드 권장
- **멀티홉 질의 불안정**: GPT-4o-mini 같은 작은 모델은 2-3홉부터 오판
- **패러프레이즈**: 데이터에 없는 개념어는 임베딩으로도 한계
- **평가 방식**: ID 매칭 기반 GT는 집계 쿼리에 부정확 (LLM-as-Judge 보완)

### 로드맵 (ROADMAP.md)
- CDC 기반 인크리멘털 인덱싱
- Doc2Query++ 쿼리 확장
- Multi-agent 협업 탐색
- 평가 GT 자동 확장 도구

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
