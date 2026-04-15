# Synaptic Memory 안내서

> "아무 데이터나 넣어주세요. LLM 에이전트가 이해할 수 있는 지식 그래프로 만들어 드릴게요."

이 문서는 **Synaptic Memory를 처음 접하는 사람**을 위한 안내서입니다.
프로그래밍 지식이 있으면 좋지만 RAG, 벡터 DB, 온톨로지 같은 용어를 몰라도
끝까지 따라올 수 있도록 작성했습니다.

---

## 1. 한 줄 요약

**LLM이 검색·추론할 수 있는 지식 그래프를, 당신이 가진 데이터로 2줄 만에 만들어 주는 Python 라이브러리입니다.**

```python
from synaptic import SynapticGraph

graph = await SynapticGraph.from_data("./my_data/")   # 2줄
result = await graph.search("내가 궁금한 것")
```

---

## 2. 왜 이런 게 필요한가요?

### LLM을 제대로 쓰려면 "관련 정보"를 먼저 찾아야 합니다

ChatGPT에게 회사 내부 문서를 질문하려면 어떻게 해야 할까요? 문서를 통째로
붙여넣으면 너무 길어서 대부분 잘려 버립니다. 그래서 **질문과 관련 있는
부분만 골라서** 프롬프트에 넣어주는 게 일반적인 방법입니다. 이걸 RAG
(Retrieval-Augmented Generation, 검색 증강 생성)이라고 부릅니다.

### 기존 RAG의 문제점

1. **키워드 매칭만으론 부족합니다**
   "스마트폰"으로 검색했는데 문서에 "휴대전화"만 있으면 못 찾습니다.

2. **단일 검색으론 복합 질문을 못 풉니다**
   "5점 리뷰가 가장 많이 달린 상품의 가격은?" 같은 질문은 1번의 검색으로
   답할 수 없습니다. 여러 단계의 조회·필터·집계가 필요하죠.

3. **정형 데이터(DB)와 비정형 데이터(문서)가 분리돼 있습니다**
   상품 목록은 PostgreSQL에, 설명서는 PDF에 있는데, 둘을 함께 검색하려면
   직접 코드를 짜야 합니다.

4. **LLM으로 데이터를 전처리하면 돈이 너무 듭니다**
   문서 10만 개를 인덱싱할 때마다 GPT-4를 호출하면... 지갑이 울어요.

### Synaptic Memory가 해결하는 방식

- **하이브리드 검색**: 키워드(BM25) + 의미(임베딩) + 그래프 구조를 모두 씁니다.
- **멀티턴 에이전트**: LLM이 도구를 여러 번 호출하면서 답을 찾아갑니다.
- **정형/비정형 통합**: CSV, 문서, DB가 **하나의 그래프**에 들어갑니다.
- **인덱싱은 LLM 없이**: 전처리에 LLM을 안 써서 비용이 거의 0입니다.

---

## 3. "지식 그래프"가 도대체 뭔가요?

데이터를 **노드(점)** 와 **엣지(선)** 로 표현한 구조입니다.

예를 들어 쇼핑몰 데이터가 있다고 해 봅시다:

```
    ┌────────────┐         ┌────────────┐
    │  iPhone 15 │◄────────│  판매이력    │
    │   (상품)    │  FK     │  #SH0001    │
    └─────┬──────┘         └────────────┘
          │
          │ FK
          ▼
    ┌────────────┐
    │   리뷰       │
    │  (5점)       │
    └────────────┘
```

- **노드**: iPhone 15, 판매이력 #SH0001, 리뷰 (각각의 레코드)
- **엣지**: "판매이력은 iPhone 15에 대한 것" (외래키 관계)

Synaptic Memory는 당신의 데이터를 이렇게 그래프로 만들어 두고, 검색할 때
**연결된 노드들을 함께 고려**합니다. "iPhone 판매량이 얼마?"를 물으면
iPhone 노드를 찾은 다음 판매이력 엣지를 따라가서 집계합니다.

---

## 4. 어떻게 작동하나요?

### 4-1. 데이터 → 그래프 (인제스트)

```
당신의 데이터
  ↓
  ├─ CSV / JSONL → TableIngester
  ├─ 텍스트 문서 → DocumentIngester
  └─ SQL DB      → DbIngester (SQLite/PostgreSQL/MySQL/Oracle/MSSQL)
  ↓
지식 그래프 (SQLite에 저장)
  ├─ 문서 그래프: Category → Document → Chunk
  └─ 정형 그래프: Table Row → ENTITY 노드 + RELATED 엣지(FK)
```

자동으로:
- 데이터 형식 감지
- 외래 키 관계 감지 → RELATED 엣지 생성
- M:N 조인 테이블 감지 → 중간 노드 없이 직접 엣지
- FTS 인덱스 + (선택적으로) 벡터 인덱스 구축

**라이브 DB라면 CDC 모드** — 매번 풀로드 대신 변경분만 그래프에 반영합니다.
같은 DSN을 `mode="cdc"`로 호출하면 deterministic 노드 ID와 sync state가
그래프 SQLite 파일에 함께 저장되고, 이후 `sync_from_database()`는 워터마크
이후 변경된 행만 읽어 ms 단위로 끝냅니다. 자세한 동작은 CONCEPTS의
"CDC: 라이브 데이터베이스 동기화" 섹션을 참고하세요.

```python
graph = await SynapticGraph.from_database(dsn, db="knowledge.db", mode="cdc")
result = await graph.sync_from_database(dsn)  # 두 번째 호출부터 증분
print(result.added, result.updated, result.deleted)
```

### 4-2. 그래프 → 검색 (Retrieval)

3세대 GraphRAG 파이프라인:

```
사용자 쿼리
  ↓
  1. 쿼리 분석: 카테고리/엔티티/키워드 anchor 추출
  ↓
  2. FTS 시드 (BM25 + Kiwi 형태소 분석)
  ↓
  3. 벡터 시드 (usearch HNSW, 선택)
  ↓
  4. Vector PRF: top-3 결과 임베딩 평균으로 2차 검색
  ↓
  5. PPR (PersonalizedPageRank): 그래프 엣지로 2-hop 전파
  ↓
  6. GraphExpander: 1-hop 이웃 (카테고리 형제, 청크 연속, FK 관련)
  ↓
  7. HybridReranker: lexical + semantic + graph + structural 점수 융합
  ↓
  8. Cross-encoder reranker (선택, bge-reranker-v2-m3)
  ↓
  9. EvidenceAggregator: MMR 다양성 + 문서당 cap
  ↓
결과
```

### 4-3. 검색 → 에이전트 (Multi-turn)

LLM이 단일 검색으로 답 못 하면 **여러 도구를 조합**합니다:

```
쿼리: "5점 리뷰 가장 많이 받은 상품의 가격은?"
  ↓
Turn 1: aggregate_nodes(
          table="reviews",
          group_by="product_id",
          metric="count",
          where_property="score",
          where_op="==",
          where_value="5"
        )
  → G00857 (11건)

Turn 2: get_document("products:G00857")
  → Flour - Masa De Harina Mexican, 5,000원
```

2턴 만에 답을 찾습니다. 기존 RAG는 이런 질문을 풀 수 없습니다.

---

## 5. 핵심 철학

### 5-1. "코드는 데이터, 판단은 LLM"

전통적 검색 시스템은 쿼리 파싱, 의도 분류, 재작성 같은 걸 **코드로** 합니다.
Synaptic Memory는 다릅니다:

- **코드**: 그래프 구축, 검색 인덱스, 도구 제공만 담당
- **LLM**: "어떤 도구를 어떤 순서로 쓸지" 직접 판단

왜? 자연어 판단은 LLM이 훨씬 잘하기 때문입니다. 코드로 분류기를 짜면
새 도메인에 대응할 때마다 수정해야 하지만, LLM은 프롬프트만 바꾸면 됩니다.

### 5-2. "BYO: Bring Your Own Embedder/Reranker"

임베딩 모델·리랭커를 라이브러리에 심지 **않습니다**. 대신 프로토콜을 정의해
두고, 사용자가 원하는 걸 주입하게 합니다:

```python
graph = await SynapticGraph.from_data(
    "./data/",
    embed_url="http://localhost:11434/v1",  # Ollama
    embed_model="qwen3-embedding:4b",
)
```

장점:
- **torch 의존성 0**: 설치 용량 최소
- **모델 교체 자유**: OpenAI, Ollama, TEI, llama.cpp 등 OpenAI 호환이면 뭐든
- **비용 제어**: 사용자가 직접 선택

### 5-3. "인덱싱에 LLM 비용 0원"

1세대 GraphRAG (Microsoft)는 인덱싱 때 LLM으로 엔티티·관계를 추출해서
**10만 토큰당 $수십** 수준입니다. Synaptic Memory는:

- 엔티티는 FTS + 빈도 기반으로 추출
- 관계는 외래 키, 카테고리, 청크 순서로 자동 구축
- 임베딩만 선택적으로 사용 (그것도 BYO)

문서 1만 개 인덱싱에 API 호출 0회 가능합니다.

---

## 6. 주요 구성 요소

### SynapticGraph — 메인 파사드
```python
from synaptic import SynapticGraph

graph = await SynapticGraph.from_data("./data/")             # 파일 (CSV/JSONL/PDF/DOCX/...)
graph = await SynapticGraph.from_database("postgres://...")  # DB (한 번만 로드)
graph = await SynapticGraph.from_database("postgres://...",  # 라이브 DB CDC
    db="knowledge.db", mode="cdc")
result = await graph.sync_from_database("postgres://...")    # 두 번째부터 증분
graph = await SynapticGraph.from_chunks(my_chunks)           # 직접 청킹한 결과
```

> **PDF/DOCX/PPTX/XLSX/HWP** 같은 오피스 파일은 선택 패키지 `xgen-doc2chunk`로
> 처리합니다. `pip install synaptic-memory[docs]`로 활성화하거나, 자체 파서로
> 청크를 만들어 `from_chunks()`에 넘겨도 됩니다.

### 36개 에이전트 도구

| 분류 | 도구 | 용도 |
|------|------|------|
| **텍스트 검색** | `deep_search` | 검색→확장→문서읽기 한 번에 (추천) |
| | `knowledge_search` | 기본 시맨틱 검색 (v0.14.2부터 EvidenceSearch 라우트) |
| | `compare_search` | 복합 쿼리 자동 분해 |
| **그래프 탐색** | `expand` | 노드의 1-hop 이웃 |
| | `follow` | 특정 엣지 타입만 순회 |
| | `get_document` | 문서 전문 + 관련 청크 |
| **정형 데이터** | `filter_nodes` | SQL WHERE (total/showing 반환) |
| | `aggregate_nodes` | GROUP BY + COUNT/SUM/AVG + WHERE 사전 필터 |
| | `join_related` | FK 기반 조인 (그래프 엣지 순회) |
| **인제스트 / CDC** (v0.14.0+) | `knowledge_add_document` | 자동 청킹 + phrase hub 링크 |
| | `knowledge_add_table` | 컬럼/행 → ENTITY 노드 + FK 엣지 |
| | `knowledge_add_chunks` | 사전 청킹된 결과 (BYO chunker) |
| | `knowledge_ingest_path` | 로컬 CSV/JSONL/TXT 파일 |
| | `knowledge_remove` | 단건 삭제 (엣지 cascade) |
| | `knowledge_sync_from_database` | 라이브 DB CDC 증분 동기화 |
| | `knowledge_backfill` | 기존 그래프 embedding/phrase hub 복구 (v0.14.4+) |
| **네비게이션** | `list_categories` | 카테고리 목록 |
| | `count` | 종류/카테고리별 카운트 |
| | `search_exact` | ID/코드 정확 매칭 |

### 5개 백엔드

| 백엔드 | 벡터 검색 | 규모 | 추천 상황 |
|--------|----------|------|----------|
| `MemoryBackend` | cosine | ~10K | 테스트/프로토타입 |
| `SqliteGraphBackend` | **usearch HNSW** | ~100K | **기본 권장** |
| `KuzuBackend` | HNSW | ~1천만 | 대규모 그래프 |
| `PostgreSQLBackend` | pgvector | ~100만 | 프로덕션 |
| `CompositeBackend` | Qdrant | 무제한 | 스케일아웃 |

---

## 7. 벤치마크 결과

> 최신 FTS-only 베이스라인 + corpus snapshot hash는
> [`eval/baselines/qa_latest.json`](../eval/baselines/qa_latest.json)의
> `_meta` 블록에. v0.14.x 이후 검색 코드가 여러 번 바뀌어서
> embedder/reranker 모드는 별도 재측정이 필요한 상태.

### 단일 검색 (v0.13.0 시점 embedder + reranker 측정값)

| 데이터셋 | 유형 | 노드 수 | MRR |
|---------|------|---------|-----|
| KRRA Hard (한국 공공기관) | 문서 | 19,720 | **1.000** (15/15) |
| X2BEE Easy (PostgreSQL) | 정형 | 19,843 | **1.000** (20/20) |
| HotPotQA-24 (영어 multi-hop) | 문서 | 226 | **0.964** |
| Allganize RAG-ko | 문서 | 200 | **0.905** |
| PublicHealthQA | 문서 | 77 | **0.600** |

### 멀티턴 에이전트 (GPT-4o-mini, 최대 5턴)

| 데이터셋 | 결과 |
|---------|------|
| **X2BEE Hard agent** | **17/19 (89%)** |
| **assort Hard agent** | **12/15 (80%)** |
| KRRA Hard agent | 10-13/15 |

X2BEE Hard는 시작 시점 1/19 (5%) → **17/19 (89%)** 까지 개선됐습니다.
정형 데이터 질의(필터/집계/FK 조인)가 그래프 기반 도구로 end-to-end 동작합니다.

---

## 8. 어디서 시작해야 하나요?

### 처음 써 보는 분
→ [`TUTORIAL.md`](TUTORIAL.md): 30분 안에 따라할 수 있는 단계별 튜토리얼

### 내부 동작이 궁금한 분
→ [`CONCEPTS.md`](CONCEPTS.md): 3세대 GraphRAG, 검색 파이프라인, 그래프 구조 심화

### 기존 아키텍처 문서
→ [`ARCHITECTURE.md`](ARCHITECTURE.md): Hebbian Learning, Memory Consolidation 등 초기 설계
→ [`COMPARISON.md`](COMPARISON.md): 다른 GraphRAG 라이브러리와의 비교
→ [`ROADMAP.md`](ROADMAP.md): 향후 로드맵

### 빠른 레퍼런스
→ [`../README.md`](../README.md): 2줄 설치, API 예제

### MCP 서버로 Claude에 붙이고 싶은 분
```bash
synaptic-mcp --db my_graph.db --embed-url http://localhost:11434/v1
```
Claude Desktop/Code에서 36개 도구를 쓸 수 있게 됩니다.

---

## 9. 자주 묻는 질문

**Q. RAG와 뭐가 달라요?**
A. RAG는 보통 "임베딩으로 top-k 검색 → LLM에 주입" 한 단계입니다.
Synaptic Memory는 그래프 구조 + 멀티턴 탐색 + 구조적 쿼리까지 지원합니다.

**Q. 벡터 DB 없이 되나요?**
A. 네. 기본은 SQLite FTS5 + usearch (HNSW) 만으로 10만 노드까지 충분합니다.
벡터 없이 BM25만 써도 돌아갑니다.

**Q. 한국어는 지원되나요?**
A. Kiwi 형태소 분석기 내장으로 한국어 FTS 품질이 매우 높습니다.
KRRA Hard MRR 1.000이 그 증거입니다.

**Q. 그래프 DB가 필요한가요?**
A. 아니요. SQLite로 충분합니다. 엣지 테이블을 자체적으로 관리합니다.
Kuzu (임베디드 그래프 DB)도 선택적으로 지원합니다.

**Q. 비용이 얼마나 드나요?**
A. 인덱싱 비용 0원 (LLM 미사용). 검색 시 임베딩 API 호출 (옵션),
에이전트 쿼리 시 LLM 토큰 (직접 호출). Ollama로 로컬 모델을 쓰면 완전 무료.

**Q. 데이터가 업데이트되면?**
A. `mode="cdc"`로 인제스트하면 두 번째 호출부터 변경분만 동기화됩니다.
X2BEE 프로덕션 PostgreSQL (19,843행) 검증 결과: full reload 35초 vs CDC
incremental sync **6초** (~6× 빠름). 검색 품질은 동일 (regression test로
잠금). SQLite / PostgreSQL / MySQL 지원.

**Q. CDC를 켜면 검색 결과가 달라지나요?**
A. 아니요. CDC는 노드 ID 생성 방식만 바꿉니다 (random UUID → deterministic
hash). 검색 알고리즘은 그대로입니다. `tests/test_cdc_search_regression.py`가
같은 데이터를 `mode="full"`과 `mode="cdc"`로 빌드해서 top-k가 일치하는지
매번 확인합니다.

**Q. 프로덕션에서 쓰기 안전한가요?**
A. v0.15.0은 Beta 단계입니다. 809개 단위 테스트 통과, 프로덕션 PostgreSQL(X2BEE) CDC 검증 완료.
중요 데이터는 백업을 권장합니다.

---

## 10. 더 읽어보기

- [CONCEPTS.md](CONCEPTS.md) — 3세대 GraphRAG·검색 파이프라인 심화
- [TUTORIAL.md](TUTORIAL.md) — 단계별 실습
- [ARCHITECTURE.md](ARCHITECTURE.md) — 원래 설계 문서 (신경망 영감)
- [COMPARISON.md](COMPARISON.md) — GraphRAG / LightRAG / LazyGraphRAG 비교
- [GitHub Issues](https://github.com/PlateerLab/synaptic-memory/issues) — 버그 리포트 / 피드백
- [CHANGELOG.md](../CHANGELOG.md) — 버전별 변경 이력

---

**MIT License · 작성자: Son Seongjun**
