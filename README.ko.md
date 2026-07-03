# Synaptic Memory

**기본 경로는 인덱싱 API 호출 0회. 인프라 0. 락인 0.**
LLM 에이전트용 지식 그래프 + MCP 도구 서버. 하이브리드 검색, CDC 기반 실시간 DB 동기화, 한국어 FTS 내장.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> [English README](README.md)

---

## 5분 만에 시작

```bash
pip install "synaptic-memory[sqlite,korean,vector]"
synaptic-quickstart --db quickstart.db
```

위 두 줄로 작은 SQLite 기반 그래프를 만들고 3개 쿼리를 실행합니다. 전 과정
**LLM 호출 0회**입니다. `--db`를 빼면 의존성 없는 인메모리 smoke test로
실행됩니다. 확장 예제: [`examples/quickstart.py`](examples/quickstart.py).

---

## 왜 그냥 RAG가 아닌가?

일반 RAG는 독립적인 chunk에서 답을 찾는 경우가 많습니다. Synaptic은 먼저
그래프를 만들기 때문에 에이전트가 검색하고, 관계를 따라가고, 정형 row를
조회하고, 어떤 evidence가 도움이 됐는지 기억할 수 있습니다.

| 일반 RAG | Synaptic Memory |
|----------|-----------------|
| chunk + vector search | 문서, chunk, row, edge |
| 데이터 변경 시 재빌드가 흔함 | live DB용 CDC sync |
| 기본은 single-shot retrieval | 멀티턴 탐색용 MCP 도구 |
| feedback은 인덱스 밖에 있음 | 선택적 memory event, feedback, health signal |

vector DB 대체재가 아닙니다. 기존 문서, SQL 데이터, embedding endpoint,
agent runtime 주변에 붙는 graph/tool layer입니다.

---

## 그래프 구축과 검색

```python
import asyncio
from synaptic import SynapticGraph

async def main():
    # 아무 데이터 → 지식 그래프 (CSV, JSONL, 디렉터리)
    graph = await SynapticGraph.from_data("./내_데이터/", preset="rag")
    try:
        result = await graph.search("내 질문")
        print(result.nodes[0].node.title if result.nodes else "결과 없음")
    finally:
        await graph.close()

asyncio.run(main())
```

파일 형식 또는 DB 스키마 자동 감지, 온톨로지 프로파일 자동 생성, 인제스트, 인덱싱, FK 엣지 구축까지 전부 자동.

자주 쓰는 옵션은 preset으로 줄일 수 있습니다.

```python
# local: 기본값, 외부 서비스 없음
graph = await SynapticGraph.from_chunks(chunks, preset="local")

# rag: SYNAPTIC_EMBED_URL / SYNAPTIC_RERANK_URL 환경변수를 읽음
graph = await SynapticGraph.from_data("./docs/", preset="rag")

# agent: rag + 멀티턴 탐색용 deterministic component bridging
graph = await SynapticGraph.from_data("./docs/", preset="agent")
```

> **라이브 DB 동기화 (CDC)** — `mode="cdc"`로 증분 업데이트:
> `updated_at`류 컬럼이 있으면 워터마크 필터로 읽고, 없으면 row 내용 해시로 폴백.
> 삭제는 TEMP TABLE + LEFT JOIN으로 감지, FK 변경 시 RELATED 엣지 재연결.
> **CDC 모드와 전체 재빌드가 동일 top-k를 반환함을 regression test로 잠금.**
> SQLite, PostgreSQL, MySQL/MariaDB 지원.

> **오피스 파일(PDF/DOCX/PPTX/XLSX/HWP)** 은 **선택 패키지** `xgen-doc2chunk`를 통해 지원합니다. `pip install synaptic-memory[docs]`로 설치하거나, 자체 파서로 청킹한 결과를 `from_chunks()`로 넘기세요.

---

## 이 라이브러리가 하는 일

```
내 데이터 (CSV, JSONL, PDF/DOCX/PPTX/XLSX/HWP, SQL 데이터베이스)
  ↓  형식 자동 감지 / DB 스키마+FK 자동 발견
  ↓  DocumentIngester (텍스트) / TableIngester / DbIngester
  ↓
지식 그래프
  ├─ 문서: Category → Document → Chunk
  └─ 정형: 테이블 row → ENTITY 노드 + RELATED 엣지 (FK)
  ↓
MCP 도구 → LLM 에이전트가 그래프 기반 멀티턴으로 탐색
```

**라이브러리가 하는 건 딱 두 가지:**
1. **그래프를 잘 구축한다** — 기본은 비용 없는 deterministic extraction
2. **LLM에게 좋은 도구를 쥐어준다** — 판단은 LLM이, 코드는 데이터만

---

## 설치

```bash
# 일반 로컬 그래프 + MCP 조합
pip install "synaptic-memory[sqlite,korean,vector,mcp]"

# 팀/프로덕션 그래프: PostgreSQL + pgvector
pip install "synaptic-memory[postgresql,embedding,reranker]"

# 스케일아웃 보조 구성: Kuzu 그래프 + Qdrant 벡터 + MinIO blob
pip install "synaptic-memory[scale]"

# LangChain retriever 예제를 실행할 때 추가
pip install "synaptic-memory[langchain]"

# 또는 Postgres / Kuzu / Qdrant / MinIO 까지 전부
pip install "synaptic-memory[all]"
```

<details>
<summary>옵션별 설치</summary>

```bash
pip install synaptic-memory                # 코어 (의존성 0, 인메모리만)
pip install synaptic-memory[sqlite]        # + SQLite FTS5 백엔드
pip install synaptic-memory[korean]        # + Kiwi 한국어 형태소 분석
pip install synaptic-memory[vector]        # + usearch HNSW 벡터 인덱스
pip install synaptic-memory[mcp]           # + Claude MCP 서버
pip install synaptic-memory[embedding]     # + 임베딩 API (aiohttp)
pip install synaptic-memory[reranker]      # + flashrank cross-encoder
pip install synaptic-memory[langchain]     # + LangChain retriever 어댑터
pip install synaptic-memory[postgresql]    # + asyncpg + pgvector
pip install synaptic-memory[mysql]         # + aiomysql DB 인제스트
pip install synaptic-memory[oracle]        # + oracledb DB 인제스트
pip install synaptic-memory[mssql]         # + aioodbc DB 인제스트
pip install synaptic-memory[kuzu]          # + 임베디드 property graph 백엔드
pip install synaptic-memory[qdrant]        # + Qdrant 벡터 helper
pip install synaptic-memory[minio]         # + MinIO/S3 호환 blob helper
pip install synaptic-memory[scale]         # + Kuzu + Qdrant + MinIO + aiohttp
pip install synaptic-memory[docs]          # + PDF/DOCX/PPTX/XLSX/HWP 로더
```

</details>

---

## 인프라 연계

기본 one-liner는 로컬 SQLite 그래프를 만듭니다. 이미 쓰는 운영 인프라에
붙일 때는 backend를 직접 만들고 연결한 뒤 `from_data()`, `from_chunks()`,
`from_database()`에 넘기면 됩니다.

```python
from synaptic import SynapticGraph
from synaptic.backends.postgresql import PostgreSQLBackend

backend = PostgreSQLBackend("postgresql://user:pass@host:5432/synaptic")
await backend.connect()

graph = await SynapticGraph.from_data("./docs/", backend=backend, preset="rag")
```

현재 backend 역할:

| 경로 | 설치 | 데이터 담당 | 적합한 상황 |
|------|------|-------------|-------------|
| 로컬 앱/노트북 | `sqlite,korean,vector` | SQLite FTS5 + 로컬 usearch HNSW | 빠른 도입, 데모, 작은 서비스 |
| 팀 서비스 | `postgresql,embedding,reranker` | PostgreSQL + pgvector + pg_trgm | 공유 그래프, 백업, SQL 운영 |
| 그래프 중심 임베디드 | `kuzu,korean,embedding` | Kuzu property graph | 로컬 graph traversal / Cypher workflow |
| 스케일아웃 조합 | `scale` | Kuzu 등 graph store + Qdrant + MinIO | graph/vector/blob 책임 분리 |

Qdrant와 MinIO는 단독 그래프 저장소가 아니라 helper service입니다.
`CompositeBackend`를 통해 사용합니다. graph storage는 node/edge를 갖고,
Qdrant는 ANN vector search를 담당하며, MinIO/S3 호환 저장소는 큰
`Node.content`를 외부 blob으로 분리합니다.

```python
from synaptic.backends.composite import CompositeBackend
from synaptic.backends.kuzu import KuzuBackend
from synaptic.backends.minio_store import MinIOBackend
from synaptic.backends.qdrant import QdrantBackend

backend = CompositeBackend(
    KuzuBackend("synaptic.kuzu"),
    vector=QdrantBackend("http://localhost:6333", collection="synaptic"),
    blob=MinIOBackend("localhost:9000", bucket="synaptic"),
)
await backend.connect()

graph = await SynapticGraph.from_data("./docs/", backend=backend, preset="scale")
```

라이브러리는 backend contract와 retrieval layer를 제공합니다. 다만 수 TB급
운영 코퍼스에서는 별도 운영 레이어도 같이 설계해야 합니다. 예를 들면 durable
ingestion queue, parser/OCR worker, 외부 lexical index, tenant/ACL filter,
index lag 모니터링, 각 저장소별 backup/restore가 필요합니다.

## 빠른 시작

### 방법 A: 2줄 (가장 쉬움)

```python
import asyncio
from synaptic import SynapticGraph

async def main():
    # CSV 파일
    graph = await SynapticGraph.from_data("products.csv")
    try:
        result = await graph.search("내 질문")
        for activated in result.nodes[:5]:
            print(activated.node.title, activated.activation)
    finally:
        await graph.close()

asyncio.run(main())
```

`preset="rag"`를 넘기면 `SYNAPTIC_EMBED_URL`, `SYNAPTIC_RERANK_URL`을 읽습니다.
여러 `from_data()`, `from_chunks()`, `from_database()` 호출에 같은 설정을 쓰고
싶다면 `GraphBuildOptions`를 사용할 수 있습니다.

### 방법 B: MCP 서버 (Claude Desktop / Code)

```bash
synaptic-mcp --db my_graph.db
synaptic-mcp --db my_graph.db --embed-url http://localhost:11434/v1
```

Claude가 MCP 도구로 그래프를 직접 탐색합니다. 검색, 인제스트, CDC 동기화까지 CLI로 내려가지 않고 대화 안에서.

복붙 가능한 `claude_desktop_config.json` 샘플:
[`examples/mcp_claude_desktop.json`](examples/mcp_claude_desktop.json).

### 방법 BX: LangChain retriever로 바로 꽂기

```bash
pip install "synaptic-memory[sqlite,korean,vector,langchain]"
```

```python
import asyncio
from synaptic import SynapticGraph
from synaptic.integrations.langchain import SynapticRetriever

async def main():
    graph = await SynapticGraph.from_data("./docs/")
    try:
        retriever = SynapticRetriever(graph=graph, k=5)
        docs = await retriever.ainvoke("내 질문")
        for doc in docs:
            print(doc.page_content[:80], "   ", doc.metadata["score"])
    finally:
        await graph.close()

asyncio.run(main())
```

실행 예제: [`examples/langchain_retriever.py`](examples/langchain_retriever.py).
각 hit이 LangChain `Document`로 변환되고 metadata에 node_id, title, score, 정형
속성이 모두 담깁니다 — RetrievalQA 체인·에이전트·RAG 그래프 어디서든 그대로 사용.

### 방법 C: 세밀한 제어

```python
import asyncio
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.document_ingester import DocumentIngester, JsonlDocumentSource

async def main():
    profile = DomainProfile.load("my_profile.toml")
    backend = SqliteGraphBackend("graph.db")
    await backend.connect()

    source = JsonlDocumentSource("docs.jsonl", "chunks.jsonl")
    ingester = DocumentIngester(profile=profile, backend=backend)
    await ingester.ingest(source)

asyncio.run(main())
```

---

## 인덱싱 비용 비교

| 방식 | 인덱싱 시 LLM | 트레이드오프 |
|------|---------------|---------------|
| GraphRAG 계열 (MS GraphRAG, Cognee, Graphiti) | LLM으로 엔티티 + 관계 + 커뮤니티 요약 추출 | 서사형 코퍼스에서 recall 최상. 대신 문서 추가마다 LLM 토큰 비용 |
| LightRAG 계열 | LLM 호출을 쿼리 시점으로 지연 | 인덱스 비용 낮음. 대신 쿼리마다 비용 |
| **Synaptic 기본 경로** | **없음.** 구조·통계 시그널만 (FK, NEXT_CHUNK, phrase DF 허브, MENTIONS) | 비용 0 + 결정론적. 명시적 cross-reference를 LLM 없이 엣지화 |

기본 인덱싱은 LLM-free입니다. 그래프는 지식 저장소가 아니라 검색 인덱스입니다.
문서에 명시된 cross-reference는 LLM 없이 `REFERENCES` 엣지로 만들 수 있습니다.
OpenIE를 opt-in하면 bounded/revertible 방식으로 LLM 기반 semantic relation을
추가할 수 있지만, 기본 deterministic 경로에는 포함되지 않습니다.

> **현재 API**: `graph.search()`는 하나의 경로만 사용합니다. BM25 + HNSW +
> PPR + cross-encoder + MMR 기반 EvidenceSearch 파이프라인입니다.
> 예전 `engine=` 스위치는 제거되었으므로 예제는
> `graph.search("질문")`처럼 바로 호출하면 됩니다.

---

## 에이전트 도구

### 텍스트 검색 도구
| 도구 | 용도 |
|------|------|
| `deep_search` | **추천.** 검색 → 확장 → 문서 읽기를 한 번에 |
| `compare_search` | 복합 질문 자동 분해 + 병렬 검색 |
| `knowledge_search` | EvidenceSearch 기반 핵심 의미 검색 |
| `agent_search` | FTS + 벡터 하이브리드 + intent routing |
| `expand` | 1-hop 그래프 이웃 탐색 |
| `get_document` | 쿼리 관련 청크만 선별한 문서 전문 |
| `search_exact` | ID/코드 정밀 매칭 (BM25 우회) |
| `follow` | 특정 엣지 타입 순회 |

### 정형 데이터 도구
| 도구 | 용도 |
|------|------|
| `filter_nodes` | 속성 필터 (>=, <=, contains) — `{total, showing}` 반환으로 카운팅 정확 |
| `aggregate_nodes` | GROUP BY + COUNT/SUM/AVG/MAX/MIN + WHERE 사전 필터 |
| `join_related` | FK 기반 관련 레코드 조회 — RELATED 엣지 순회 (O(degree)) |
| `top_nodes` | “가장 X한”, “top N”, “최대/최소”, “최근” 질의를 단일 호출로 처리 |

### 인제스트 / CDC 도구
대화 중에 Claude가 새 자료를 배울 수 있도록 하는 도구.

| 도구 | 용도 |
|------|------|
| `knowledge_add_document` | 긴 텍스트를 자동 청킹해 그래프에 추가 |
| `knowledge_add_table` | 컬럼+행 리스트를 ENTITY + FK 엣지로 인제스트 |
| `knowledge_add_chunks` | BYO-chunker 경로 |
| `knowledge_ingest_path` | 로컬 CSV/JSONL/TXT 파일 단건 인제스트 |
| `knowledge_remove` | 단건 노드 삭제 (엣지 cascade) |
| `knowledge_sync_from_database` | CDC 증분 동기화 |
| `knowledge_backfill` | 누락된 임베딩·phrase 허브 복구 |

### 탐색 도구
| 도구 | 용도 |
|------|------|
| `list_categories` | 카테고리 목록 + 문서 수 |
| `count` | 종류/카테고리/연도별 카운트 |
| `session_info` | 멀티턴 세션 상태 조회 |

모든 도구는 `{ data, hints, session }` 형태로 반환. `SearchSession`이 턴 간 상태를 추적하므로 같은 청크를 두 번 읽지 않습니다.

---

## 검색 파이프라인

```
쿼리
  ↓  Kiwi 형태소 분석 (한국어) 또는 정규식 (기타)
  ↓  BM25 FTS + title 3배 가중치 + substring fallback
  ↓  벡터 검색 (usearch HNSW, 선택)
  ↓  Vector PRF (유사 관련 피드백, 2-pass)
  ↓  PPR 그래프 탐색 (PersonalizedPageRank)
  ↓  GraphExpander (1-hop: 카테고리 형제, 다음 청크, 엔티티 멘션)
  ↓  HybridReranker (어휘 + 의미 + 그래프 + 구조 + 메모리 + 권위 + 시간)
  ↓  MaxP 문서 집계 (커버리지 보너스)
  ↓  Cross-encoder reranker (bge-reranker-v2-m3, 선택)
  ↓  EvidenceAggregator (MMR 다양성 + 문서당 캡 + 카테고리 커버리지)
결과
```

**사용·시간 메모리 축 (opt-in, 기본 off).** reranker에 다섯 번째 가중 신호
`memory` — 각 노드를 *어떻게 사용됐는지*로 점수화한다: 중요도(강화된 성공 vs 실패),
최신성(`updated_at`), 활력(vitality). `memory=0.0`(기본)이면 랭킹은 그대로다. 켜면
검색이 *진화*한다 — 쿼리에 답이 된 결과를 reinforce하면 다음 검색에서 올라가고, decay된
노드는 가라앉는다. 정적 인덱스는 구조적으로 못 하는 일이다.

```python
from synaptic.extensions.hybrid_reranker import RerankerWeights

# 메모리 축 활성화 (나머지 가중치는 합이 ~1 이 되도록 재조정)
graph.reranker_weights = RerankerWeights(
    lexical=0.35, semantic=0.20, graph=0.10, structural=0.10, memory=0.25,
)
await graph.reinforce([node_id], success=True)  # 이 결과가 도움 됨 → 다음엔 상위로
```

### Memory operating layer

검색을 항상 stateful하게 만들지 않고도 관찰할 수 있습니다.

```python
from synaptic import FeedbackSignal, MemoryScope

scope = MemoryScope(workspace_id="docs", user_id="alice")
result = await graph.search("환불 예외", record=True, scope=scope)

await graph.record_feedback(
    event_id=result.event_id,
    signal=FeedbackSignal.EXPLICIT_POSITIVE,
    success=True,
    scope=scope,
)

health = await graph.memory_health(scope=scope)
signals = await graph.scan_memory_signals(scope=scope)
```

event, feedback, provenance, health signal은 그래프 metadata로 저장됩니다.
`Node.content`에 섞지 않고, LLM prompt에 원본 metadata 전체를 자동으로 밀어 넣지도 않습니다.

---

## 벤치마크와 보고서

루트 README는 현재 설치 경로와 공개 API를 기준으로 유지합니다. 상세 수치는
오래된 측정값이 현재 API 계약처럼 보이지 않도록 버전이 붙은 보고서에 둡니다.

빠른 로컬 smoke:

```bash
synaptic-quickstart --json
```

가벼운 한국어 FTS 벤치마크:

```bash
pip install "synaptic-memory[korean]"
python examples/benchmark_allganize.py
```

선택 패키지/API key가 준비된 경우 competitor harness:

```bash
python examples/benchmark_vs_competitors/run_comparison.py --only synaptic
```

참고 보고서:

| 보고서 | 내용 |
|--------|------|
| [docs/comparison/synaptic_results.md](docs/comparison/synaptic_results.md) | 재현 가능한 Synaptic 벤치마크 결과와 provenance |
| [docs/REPORT-rag-vs-synaptic.md](docs/REPORT-rag-vs-synaptic.md) | 금융 법령 multi-hop 검색에서 RAG와 synaptic-memory 비교 |
| [docs/REPORT-memory-operating-layer-eval.md](docs/REPORT-memory-operating-layer-eval.md) | memory operating layer 평가와 health/reporting gate |
| [examples/benchmark_vs_competitors/README.md](examples/benchmark_vs_competitors/README.md) | competitor adapter 공정성 주의사항 |

---

## 아키텍처

```
SynapticGraph.from_data("./data/")          ← Easy API
  ↓
자동 감지 → DomainProfile → 인제스트 → 인덱싱
  ↓
StorageBackend (Protocol)
  ├── MemoryBackend        (테스트용)
  ├── SqliteGraphBackend   (권장, FTS5 + HNSW)
  ├── KuzuBackend          (임베디드 Cypher)
  ├── PostgreSQLBackend    (pgvector)
  └── CompositeBackend     (백엔드 조합)
  ↓
검색 파이프라인 (BM25 + 벡터 + PRF + PPR + reranker + MMR)
  ↓
에이전트 도구 → MCP 서버 → LLM 에이전트
```

---

## 백엔드

| 백엔드 | 설치 옵션 | 역할 | 용도 |
|--------|-----------|------|------|
| `MemoryBackend` | core | 인프로세스 그래프 | 테스트와 예제 |
| `SqliteGraphBackend` | `sqlite`, `vector` | 로컬 그래프 + FTS5 + usearch HNSW | 기본 로컬/임베디드 배포 |
| `KuzuBackend` | `kuzu` | 임베디드 property graph + Cypher | 그래프 중심 로컬 workflow |
| `PostgreSQLBackend` | `postgresql` | durable graph + pgvector + pg_trgm | 공유 프로덕션 서비스 |
| `QdrantBackend` | `qdrant` | vector-only helper | `CompositeBackend` 뒤 ANN search |
| `MinIOBackend` | `minio` | blob-only helper | `CompositeBackend` 뒤 큰 content offload |
| `CompositeBackend` | `scale` | graph + vector + blob store 라우터 | 스케일아웃 조합 |

---

## 선택 옵션

| 옵션 | 추가 기능 |
|------|----------|
| `sqlite` | aiosqlite 백엔드 (실사용 기본) |
| `korean` | Kiwi 한국어 형태소 분석기 |
| `vector` | usearch HNSW 인덱스 |
| `embedding` | 임베딩 API 호출용 aiohttp |
| `reranker` | flashrank cross-encoder |
| `mcp` | Claude Desktop/Code MCP 서버 |
| `langchain` | LangChain retriever 어댑터 |
| `postgresql` | asyncpg + pgvector |
| `mysql` | aiomysql DB 인제스트 |
| `oracle` | oracledb DB 인제스트 |
| `mssql` | aioodbc DB 인제스트 |
| `kuzu` | 임베디드 Kuzu graph 백엔드 |
| `qdrant` | Qdrant vector helper |
| `minio` | MinIO/S3 호환 blob helper |
| `scale` | Kuzu + Qdrant + MinIO + aiohttp |
| `rag` | spaCy + aiohttp endpoint helper |
| `all` | 주요 DB, vector, MCP, 한국어, reranker 옵션 묶음 |
| `docs` | PDF/DOCX/PPTX/XLSX/HWP 문서 로더 (xgen-doc2chunk) |

---

## 문서

| 문서 | 내용 |
|------|------|
| [docs/GUIDE.md](docs/GUIDE.md) | 친절한 전체 안내서 (처음 접하는 사람용) |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | 30분 단계별 실습 |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | 파이프라인 심화 설명 |
| [docs/ADOPTION.md](docs/ADOPTION.md) | 설치 옵션, preset, 첫 적용 경로 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 신경망 영감 초기 설계 |
| [docs/COMPARISON.md](docs/COMPARISON.md) | GraphRAG / LightRAG 등과 비교 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 역사적 로드맵 |

## 개발

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q
uv run ruff check --fix
```

## 라이선스

Apache-2.0 — [LICENSE](LICENSE) 참조. 출처(copyright/attribution notice)만 보존하면
상용·수정·재배포 모두 자유.
