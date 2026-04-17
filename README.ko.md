# Synaptic Memory

**인덱싱에 API 호출 0회. 인프라 0. 락인 0.**
LLM 에이전트용 지식 그래프 + MCP 도구 서버. 하이브리드 검색, CDC 기반 실시간 DB 동기화, 한국어 FTS 내장.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> [English README](README.md)

---

## 5분 만에 시작

```bash
pip install "synaptic-memory[sqlite,korean,vector]"
python examples/quickstart.py
```

위 두 줄로 [`examples/data/products.csv`](examples/data/products.csv)를
SQLite 기반 그래프로 인제스트하고 3개 쿼리를 실행합니다. 전 과정 **LLM 호출 0회**.
전체 소스: [`examples/quickstart.py`](examples/quickstart.py).

---

## 두 번의 호출로 그래프 구축

```python
import asyncio
from synaptic import SynapticGraph

async def main():
    # 아무 데이터 → 지식 그래프 (CSV, JSONL, 디렉터리)
    graph = await SynapticGraph.from_data("./내_데이터/")

    # 또는 DB에서 바로 — SQLite / PostgreSQL / MySQL / Oracle / MSSQL
    graph = await SynapticGraph.from_database(
        "postgresql://user:pass@host:5432/dbname"
    )

    # 라이브 DB? CDC 모드로 변경분만 다시 읽기
    graph = await SynapticGraph.from_database(
        "postgresql://user:pass@host:5432/dbname",
        db="knowledge.db",
        mode="cdc",       # deterministic node ID + sync state 기록
    )
    result = await graph.sync_from_database(
        "postgresql://user:pass@host:5432/dbname"
    )
    print(result.added, result.updated, result.deleted)

    # 또는 직접 청킹한 문서 전달 (LangChain, Unstructured, 자체 OCR 등)
    chunks = my_parser.split("manual.pdf")
    graph = await SynapticGraph.from_chunks(chunks)

    # 검색
    result = await graph.search("내 질문", engine="evidence")

asyncio.run(main())
```

파일 형식 또는 DB 스키마 자동 감지, 온톨로지 프로파일 자동 생성, 인제스트, 인덱싱, FK 엣지 구축까지 전부 자동.

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
36개 MCP 도구 → LLM 에이전트가 그래프 기반 멀티턴으로 탐색
```

**라이브러리가 하는 건 딱 두 가지:**
1. **그래프를 잘 구축한다** — 인덱싱에 LLM 비용 0원
2. **LLM에게 좋은 도구를 쥐어준다** — 판단은 LLM이, 코드는 데이터만

---

## 설치

```bash
# 추천 — README의 모든 예제가 작동하는 조합
pip install "synaptic-memory[sqlite,korean,vector,mcp]"

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
pip install synaptic-memory[docs]          # + PDF/DOCX/PPTX/XLSX/HWP 로더
```

</details>

---

## 빠른 시작

### 방법 A: 2줄 (가장 쉬움)

```python
import asyncio
from synaptic import SynapticGraph

async def main():
    # CSV 파일
    graph = await SynapticGraph.from_data("products.csv")

    # JSONL 문서
    graph = await SynapticGraph.from_data("documents.jsonl")

    # 디렉터리 전체 (CSV/JSONL 자동 스캔)
    graph = await SynapticGraph.from_data("./내_코퍼스/")

    # 임베딩 추가 (선택, 의미 검색 품질 향상)
    graph = await SynapticGraph.from_data(
        "./내_코퍼스/",
        embed_url="http://localhost:11434/v1",
    )

    # 검색
    result = await graph.search("내 질문", engine="evidence")
    for activated in result.nodes[:5]:
        print(activated.node.title, activated.activation)

asyncio.run(main())
```

### 방법 B: MCP 서버 (Claude Desktop / Code)

```bash
synaptic-mcp --db my_graph.db
synaptic-mcp --db my_graph.db --embed-url http://localhost:11434/v1
```

Claude가 36개 도구로 그래프를 직접 탐색합니다. 검색, 인제스트, CDC 동기화까지 CLI로 내려가지 않고 대화 안에서.

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
    retriever = SynapticRetriever(graph=graph, k=5, engine="evidence")

    docs = await retriever.ainvoke("내 질문")
    for doc in docs:
        print(doc.page_content[:80], "   ", doc.metadata["score"])

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
| **Synaptic** | **없음.** 구조·통계 시그널만 (FK, NEXT_CHUNK, phrase DF 허브, MENTIONS) | 비용 0 + 결정론적. 단, 새로운 관계를 스스로 합성하지 않음 |

인덱싱에 LLM을 쓰지 않습니다. 그래프는 지식 저장소가 아니라 검색 인덱스입니다.
LLM이 합성한 요약이 필요하면 그래프 위에 별도 에이전트 레이어로 쌓으세요 —
Synaptic은 primitive를 제공하고, 합성 여부는 사용자가 선택합니다.

> **v0.16.0**: `graph.search()` 기본 엔진이 **`"evidence"`** 로 전환되었습니다.
> `engine="legacy"`는 `DeprecationWarning`을 띄우며 v0.17.0에서 제거 예정.
> Korean 공개 벤치마크에서 **MRR 0.621 → 0.947** (Allganize RAG-ko) 달성.

---

## 에이전트 도구 (36개)

### 텍스트 검색 도구
| 도구 | 용도 |
|------|------|
| `deep_search` | **추천.** 검색 → 확장 → 문서 읽기를 한 번에 |
| `compare_search` | 복합 질문 자동 분해 + 병렬 검색 |
| `knowledge_search` | 핵심 의미 검색 (v0.14.2+에서 EvidenceSearch 경유) |
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

### 인제스트 / CDC 도구 (v0.14.0+)
대화 중에 Claude가 새 자료를 배울 수 있도록 하는 6개 도구.

| 도구 | 용도 |
|------|------|
| `knowledge_add_document` | 긴 텍스트를 자동 청킹해 그래프에 추가 |
| `knowledge_add_table` | 컬럼+행 리스트를 ENTITY + FK 엣지로 인제스트 |
| `knowledge_add_chunks` | BYO-chunker 경로 |
| `knowledge_ingest_path` | 로컬 CSV/JSONL/TXT 파일 단건 인제스트 |
| `knowledge_remove` | 단건 노드 삭제 (엣지 cascade) |
| `knowledge_sync_from_database` | CDC 증분 동기화 |
| `knowledge_backfill` | 누락된 임베딩·phrase 허브 복구 (v0.14.4+) |

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
  ↓  HybridReranker (어휘 + 의미 + 그래프 + 구조 + 권위 + 시간)
  ↓  MaxP 문서 집계 (커버리지 보너스)
  ↓  Cross-encoder reranker (bge-reranker-v2-m3, 선택)
  ↓  EvidenceAggregator (MMR 다양성 + 문서당 캡 + 카테고리 커버리지)
결과
```

---

## 벤치마크

### 재현 가능한 임베더-프리 베이스라인 (노트북 2초)

```bash
pip install "synaptic-memory[korean]"
python examples/benchmark_allganize.py
```

결정론적 출력 (v0.16.0):

```
Dataset                  Corpus  Queries      MRR     R@10        Hit     Time
--------------------------------------------------------------------------------
Allganize RAG-ko            200      200    0.947    1.000   200/200     9.3s
Allganize RAG-Eval          300      300    0.911    0.950   285/300     5.9s
```

임베더·reranker 없이 **EvidenceSearch 파이프라인(BM25 + PPR + MMR)** 만으로
낸 결과입니다. 전체 소스:
[`examples/benchmark_allganize.py`](examples/benchmark_allganize.py).
원본: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO).

> **v0.16.0 개선 누적**. v0.15.1의 query-mode Kiwi + v0.16.0의 engine default
> 전환으로 Korean 공개 벤치마크 전반에서 MRR이 **+0.22~+0.33 상승**. 영어
> HotPotQA-24도 **+0.148**. 자세한 ablation:
> [`examples/ablation/run_ablation.py`](examples/ablation/run_ablation.py).

### 영어 multi-hop 표준 벤치마크 (500q subset, v0.16.0)

```bash
pip install "synaptic-memory[eval]"
python examples/ablation/download_benchmarks.py
python examples/ablation/run_tier1_benchmarks.py --subset 500
```

| 데이터셋 | 소스 | Docs | MRR@10 | R@5 | Hit@10 |
|---------|------|------|--------|-----|--------|
| HotPotQA dev (distractor) | HuggingFace | 66,635 | **0.784** | 0.585 | 459/500 (91.8%) |
| 2WikiMultihopQA dev | HuggingFace | 56,687 | **0.795** | 0.501 | 456/500 (91.2%) |
| MuSiQue-Ans dev | HuggingFace | 21,100 | 0.590 | 0.379 | 381/500 (76.2%) |

HippoRAG2 등 선행 연구와의 해석은
[docs/comparison/synaptic_results.md](docs/comparison/synaptic_results.md#tier-15--english-multi-hop-standard-benchmarks-v0160)
참고. 직접 비교 지표가 아니므로 엄격한 head-to-head는 조심하지만,
**MuSiQue R@5 0.379 vs HippoRAG2 0.747 격차**는 embedder 도입 시 좁힐 여지.

### 전체 파이프라인 (임베더 + reranker, pre-v0.16.0 측정)

아래 수치는 **v0.16.0 engine flip 이전** 측정값으로, **EvidenceSearch + 임베더
(Ollama `qwen3-embedding:4b`) + cross-encoder reranker(TEI `bge-reranker-v2-m3`)**
조합이었습니다. 재현에는 GPU 환경이 필요합니다. 전체 harness:
[`eval/run_all.py`](eval/run_all.py). v0.16.1에서 재측정 예정.

### 단일 검색

| 데이터셋 | 유형 | 노드 | MRR | Hit |
|---------|------|------|-----|-----|
| KRRA Easy | 한국어 문서 (비공개) | 19,720 | **0.967** | 20/20 |
| KRRA Hard | 한국어 문서 (비공개) | 19,720 | **1.000** | 15/15 |
| X2BEE Easy | PostgreSQL 이커머스 (비공개) | 19,843 | **1.000** | 20/20 |
| assort Easy | 패션 CSV (비공개) | 13,909 | **0.867** | 13/15 |
| HotPotQA-24 | 영어 multi-hop (공개 서브셋) | 226 | **0.964** | 24/24 |

> HotPotQA-24는 24문항 서브셋. 전체 HotPotQA-dev(7,405q) 전체 실행은
> v0.16.1에서 예정 (PPR O(corpus) 최적화 후).

### 멀티턴 에이전트 (GPT-4o-mini, 최대 5턴)

| 데이터셋 | 결과 |
|---------|------|
| KRRA Hard agent | 10-13/15 (67-87%) |
| **X2BEE Hard agent** | **17/19 (89%)** |
| **assort Hard agent** | **12/15 (80%)** |

필터 / 집계 / FK 조인 / 카운팅 같은 정형 데이터 질의가 그래프 기반 도구로 end-to-end 동작합니다.

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
에이전트 도구 (36개) → MCP 서버 → LLM 에이전트
```

---

## 백엔드

| 백엔드 | 벡터 검색 | 규모 | 용도 |
|--------|----------|------|------|
| `MemoryBackend` | cosine | ~1만 | 테스트 |
| `SqliteGraphBackend` | **usearch HNSW** | ~10만 | **기본 권장** |
| `KuzuBackend` | HNSW | ~1천만 | 그래프 중심 |
| `PostgreSQLBackend` | pgvector | ~100만 | 프로덕션 |
| `CompositeBackend` | Qdrant | 무제한 | 스케일아웃 |

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
| `docs` | PDF/DOCX/PPTX/XLSX/HWP 문서 로더 (xgen-doc2chunk) |

---

## 문서

| 문서 | 내용 |
|------|------|
| [docs/GUIDE.md](docs/GUIDE.md) | 친절한 전체 안내서 (처음 접하는 사람용) |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | 30분 단계별 실습 |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | 파이프라인 심화 설명 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 신경망 영감 초기 설계 |
| [docs/COMPARISON.md](docs/COMPARISON.md) | GraphRAG / LightRAG 등과 비교 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 향후 로드맵 |

## 개발

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q                   # 818+ 테스트
uv run ruff check --fix
```

## 라이선스

MIT
