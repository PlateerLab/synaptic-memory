# Synaptic Memory

LLM 에이전트를 위한 지식 그래프 + MCP 도구 서버.

아무 데이터나 넣으면 구조화된 그래프가 만들어지고, LLM 에이전트가 29개 도구로 탐색합니다.

[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> [English README](README.md)

---

## 2줄이면 시작

```python
from synaptic import SynapticGraph

# 아무 데이터 → 지식 그래프 (CSV, JSONL, 디렉터리)
graph = await SynapticGraph.from_data("./내_데이터/")

# 또는 DB에서 바로 — SQLite / PostgreSQL / MySQL / Oracle / MSSQL
graph = await SynapticGraph.from_database(
    "postgresql://user:pass@host:5432/dbname"
)

# 검색
result = await graph.search("내 질문")
```

파일 형식 또는 DB 스키마 자동 감지, 온톨로지 프로파일 자동 생성, 인제스트, 인덱싱, FK 엣지 구축까지 전부 자동.

---

## 이 라이브러리가 하는 일

```
내 데이터 (CSV, JSONL, PDF, SQL 데이터베이스)
  ↓  형식 자동 감지 / DB 스키마+FK 자동 발견
  ↓  DocumentIngester (텍스트) / TableIngester / DbIngester
  ↓
지식 그래프
  ├─ 문서: Category → Document → Chunk
  └─ 정형: 테이블 row → ENTITY 노드 + RELATED 엣지 (FK)
  ↓
29개 MCP 도구 → LLM 에이전트가 그래프 기반 멀티턴으로 탐색
```

**라이브러리가 하는 건 딱 두 가지:**
1. **그래프를 잘 구축한다** — 인덱싱에 LLM 비용 0원
2. **LLM에게 좋은 도구를 쥐어준다** — 판단은 LLM이, 코드는 데이터만

---

## 설치

```bash
pip install synaptic-memory                # 코어 (의존성 0)
pip install synaptic-memory[sqlite]        # + SQLite FTS5 백엔드
pip install synaptic-memory[korean]        # + Kiwi 한국어 형태소 분석
pip install synaptic-memory[vector]        # + usearch HNSW 벡터 인덱스
pip install synaptic-memory[mcp]           # + Claude MCP 서버
pip install synaptic-memory[all]           # 전부
```

---

## 빠른 시작

### 방법 A: 2줄 (가장 쉬움)

```python
from synaptic import SynapticGraph

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
result = await graph.search("내 질문")
```

### 방법 B: MCP 서버 (Claude Desktop / Code)

```bash
synaptic-mcp --db my_graph.db
synaptic-mcp --db my_graph.db --embed-url http://localhost:11434/v1
```

Claude가 29개 도구로 그래프를 직접 탐색합니다.

### 방법 C: 세밀한 제어

```python
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.document_ingester import DocumentIngester, JsonlDocumentSource

profile = DomainProfile.load("my_profile.toml")
backend = SqliteGraphBackend("graph.db")
await backend.connect()

source = JsonlDocumentSource("docs.jsonl", "chunks.jsonl")
ingester = DocumentIngester(profile=profile, backend=backend)
await ingester.ingest(source)
```

---

## 3세대 검색

| 세대 | 방식 | 인덱싱 LLM 비용 |
|-----|------|---------------|
| 1세대 (GraphRAG) | LLM으로 엔티티+관계+요약 추출 | 높음 |
| 2세대 (LightRAG) | LLM 호출을 쿼리 시점으로 미룸 | 중간 |
| **3세대 (이것)** | **관계 없는 그래프, 하이브리드 검색** | **0원** |

인덱싱에 LLM을 쓰지 않습니다. 그래프는 지식 저장소가 아니라 검색 인덱스입니다.

---

## 에이전트 도구 (29개)

### 텍스트 검색 도구
| 도구 | 용도 |
|------|------|
| `deep_search` | **추천.** 검색→확장→문서 읽기를 한 번에 |
| `compare_search` | 복합 질문 자동 분해 + 병렬 검색 |
| `search` | FTS + 벡터 하이브리드 검색 |
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

### 단일 검색 (EvidenceSearch + embed + reranker)

| 데이터셋 | 유형 | 노드 | MRR | Hit |
|---------|------|------|-----|-----|
| KRRA Easy | 한국어 문서 | 19,720 | **0.967** | 20/20 |
| KRRA Hard | 한국어 문서 | 19,720 | **1.000** | 15/15 |
| X2BEE Easy | PostgreSQL 이커머스 | 19,843 | **1.000** | 20/20 |
| assort Easy | 패션 CSV | 13,909 | **0.867** | 13/15 |
| HotPotQA-24 | 영어 multi-hop | 226 | **0.964** | 24/24 |
| Allganize RAG-ko | 한국어 기업 문서 | 200 | **0.905** | — |
| Allganize RAG-Eval | 금융/의료/법률 | 300 | **0.874** | — |
| PublicHealthQA | 한국어 공중보건 | 77 | **0.600** | 56/77 |

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
에이전트 도구 (29개) → MCP 서버 → LLM 에이전트
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
| `korean` | Kiwi 한국어 형태소 분석기 (FTS 품질 향상) |
| `vector` | usearch HNSW 인덱스 (벡터 검색 100배 빠름) |
| `embedding` | 임베딩 API 호출용 aiohttp |
| `mcp` | Claude Desktop/Code MCP 서버 |
| `sqlite` | aiosqlite 백엔드 |

---

## 문서

| 문서 | 내용 |
|------|------|
| [docs/GUIDE.md](docs/GUIDE.md) | 친절한 전체 안내서 (처음 접하는 사람용) |
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | 30분 단계별 실습 |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | 3세대 GraphRAG + 파이프라인 심화 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 신경망 영감 초기 설계 |
| [docs/COMPARISON.md](docs/COMPARISON.md) | GraphRAG/LightRAG와 비교 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 향후 로드맵 |

## 개발

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q                   # 687+ 테스트
uv run ruff check --fix
```

## 라이선스

MIT
