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

# 검색
result = await graph.search("내 질문")
```

파일 형식 자동 감지, 온톨로지 프로파일 자동 생성, 인제스트, 인덱싱까지 전부 자동.

---

## 이 라이브러리가 하는 일

```
내 데이터 (CSV, JSONL, PDF 등)
  ↓  형식 자동 감지 + DomainProfile 자동 생성
  ↓  DocumentIngester (텍스트) / TableIngester (정형 데이터)
  ↓
지식 그래프 (Category → Document → Chunk)
  ↓
29개 MCP 도구 → LLM 에이전트가 멀티턴으로 탐색
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
| `filter_nodes` | 속성 필터 (>=, <=, contains) — SQL WHERE 대체 |
| `aggregate_nodes` | GROUP BY + COUNT/SUM/AVG |
| `join_related` | FK 기반 관련 레코드 조회 — SQL JOIN 대체 |

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

### 단일 검색 (single-shot)

| 데이터셋 | 유형 | 노드 | Easy MRR | Hard MRR |
|---------|------|------|----------|----------|
| KRRA (한국 공공기관) | 텍스트 문서 | 19,720 | **0.967** | 0.507 |
| assort (패션 이커머스) | 정형 CSV | 13,909 | **0.880** | 0.127 |

### 멀티턴 에이전트 (Claude Sonnet 4.6)

| 쿼리 유형 | 예시 | 턴 수 | 결과 |
|----------|------|-------|------|
| 정확 매칭 | "인권영향평가 결과" | 6 | 상세 테이블 |
| 교차 문서 | "운영계획과 인권경영" | 10 | 다중 출처 종합 |
| 부재 증명 | "환불 예외 있나?" | 7 | 3가지 예외 조항 발견 |
| 패러프레이즈 | "말 복지 프로그램" | 8 | 재활힐링승마 발견 |
| **Hard (single-shot 실패)** | **4개 쿼리** | **6-10** | **4/4 해결** |

Single-shot MRR 0.507 → 멀티턴 **100% 해결**.

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

## 개발

```bash
uv sync --extra dev --extra sqlite --extra mcp
uv run pytest tests/ -q                   # 687+ 테스트
uv run ruff check --fix
```

## 라이선스

MIT
