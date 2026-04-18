# Synaptic Memory — 프로젝트 지침

## 프로젝트 개요
LLM 에이전트용 지식 그래프 + MCP 도구 서버.
아무 데이터(CSV, JSONL, PDF/DOCX/PPTX/XLSX/HWP, SQL DB)를 넣으면 그래프를 자동 구축하고, 36개 도구로 LLM이 탐색.

- PyPI: `synaptic-memory` (v0.15.0)
- 라이선스: MIT
- Python: >=3.12
- 코어 의존성: **0** (백엔드/임베더/한국어 분석 전부 optional)

## 핵심 원칙

1. **코드는 데이터와 도구만** — 판단 로직은 전부 LLM에 위임
2. **torch 의존성 0** — BYO embedder/reranker (Ollama, TEI, API 주입)
3. **범용화** — 도메인 종속 코드 금지, DomainProfile TOML로 주입
4. **3세대 검색** — 인덱싱에 LLM 비용 0원, relation-free graph

## Easy API

```python
from synaptic import SynapticGraph

# 2줄로 시작
graph = await SynapticGraph.from_data("./my_data/")
result = await graph.search("my question")
```

## 아키텍처

```
SynapticGraph.from_data("./data/")
  ↓ 자동: 형식 감지 → DomainProfile → Ingest → Index
  ↓
StorageBackend
  ├── MemoryBackend (테스트)
  ├── SqliteGraphBackend (기본 권장, FTS5 + HNSW)
  ├── KuzuBackend (임베디드 Cypher)
  ├── PostgreSQLBackend (pgvector)
  └── CompositeBackend (조합)
  ↓
검색 파이프라인
  Kiwi 형태소 → BM25 → Vector(HNSW) → PRF → PPR → Reranker → MaxP → MMR
  ↓
Agent tools (36개) → MCP server → LLM agent
```

### 핵심 모듈
| 모듈 | 역할 |
|------|------|
| `graph.py` | SynapticGraph — 메인 facade + `from_data()` Easy API |
| `search.py` | HybridSearch (legacy, v0.5) |
| `extensions/evidence_search.py` | **EvidenceSearch** — 3세대 파이프라인 (v0.12) |
| `agent_tools.py` | 7개 원자적 도구 (search, expand, get_document, ...) |
| `agent_tools_v2.py` | compound 도구 (deep_search, compare_search) |
| `agent_tools_structured.py` | 정형 데이터 도구 (filter, aggregate, join) |
| `search_session.py` | 멀티턴 상태 + `build_graph_context()` |
| `extensions/reranker_cross.py` | Cross-encoder BYO protocol |

### 인제스트
| 모듈 | 역할 |
|------|------|
| `extensions/document_ingester.py` | 텍스트 문서 → Category→Document→Chunk |
| `extensions/table_ingester.py` | CSV/테이블 → typed property nodes (`source_url=`로 deterministic ID 활성화) |
| `extensions/db_ingester.py` | 5종 DB 자동 인제스트 + CDC 동기화 오케스트레이터 |
| `extensions/cdc/` | Change Data Capture — 증분 동기화 (timestamp/hash 전략, 삭제 감지, FK 재계산) |
| `extensions/domain_profile.py` | TOML 도메인 설정 (stopwords, ontology_hints) |
| `extensions/profile_generator.py` | 자동 프로파일 생성 (3-tier: rule→classifier→LLM) |
| `extensions/entity_linker.py` | DF 필터 phrase hub + MENTIONS 엣지 |

### CDC (Live database sync)
프로덕션 DB와 연동할 때 매번 전체 재빌드 대신 변경분만 반영.

- **`from_database(mode="cdc")`** — 첫 호출은 deterministic ID로 풀로드 + sync state 기록
- **`sync_from_database(dsn)`** — 두 번째 호출부터 증분 (added/updated/deleted)
- **`mode="auto"`** — 기존 state 있으면 cdc, 없으면 full
- **전략**: `updated_at`류 컬럼 있으면 timestamp (`WHERE col >= watermark`), 없으면 hash fallback (row content blake2b)
- **삭제 감지**: TEMP TABLE + LEFT JOIN (메모리 부담 0)
- **FK 재계산**: row의 FK가 바뀌면 RELATED 엣지 자동 rewire
- **dialects**: SQLite, PostgreSQL, MySQL/MariaDB. Oracle/MSSQL은 legacy full-reload만.
- **검색 품질 보장**: `mode="cdc"`와 `mode="full"`이 동일 top-k 반환 (regression test로 잠금)

### 백엔드
| 백엔드 | 벡터 검색 | 규모 | 의존성 |
|--------|----------|------|--------|
| `MemoryBackend` | cosine | ~1만 | 없음 |
| `SqliteGraphBackend` | **usearch HNSW** | ~10만 | aiosqlite |
| `KuzuBackend` | HNSW | ~1천만 | kuzu |
| `PostgreSQLBackend` | pgvector | ~100만 | asyncpg |
| `CompositeBackend` | Qdrant | 무제한 | 조합 |

## 테스트

```bash
# 단위 테스트 (687+ 건)
uv run pytest tests/ -q \
  --ignore=tests/test_backend_postgresql.py \
  --ignore=tests/test_backend_qdrant.py \
  --ignore=tests/test_backend_minio.py \
  --ignore=tests/test_backend_composite.py \
  --ignore=tests/test_backend_kuzu.py \
  --ignore=tests/benchmark

# lint
uv run ruff check src/ tests/ --fix
```

## QA 벤치마크

### 실행
```bash
# 개발 후 QA (9개 데이터셋 자동 실행)
uv run python eval/run_all.py --quick

# 전체 (대규모 포함)
uv run python eval/run_all.py

# 회귀 감지 (이전 결과 비교)
uv run python eval/run_all.py --compare eval/results/qa_latest.json
```

### 현재 베이스라인 (v0.17.0, 2026-04-19 확정)

두 모드를 병기한다:

- **FTS-only** — `eval/run_all.py --quick`. Embedder/Reranker 없음. CI/일상 회귀 검증용. v0.16.0 engine flip + Kiwi 개선으로 이미 강함 (Korean MRR 0.547-0.947).
- **Full pipeline** — `eval/run_all.py --quick --local-bge`. `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3` (cuda:0, FP16), `rerank_blend=0.1` (v0.17.0 튜닝 값).

> Corpus 스냅샷 hash + 코드 버전은 `eval/baselines/qa_latest.json:_meta`에 인라인.

#### 공개 데이터셋 — FTS-only vs Full pipeline

| 데이터셋 | 언어 | 쿼리 | FTS-only MRR | Full pipeline MRR (blend=0.1) | Δ | 비고 |
|---------|------|-----|-------------:|------------------------------:|---:|---|
| HotPotQA-24 | EN | 24q | 0.875 | **0.979** | +12% | |
| Allganize RAG-ko | KO | 200q | 0.947 | **0.982** | +3.7% | |
| Allganize RAG-Eval | KO | 300q | 0.911 | **0.946** | +3.8% | |
| PublicHealthQA | KO | 77q | 0.547 | **0.734** | +34% | paraphrase-heavy 의료 |
| AutoRAG | KO | 720q | **0.906** | 0.766 | **−15%** | ⚠️ retrieval-style regression |
| **평균** | | | 0.837 | **0.881** | +5.3% | |

**중요 관찰**
- Full pipeline은 **paraphrase-heavy corpora** (PublicHealthQA +34%, HotPotQA +12%) 에서 효과 큼. Synaptic의 한국어 FTS 베이스가 이미 강해서 (Kiwi + EvidenceSearch) 마진적 corpus에선 개선 폭 ~4%.
- **AutoRAG 같은 retrieval-style corpus 에선 reranker 가 구조적으로 해로움** — FTS 순위가 이미 최적인데 cross-encoder 재정렬이 꼬임. v0.17.0 의 `rerank_blend=0.1` 튜닝으로 완화했지만 완전 해결 불가. 이런 corpus엔 `reranker=None` 권장.
- **Blend weight 튜닝 근거**: `examples/ablation/sweep_rerank_blend.py` — 0.1이 5개 벤치 전부에서 0.2/0.4보다 우수. 평균 MRR 0.849 (b=0.4) → 0.881 (b=0.1).
- `EntityLinker` (`--entity-linker`) 는 이 벤치들에서 ±1% 미만의 영향. Release scope에서 기본 비활성화.

#### 단일 검색 — FTS only (custom, v0.14.4 2026-04-15)

> 아래 custom 벤치는 pre-built SQLite graph (`eval/data/graphs/*.db`) 기반이므로
> Full pipeline 재측정이 H100 에서 당장 불가 (graph 파일 laptop/home 서버에 있음).
> R1 후속 작업에서 전송 or 재인제스트 후 갱신 예정.

| 데이터셋 | 언어 | 쿼리 수 | MRR | Hit | 비고 |
|---------|------|--------|-----|-----|------|
| KRRA Easy | KO | 20q | **0.450** | 9/20 | krra chunks 재파싱 후 |
| KRRA Hard | KO | 40q | **0.391** | 21/40 | v0.14.x 보강 후 |
| assort Easy | KO | 15q | **0.883** | 14/15 | 정형 CSV |
| assort Hard | KO | 40q | 0.000 | 0/40 | structured-only — agent 필요 |
| X2BEE Easy | EN | 20q | **1.000** | 20/20 | DB→온톨로지 |
| X2BEE Hard | EN/KO | 20q | 0.263 | 5/20 | 패러프레이즈+필터+집계 |
| KRRA Conv | KO | 30q | 0.176 | 9/30 | conv는 single-shot 한계 |
| assort Conv | KO | 30q | 0.425 | 11/30 | |
| X2BEE Conv | EN/KO | 30q | 0.167 | 5/30 | |

#### 알려진 격차 — MuSiQue (영어 multi-hop)

MuSiQue-Ans dev 500q full pipeline 측정 (`run_tier1_benchmarks.py --only musique --subset 500 --local-bge`):

- **R@5 0.453** vs HippoRAG2 publish **0.747** (−0.294)
- 3-round ablation (decomposer / inline phrase / DF-filtered entity linker) 모두 baseline 개선 못함
- 결론: **OpenIE triple extraction + query-to-triple dense linking** 같은 architecture 교체가 필요. v0.18.0+ 연구 트랙.
- 세부: `docs/PLAN-v0.17-ontology.md` §4.5, §9

#### 멀티턴 에이전트 (GPT-4o-mini, 5턴, LLM-judge — 별도 측정 필요)
| 데이터셋 | 결과 | 비고 |
|---------|------|------|
| **KRRA Hard agent** | **11/15 (73%)** | v0.13 측정값 — 보강 query set으로 재측정 필요 |
| **assort Hard agent** | **13/15 (87%)** | 동상 |
| **X2BEE Hard agent** | **17/19 (89%)** | 동상 |
| **KRRA Conv agent** | **21/30 (70%)** | 동상 |
| **assort Conv agent** | **20/24 (83%)** | 동상 |
| **X2BEE Conv agent** | **22/27 (81%)** | 동상 |

> Agent 벤치마크 + Embedder/Reranker single-shot 둘 다 v0.13.0 시점 측정값. v0.14.x
> 코드는 search 경로가 변경됐으므로 (HybridSearch threshold + EvidenceSearch
> migration) 이 두 모드는 재측정 필요. follow-up task로 trace.

### 평가 쿼리 위치
```
eval/data/queries/
├── krra.json                  # KRRA Easy 20q (키워드 직접 매칭)
├── krra_hard.json             # KRRA Hard 15q (패러프레이즈, 교차문서, 대화체)
├── krra_multihop.json         # KRRA 교차 문서 10q
├── krra_conversational.json   # KRRA 복합/대화형 30q (auto-GT)
├── assort.json                # assort Easy 15q
├── assort_hard.json           # assort Hard 15q (필터, 집계, FK조인)
├── assort_conversational.json # assort 복합/대화형 30q
├── x2bee.json                 # X2BEE Easy 20q (DB→온톨로지 키워드 검색)
├── x2bee_hard.json            # X2BEE Hard 20q (패러프레이즈, 필터, 집계, 멀티홉)
└── x2bee_conversational.json  # X2BEE 복합/대화형 30q

# 통합 엑셀 (정답 포함, 11 sheets, 200 queries)
eval/data/gt_datasets.xlsx
```

## MCP 서버 (36개 도구)

```bash
synaptic-mcp --db knowledge.db
synaptic-mcp --db knowledge.db --embed-url http://localhost:11434/v1
# CDC sync용 소스 DB를 미리 바인딩 (tool 호출 시 dsn 생략 가능)
synaptic-mcp --db knowledge.db --source-dsn postgresql://user:pw@host/db
```

### 도구 분류
| 분류 | 도구 수 | 예시 |
|------|--------|------|
| Knowledge CRUD | 8 | search, add, link, reinforce, stats, export, consolidate, **backfill** |
| **Ingest / CDC** | 6 | add_document, add_table, add_chunks, ingest_path, remove, sync_from_database |
| Agent workflow | 4 | start_session, log_action, record_decision, record_outcome |
| Semantic search | 3 | find_similar, get_reasoning_chain, explore_context |
| Ontology | 2 | define_type, query_schema |
| **Agent v1** | 8 | search, expand, get_document, list_categories, count, search_exact, follow, session_info |
| **Agent v2** | 2 | deep_search, compare_search |
| **Structured** | 3 | filter_nodes, aggregate_nodes, join_related |

### Ingest / CDC 도구 (v0.14.0+)
에이전트가 대화 중에 직접 지식 베이스를 업데이트할 수 있게 하는 6개 도구.
기존에는 CLI 스크립트로만 가능했던 인제스트를 MCP tool call로 수행 가능.

- **`knowledge_add_document`** — 긴 텍스트를 자동 청킹해서 그래프에 추가
- **`knowledge_add_table`** — 컬럼 정의 + 행 리스트를 받아 ENTITY 노드 + FK 엣지로 인제스트
- **`knowledge_add_chunks`** — 이미 청킹된 결과(BYO-chunker)를 일괄 추가
- **`knowledge_ingest_path`** — 로컬 파일(CSV/JSONL/TXT) 단일 파일 인제스트
- **`knowledge_remove`** — 단건 노드 삭제 (엣지 cascade)
- **`knowledge_sync_from_database`** — CDC 증분 동기화. 첫 호출은 풀 로드, 이후는
  변경분만. `--source-dsn`로 기본 DSN을 바인딩하면 dsn 인자 생략 가능.

## 배포

### PyPI
```bash
uv build && uv publish --username __token__ --password "$PYPI_TOKEN"
```

### 설치
```bash
pip install synaptic-memory              # 코어 (의존성 0)
pip install synaptic-memory[sqlite]      # SQLite FTS5
pip install synaptic-memory[korean]      # Kiwi 한국어 형태소
pip install synaptic-memory[vector]      # usearch HNSW
pip install synaptic-memory[embedding]   # 임베딩 API
pip install synaptic-memory[mcp]         # MCP 서버
pip install synaptic-memory[all]         # 전부
```

## 검색 파이프라인 상세

### EvidenceSearch (3세대, v0.12)
```
Step 0: 쿼리 임베딩 (BYO embedder, 선택)
Step 1: QueryAnchorExtractor (카테고리/엔티티/키워드)
Step 2a: FTS seed (BM25 + Kiwi 형태소 + title 3x boost)
Step 2b: Vector seed (usearch HNSW, cascade)
Step 2c: Vector PRF (top-3 임베딩 평균 → 2차 검색)
Step 3: GraphExpander (1-hop: category siblings, chunk-next, MENTIONS)
Step 3b: PPR graph discovery
Step 4: HybridReranker (lexical + semantic + graph + structural + authority + temporal + MaxP)
Step 4b: Cross-encoder reranker (BYO, TEI/Ollama)
Step 5: EvidenceAggregator (MMR + per-doc cap + category coverage)
```

### 알고리즘 개선 내역 (v0.12)
- Phase 1: MaxP document aggregation + Vector PRF → Hard MRR +21%
- Phase 2: usearch HNSW → latency 11s → 1ms (100x)
- Phase 3: Cross-encoder reranker (bge-reranker-v2-m3) → Hard MRR +22%
- Phase 4: PPR graph discovery in EvidenceSearch
- Kiwi 형태소 분석기: 한국어 FTS 조사 분리 (한글 비율 50%+ 자동 감지)
- rank_to_score: step 0.03, floor 0.10 (하위 결과 순위 보존)
- 구조 개선 6건: batch INSERT, count_nodes, get_nodes_batch, HNSW 무효화 등

## Home 서버
- IP: 14.6.220.78 (ssh home)
- GPU: RTX 3080 (10GB VRAM)
- Ollama: qwen3-embedding:4b, bge-m3, qwen3.5:4b, qwen2.5:14b
- TEI: bge-reranker-v2-m3 (Docker, port 8180)
- Qdrant, PostgreSQL, MinIO 등 인프라

## 방향성
- **3세대 GraphRAG**: relation-free graph + hybrid retrieval + LLM-free indexing
- **멀티턴 에이전트**: deep_search + compare_search + graph context injection → 10턴 → 2-3턴
- **정형 + 비정형 통합**: 같은 그래프에 문서(FTS)와 테이블(filter/aggregate/join)
- **범용 라이브러리**: `from_data()` 2줄이면 어떤 데이터든 온톨로지 구축
