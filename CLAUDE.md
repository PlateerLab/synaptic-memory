# Synaptic Memory — 프로젝트 지침

## 프로젝트 개요
LLM/멀티에이전트용 뇌 기반 지식 그래프 라이브러리 + MCP 서버.
에이전트가 경험을 구조화하고, 과거 패턴을 검색/추론할 수 있게 하는 적응형 메모리 아키텍처.

- PyPI: `synaptic-memory` (v0.5.0 배포 완료)
- 라이선스: MIT
- Python: >=3.12

## 아키텍처

### 핵심 모듈
| 모듈 | 역할 |
|------|------|
| `graph.py` | SynapticGraph — 메인 facade (add, search, link, reinforce, consolidate) |
| `search.py` | HybridSearch — 3단계 폴백 (FTS → fuzzy → synonym → rewriter) + spreading activation |
| `agent_search.py` | AgentSearch — intent 기반 검색 (similar_decisions, past_failures, related_rules 등) |
| `resonance.py` | ResonanceScorer — 4축 공명 (relevance × importance × recency × vitality) |
| `hebbian.py` | HebbianEngine — co-activation 강화/약화 |
| `consolidation.py` | ConsolidationCascade — L0→L1→L2→L3 메모리 정리 |
| `ontology.py` | OntologyRegistry — 타입 계층, 관계 제약, 검증 |
| `activity.py` | ActivityTracker — 에이전트 세션/tool call/결정/결과 추적 |
| `models.py` | Node, Edge, NodeKind, EdgeKind, SearchResult 등 |

### 백엔드 (7종)
| 백엔드 | 용도 | 의존성 |
|--------|------|--------|
| `MemoryBackend` | 테스트/개발 | 없음 |
| `SQLiteBackend` | 경량 프로덕션 (그래프 없음) | aiosqlite |
| `KuzuBackend` | **임베디드 그래프 DB (기본 권장)** | kuzu |
| `PostgreSQLBackend` | 프로덕션 단일 DB | asyncpg, pgvector |
| `QdrantBackend` | 벡터 검색 | qdrant-client |
| `MinIOBackend` | 대용량 콘텐츠 | miniopy-async |
| `CompositeBackend` | 용도별 분리 | Kuzu + Qdrant + MinIO |

## 테스트

### 인프라 요구사항
```bash
# Kuzu — 임베디드, 인프라 불필요 (pip install synaptic-memory[kuzu])

# Qdrant
docker start qdrant  # 또는 docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# PostgreSQL — 기존 컨테이너 사용 (ailab:ailab123@localhost:5432/plateerag)
# MinIO — 기존 컨테이너 사용 (localhost:9000)
```

### 실행
```bash
# 전체 테스트 (281건)
uv run pytest tests/ -v

# 외부 인프라 의존 백엔드 제외 (로컬 빠른 실행용)
uv run pytest tests/ \
  --ignore=tests/test_backend_postgresql.py \
  --ignore=tests/test_backend_qdrant.py \
  --ignore=tests/test_backend_minio.py \
  --ignore=tests/test_backend_composite.py -v

# 벤치마크만
uv run pytest tests/benchmark/ -v -s
```

### Qdrant 테스트 주의
- fixture에서 매 테스트 전 `test_synaptic` collection 삭제 후 재생성
- 이전 비정상 종료 시 segment 잔재로 500 에러 발생 가능 → collection 수동 삭제 후 재실행

## 벤치마크

### 구조
```
tests/benchmark/
├── conftest.py                    # 엔터프라이즈 시나리오 fixture
├── metrics.py                     # IR 평가 지표 (MRR, nDCG, P@K, R@K, F1@K)
├── generate_data.py               # 시나리오 데이터 생성기 (아직 미사용)
├── download_datasets.py           # HuggingFace 외부 데이터셋 다운로드
├── test_enterprise_benchmark.py   # 자체 시나리오 벤치마크 (50개 쿼리)
├── test_external_datasets.py      # 외부 데이터셋 벤치마크
└── data/
    ├── enterprise_scenario.json   # 자체 시나리오 v1 (12 지식 + 4 세션 + 15 쿼리)
    ├── ko_strategyqa.json         # MTEB Ko-StrategyQA (9.2K corpus, 592 queries)
    ├── autorag_retrieval.json     # MTEB AutoRAGRetrieval (720 corpus, 114 queries)
    ├── klue_mrc.json              # KLUE-MRC (5.8K corpus, 5.8K queries)
    ├── nfcorpus.json              # BeIR NFCorpus (3.6K corpus, 의료/영양)
    ├── scifact.json               # BeIR SciFact (5.2K corpus, 과학 fact-checking)
    ├── fiqa.json                  # BeIR FiQA (57.6K corpus, 금융 QA)
    ├── miracl_retrieval_ko.json   # MTEB MIRACLRetrieval-ko (10K sampled, 위키)
    ├── multilongdoc_ko.json       # MTEB MultiLongDocRetrieval-ko (6.2K, 장문서)
    └── xpqa_ko.json               # MTEB XPQARetrieval-ko (889 corpus, 다도메인)
```

### 외부 데이터셋 다운로드
```bash
uv run python tests/benchmark/download_datasets.py
```
- MIRACL (레거시 형식), Mr. TyDi는 HuggingFace datasets 호환 이슈로 skip
- MIRACLRetrieval (mteb 형식)은 정상 동작 (1.49M → 10K 샘플링)

### 외부 데이터셋 벤치마크 결과 (MemoryBackend, qwen3-embedding:4b)
| 데이터셋 | 언어 | Corpus | FTS only MRR | FTS+Embed MRR | 개선 |
|----------|------|--------|-------------|--------------|------|
| HotPotQA-24 | EN | 226 | 0.752 | **0.873** | +16.1% |
| HotPotQA-200 | EN | 1,990 | 0.742 | **0.846** | +14.0% |
| Allganize rag-ko | KO | 200 | 0.782 | **0.841** | +7.5% |
| Allganize RAG-Eval | KO | 300 | 0.796 | **0.828** | +4.0% |
| KLUE-MRC | KO | 500 | 0.607 | **0.727** | +19.8% |
| SciFact | EN | 5,183 | 0.415 | **0.548** | +32.0% |
| NFCorpus | EN | 3,633 | 0.443 | **0.511** | +15.3% |
| Ko-StrategyQA | KO | 9,251 | 0.317 | **0.459** | +44.8% |
| PublicHealthQA | KO | 77 | 0.346 | **0.402** | +16.2% |
| MIRACLRetrieval | KO | 10,000 | 0.792 | (미측정) | - |
| XPQARetrieval | KO | 889 | 0.167 | (미측정) | - |
| FiQA | EN | 57,638 | 0.132 | (미측정) | - |
| MultiLongDocRetrieval | KO | 6,176 | 0.070 | (미측정) | - |

### 자체 시나리오 벤치마크 결과
| 지표 | v0.5.0 Baseline | v0.5.0 개선 | v0.9.0 + Embedding |
|------|-----------------|------------|-------------------|
| MRR | 0.326 | 0.477 | **0.791** (+66%) |
| Mean P@5 | 0.160 | 0.227 | **0.293** (+29%) |
| Mean R@5 | 0.467 | 0.533 | **0.767** (+44%) |
| Mean nDCG@5 | 0.351 | 0.431 | **0.695** (+61%) |
| Hit rate | 9/15 | 13/15 | **15/15** |

### KRRA (eval/) 벤치마크 결과 — 2026-04-12 Day 1 baseline
독립 eval harness (`eval/scripts/score_krra.py`), 20 seed 쿼리, k=10, FTS only.

| 지표 | NFD graph | NFC graph | Δ |
|------|-----------|-----------|---|
| MRR | 0.525 | **0.650** | +23.8% |
| Mean P@10 | 0.186 | **0.392** | +110.8% |
| Mean R@10 | 0.417 | 0.453 | +8.6% |
| Mean nDCG@10 | 0.431 | **0.503** | +16.7% |
| Hit rate | 12/20 | 13/20 | +1 |
| Avg latency | 728ms | **585ms** | -20% |

**파이프라인**: `parse_krra.py` → `ingest_krra.py` (NFC 정규화 포함) → `score_krra.py` → `eval/results/krra_baseline_*.json`.
**GT**: `eval/data/queries/krra.json` — 카테고리 분산 20개, title 키워드 매칭으로 doc_id 시드.
**현재 그래프**: 구조적 ingestion만 (Category×10 + Document×1,110 + Chunk×18,600, 엔티티/관계 추출 미적용).
**다음 단계 (Day 2)**: home Ollama `qwen3-embedding:4b` 주입해서 Baseline C (embedding cascade) 측정.

### KRRA Day 1 발견 이슈
- 🔴 **라이브러리 NFC/NFD 버그**: `graph.add()`, `graph.search()` 둘 다 Unicode 정규화 없음. Mac HFS+/zfs 소스 한글 데이터 → 검색 실패. 수정 위치: `graph.py:300` (add), `graph.py:586` (search). phrase_extractor.py에만 정규화 있음.
- 🟡 **chunk granularity mismatch**: Document 노드 `content=""`로 FTS 약함. 본문 키워드 조합이 우연히 맞는 무관 청크가 정답 청크를 밀어냄. 7/20 zero-hit의 원인. 수정 옵션: (a) title 가중치 상향, (b) Document.content=title 복제, (c) chunk→doc score aggregation (HippoRAG2).
- 🟡 **parse_krra.py year=null 전건**: NFD filename에서 `(\d{4})년도` 정규식 실패. 텍스트 필드만 NFC 정규화하고 `_doc_id()`는 NFD 유지 (GT 호환).
- 🟢 **ingest_krra.py Kuzu 파일/디렉터리** (수정 완료): Kuzu 0.x 단일 파일, shutil.rmtree 실패 → 파일/디렉터리 + WAL sibling 처리 추가.

### Ablation Study 핵심 발견
- S1 Ontology: 현재 graph.search()가 NodeKind를 랭킹에 미활용 → 효과 없음
- S2 Relations: spreading activation이 노이즈 유입 (MRR -14~-32%)
- S3 Hebbian: HotPotQA multi-hop에서만 +3.9% 기여
- S5 agent_search: kind 필터링이 과도하게 공격적 → recall 하락
- S6 Auto ontology: 보수적 동작으로 성능 유지하지만 개선도 없음

### 경쟁 제품 비교
| 제품 | 벤치마크 | 결과 | 비고 |
|------|----------|------|------|
| Cognee | HotPotQA 24문항 | Correctness 0.925 | end-to-end QA (LLM 포함) |
| Mem0 | LoCoMo | 66.9% | 메모리 정확도 |
| LightRAG | NaiveRAG 비교 | 39% win rate | 독립 검증 (원래 66.7%) |
| HippoRAG2 | HotPotQA | Recall 95.4% | 최고 수준 |

### 검색 개선 내역
1. FTS: title 가중치 3x, bigram 서브스트링 매칭, tag 매칭
2. FTS: 순위 기반 점수 (1위 0.95 → 감소)
3. Fuzzy: threshold 0.3→0.4, content 샘플 50→100 단어, title boost
4. Spreading activation: depth 1→2, 다중 경로 보상 → PPR로 교체
5. AgentSearch past_failures: LESSON 노드 포함, fallback 추가
6. **Embedding cascade**: FTS 순위 보존 + vector-only 결과 보완 (corpus 크기 적응)
   - vec_alpha: 소규모 0.3 → 대규모 0.85 (corpus_size 기반)
   - cos_threshold: 0.45 (전 규모 통일)
   - 실패 실험: fusion/blend/RRF → FTS 순위 교란, 중복 boost cos*0.15/0.05 → regression, threshold 0.40 → 노이즈

## 배포

### PyPI
```bash
source ~/.claude/.secrets  # PYPI_TOKEN 로드
uv build && uv publish --username __token__ --password "$PYPI_TOKEN"
```

### 설치
```bash
pip install synaptic-memory            # 코어만
pip install synaptic-memory[embedding]  # auto-embedding
pip install synaptic-memory[scale]      # Neo4j + Qdrant + MinIO
pip install synaptic-memory[all]        # 전부
pip install synaptic-memory[mcp]       # MCP 서버
```

## 로드맵

### 온톨로지 자동 구축 (3단계)
| Phase | 방식 | 비용 | 한국어 | 의존성 |
|-------|------|------|--------|--------|
| Phase 1 | 임베딩 유사도 자동 연결 | 저렴 | O | sentence-transformers |
| Phase 2 | spaCy dependency parsing | 무료 | 제한적 | spacy (ko_core_news) |
| Phase 3 | LLM 프롬프트 트리플 추출 | 높음 | O | LLM API |

### 검색 엔진 개선 포인트
- spreading activation 가중치 튜닝: edge type별 차등 전파 (RELATED > TAGGED_WITH)
- NodeKind 활용: 검색 랭킹에 kind 정보 반영 (ontology ablation 결과 반영)
- tag 기반 부스팅: tag 매칭 시 가중치 조절
- agent_search kind 필터 완화: recall 보존하면서 precision 유지

### 타겟 수치
| 지표 | v0.9.0 FTS only | v0.9.0 + Embed | 목표 | 달성 |
|------|-----------------|----------------|------|------|
| MRR (Allganize) | 0.796 | **0.828** | 0.85+ | 근접 |
| MRR (HotPotQA-200) | 0.742 | **0.846** | 0.80+ | ✅ |
| MRR (Ko-StrategyQA) | 0.317 | **0.459** | 0.50+ | 근접 |
| 자체 시나리오 MRR | 0.477 | **0.791** | 0.65+ | ✅ |
| MRR (KLUE-MRC) | 0.607 | **0.727** | 0.75+ | 근접 |

## 방향성
- 플래티어 온톨로지 비전과 연계: 엔터프라이즈 시맨틱 레이어
- 정적 검색 → 동적 메모리 (사용 경험에 따라 재편)
- 다층 랭킹: relevance + importance + recency + vitality
- Hebbian 학습: 함께 쓰이는 지식 연결 강화, 실패 패턴 약화
- 선제적 활성화: 태스크 맥락 기반 proactive loading
- **차별점**: 단순 retrieval이 아닌 "에이전트 경험 메모리" — 검색+학습+적응의 통합
