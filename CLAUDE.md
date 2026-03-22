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
| `SQLiteBackend` | 경량 프로덕션 | aiosqlite |
| `PostgreSQLBackend` | 프로덕션 | asyncpg, pgvector |
| `Neo4jBackend` | 그래프 탐색 | neo4j |
| `QdrantBackend` | 벡터 검색 | qdrant-client |
| `MinIOBackend` | 대용량 콘텐츠 | miniopy-async |
| `CompositeBackend` | 용도별 분리 | Neo4j + Qdrant + MinIO |

## 테스트

### 인프라 요구사항
```bash
# Neo4j (테스트용 인증: neo4j/password)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5-community

# Qdrant
docker start qdrant  # 또는 docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# PostgreSQL — 기존 컨테이너 사용 (ailab:ailab123@localhost:5432/plateerag)
# MinIO — 기존 컨테이너 사용 (localhost:9000)
```

### 실행
```bash
# 전체 테스트 (281건)
uv run pytest tests/ -v

# PostgreSQL 제외 (asyncpg 미설치 시)
uv run pytest tests/ --ignore=tests/test_backend_postgresql.py -v

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
    └── klue_mrc.json              # KLUE-MRC (5.8K corpus, 5.8K queries)
```

### 외부 데이터셋 다운로드
```bash
uv run python tests/benchmark/download_datasets.py
```
- MIRACL, Mr. TyDi는 HuggingFace datasets 호환 이슈로 현재 skip

### 외부 데이터셋 벤치마크 결과 (FTS only, MemoryBackend)
| 데이터셋 | Corpus | Queries | MRR | nDCG@10 | R@10 |
|----------|--------|---------|-----|---------|------|
| Allganize RAG-Eval | 300 | 300 | 0.796 | 0.811 | 0.863 |
| Allganize rag-ko | 200 | 200 | 0.780 | 0.797 | 0.855 |
| HotPotQA-24 | 226 | 24 | 0.754 | 0.636 | 0.729 |
| HotPotQA-200 | 1990 | 200 | 0.742 | 0.599 | 0.652 |
| AutoRAGRetrieval | 720 | 114 | 0.646 | 0.681 | 0.798 |
| KLUE-MRC | 500 | 100 | 0.607 | 0.643 | 0.760 |
| PublicHealthQA | 77 | 77 | 0.342 | 0.390 | 0.558 |
| Ko-StrategyQA | 9,251 | 100 | 0.315 | 0.261 | 0.293 |

### 자체 시나리오 벤치마크 결과 (v0.5.0 + 검색 개선)
| 지표 | Baseline | 개선 후 |
|------|----------|--------|
| MRR | 0.326 | **0.477** (+46%) |
| Mean P@5 | 0.160 | **0.227** (+42%) |
| Mean R@5 | 0.467 | **0.533** (+14%) |
| Mean nDCG@5 | 0.351 | **0.431** (+23%) |
| Hit rate | 9/15 | **13/15** |

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
4. Spreading activation: depth 1→2, 다중 경로 보상
5. AgentSearch past_failures: LESSON 노드 포함, fallback 추가

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
| 지표 | 현재 (최고) | 목표 | 근거 |
|------|-------------|------|------|
| MRR (Allganize) | 0.796 | 0.85+ | FTS+embedding 결합 시 |
| MRR (HotPotQA-200) | 0.742 | 0.80+ | spreading activation 개선 시 |
| MRR (Ko-StrategyQA) | 0.315 | 0.50+ | embedding 필수 (9K corpus) |
| 자체 시나리오 MRR | 0.477 | 0.65+ | ontology+embedding 결합 |

## 방향성
- 플래티어 온톨로지 비전과 연계: 엔터프라이즈 시맨틱 레이어
- 정적 검색 → 동적 메모리 (사용 경험에 따라 재편)
- 다층 랭킹: relevance + importance + recency + vitality
- Hebbian 학습: 함께 쓰이는 지식 연결 강화, 실패 패턴 약화
- 선제적 활성화: 태스크 맥락 기반 proactive loading
- **차별점**: 단순 retrieval이 아닌 "에이전트 경험 메모리" — 검색+학습+적응의 통합
