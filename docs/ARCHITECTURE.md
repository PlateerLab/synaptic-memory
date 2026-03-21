# Synaptic Memory — Architecture

## 개요

뇌의 신경망에서 영감받은 knowledge graph 라이브러리.
FTS 키워드 매칭만으로 못 잡는 연관 지식을 Spreading Activation으로 발견하고,
성공/실패 경험에서 Hebbian Learning으로 자동 학습하며,
L0→L3 Memory Consolidation으로 원시 데이터는 소멸, 검증된 지식은 영구 보존.

## 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│              SynapticGraph (Facade)                       │
│  add() · link() · search() · reinforce() · consolidate() │
└──────┬───────────┬──────────┬───────────┬────────────────┘
       │           │          │           │
  ┌────▼───┐  ┌───▼──────┐  ┌▼────────┐  ┌▼──────────────┐
  │ Store  │  │ Hybrid   │  │Hebbian  │  │Consolidation  │
  │ (CRUD) │  │ Search   │  │Engine   │  │Cascade        │
  │        │  │          │  │         │  │               │
  │ tag    │  │ 3-stage  │  │ co-act  │  │ L0 → L1 → L2 │
  │ extract│  │ fallback │  │ +/- wt  │  │ → L3          │
  └───┬────┘  └────┬─────┘  └────┬────┘  └───────┬───────┘
      │            │              │                │
      │       ┌────▼─────┐       │                │
      │       │Resonance │       │                │
      │       │ 4-axis   │       │                │
      │       │ scoring  │       │                │
      │       └──────────┘       │                │
      │                          │                │
      └──────────┬───────────────┴────────────────┘
                 │
         StorageBackend (Protocol)
                 │
    ┌────────────┼────────────────┐
    │            │                │
┌───▼────┐  ┌───▼──────┐  ┌─────▼───────┐
│Memory  │  │SQLite    │  │PostgreSQL   │
│Backend │  │Backend   │  │Backend      │
│(dict)  │  │(FTS5+CTE)│  │(AGE+pgvec) │
└────────┘  └──────────┘  └─────────────┘
```

## 핵심 메커니즘

### 1. 하이브리드 3단계 검색

```
Query: "배포 실패 원인"
  │
  ├─ Stage 1: FTS + Fuzzy + Vector (병렬)
  │   ├─ FTS5 "배포" OR "실패" OR "원인" → score 0.8
  │   ├─ Fuzzy LIKE '%배포%' → score 0.6
  │   └─ Vector cosine(embedding) → score 0.7
  │   (결과 부족하면 ↓)
  │
  ├─ Stage 2: 동의어 확장
  │   └─ "배포" → "deploy", "deployment", "릴리즈"
  │      재검색 → score 0.5
  │   (결과 부족하면 ↓)
  │
  └─ Stage 3: Query Rewriter (LLM, optional)
      └─ Haiku: "배포 실패 원인" → ["CI/CD 파이프라인 에러", "rollback 이슈"]
         재검색 → score 0.4

  → Spreading Activation: top-5 → depth-1 이웃 확장
  → Resonance Scoring: 4축 점수로 최종 정렬
```

### 2. Hebbian Learning

"함께 활성화된 뉴런은 연결이 강화된다" (Hebb's Rule)

```
reinforce([node_A, node_B, node_C], success=True)
  │
  ├─ 각 노드: success_count += 1, access_count += 1
  │
  └─ 노드 쌍 간 edge:
      A↔B: weight += 0.1 (성공)
      A↔C: weight += 0.1
      B↔C: weight += 0.1
      (엣지 없으면 새로 생성)

reinforce([node_A, node_B], success=False)
  │
  ├─ 각 노드: failure_count += 1
  │
  └─ A↔B: weight -= 0.15 (실패가 더 강하게 학습)
      weight 범위: [-2.0, 5.0] (Anti-resonance 지원)
```

### 3. 4축 Resonance Scoring

```
Score = 0.40 × relevance    (검색 점수)
      + 0.25 × importance   (성공률)
      + 0.20 × recency      (최신성, 일별 5% 감쇠)
      + 0.15 × vitality     (생존력)
```

- **Relevance**: 검색 엔진이 매긴 원점수 [0, 1]
- **Importance**: (success - failure) / access → 정규화 [0, 1]
- **Recency**: exp(-0.05 × days_since_update)
- **Vitality**: 주기적 decay (매일 ×0.95), 접근 시 회복

### 4. Memory Consolidation (L0 → L3)

```
L0_RAW (72시간 TTL)
  ├─ 72시간 내 3회 이상 접근 → L1 승격
  └─ 72시간 후 미접근 → 삭제

L1_SPRINT (90일 TTL)
  ├─ 10회 이상 접근 → L2 승격
  └─ 90일 후 미접근 → 삭제

L2_MONTHLY (365일 TTL)
  ├─ 성공 10회 이상 + 성공률 80%+ → L3 승격
  └─ 365일 후 미달 → 삭제

L3_PERMANENT
  └─ 영구 보존 (검증된 지식)
```

## 데이터 모델

### Node

| 필드 | 타입 | 설명 |
|------|------|------|
| id | str (16 hex) | UUID 기반 고유 ID |
| kind | NodeKind | concept, entity, lesson, decision, rule, artifact, agent, task, sprint |
| title | str | 노드 제목 |
| content | str | 본문 내용 |
| tags | list[str] | 태그 목록 |
| level | ConsolidationLevel | L0 → L3 |
| embedding | list[float] | 벡터 임베딩 (optional) |
| vitality | float [0,1] | 생존력 (주기적 decay) |
| access_count | int | 접근 횟수 |
| success_count | int | 성공 사용 횟수 |
| failure_count | int | 실패 사용 횟수 |
| source | str | 출처 (예: "sprint:sprint-123") |
| created_at | float | 생성 시간 (epoch) |
| updated_at | float | 최종 갱신 시간 |

### Edge

| 필드 | 타입 | 설명 |
|------|------|------|
| id | str | 고유 ID |
| source_id | str | 시작 노드 |
| target_id | str | 끝 노드 |
| kind | EdgeKind | related, caused, learned_from, depends_on, produced, contradicts, supersedes |
| weight | float [-2, 5] | 연결 강도 (Hebbian 학습으로 조정) |
| created_at | float | 생성 시간 |

## 백엔드별 기능 매트릭스

| 기능 | MemoryBackend | SQLiteBackend | PostgreSQLBackend |
|------|:---:|:---:|:---:|
| FTS 검색 | ✅ (word match) | ✅ (FTS5) | ✅ (tsvector) |
| Fuzzy 검색 | ✅ (SequenceMatcher) | ⚠️ (LIKE) | ✅ (pg_trgm) |
| Vector 검색 | ✅ (cosine) | ❌ | ✅ (pgvector HNSW) |
| 그래프 순회 | ✅ (BFS) | ✅ (recursive CTE) | ✅ (AGE Cypher) |
| 영속성 | ❌ (메모리) | ✅ (파일) | ✅ (서버) |
| 동시성 | ❌ | ⚠️ (WAL) | ✅ (MVCC) |
| 한글 지원 | ✅ | ✅ (unicode61) | ✅ (simple + trgm) |
| 의존성 | 없음 | aiosqlite | asyncpg + pgvector |
