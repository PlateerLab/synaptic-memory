# PLAN v0.24 — Relation Enrichment (대형 고도화 플랜)

상태: **active** — 개발 페이즈. 측정은 각 Phase 종료 시 1회.

## 0. 이 플랜이 서게 된 근거 (측정된 사실)

finreg 금융 법령 corpus (4,417 조문, law.go.kr) 위에서 vanilla RAG와
Synaptic agent를 같은 LLM-judge로 head-to-head 측정:

| 데이터셋 | vanilla RAG | Agent (관계 엣지 없음) | Agent (REFERENCES 엣지) |
|---|---:|---:|---:|
| single-hop (120q) | 94% | 94% | 94% |
| **multi-hop (120q)** | **0%** | **25%** | **73%** |

multi-hop GT는 FTS 검증으로 "single-shot RAG가 구조적으로 못 푸는 질의"만
채택 — 인용 대상 조문 B가 질의 어휘에 노출되지 않음.

### 핵심 발견

1. **single-shot 검색은 개선 여지가 없다.** RAG single-hop 94% = ceiling.
   CONCEPTS §13.6 ("mechanism 추가 ≠ 품질 개선")이 다시 확인됨.
2. **agent 루프가 해자(moat)다.** multi-hop RAG 0% vs agent 73%.
3. **그래프 관계가 성능을 지배한다.** REFERENCES 엣지 1종 추가 = +48pp.
   그래프가 명시적 관계를 가질수록 agent가 강해진다.
4. **ReferenceLinker(v0.23)는 틀린 게 아니라 corpus를 잘못 만났다.** KRRA의
   70k 노이즈 phrase-hub에선 target 해소 불가 → measured negative. finreg는
   조문마다 정규 `article_no` → 같은 메커니즘이 작동. **관계 추출 품질은
   전적으로 target inventory의 깨끗함에 달려 있다.**
5. **agent 루프에 낭비가 있다.** multi-hop 120q 중 94q가 5턴을 다 소진.
   실패한 tool 호출이 budget을 태움 (KRRA Conv 진단).
6. 측정 하베스를 병렬화 → 5시간 → ~14분. 개발 iteration이 가능해짐.

### 원칙 진화 제안

CLAUDE.md의 "relation-free graph"는 다음으로 갱신한다:

> **LLM-free graph + clean-target relations.** 인덱싱에 LLM을 쓰지 않는다는
> 원칙은 유지한다. 단, (a) LLM 없이 규칙으로 추출 가능하고 (b) 깨끗한 target
> inventory가 있는 관계는 그래프에 명시적으로 박는다 — measured +48pp.

## 1. Phase 1 — 엔진 개선 (측정 1회: 끝에서)

목표: finreg multi-hop 73% → **90%+**. 현재 실패 33/120 공략.

### WS-B: 참조-인지 reranking  [최우선]

가설: 실패 33건의 상당수는 GraphExpander가 인용 조문 B를 후보로 끌어오지만
HybridReranker에서 탈락 — B는 설계상 query와 어휘 겹침 0, semantic 낮음,
graph 0.20만으로 top-k 생존 불가.

- HybridReranker에서 `reason="references"`로 확장된 노드에 graph score
  floor / boost 부여.
- seed가 hit이고 그 seed에서 REFERENCES로 끌려온 노드는 "동반 증거"로
  per-document cap과 무관하게 top-k 1슬롯 보장 검토.
- 측정: multi-hop에서 B가 후보엔 있지만 top-k 밖인 비율을 먼저 계측.

### WS-C: agent 루프 하드닝

- 실패한 tool 호출 처리 — 에러 응답 시 동일 호출 재시도 대신 대안 경로
  유도. budget 소진 방지 (KRRA Conv c012/c013/c023/c024 회귀의 잔여 원인).
- turn budget — 94/120이 5턴 소진. adaptive budget 또는 planning 개선으로
  턴 효율 상승.
- `get_document` chunk-only corpus 복구는 v0.24 직전 commit에서 완료.

### WS-D: 참조를 agent 추론에 노출

GraphExpander는 REFERENCES를 retrieval에 수동 반영한다. agent가 능동적으로
쓰게 한다:

- `search` / `get_document` 결과에 "이 문서가 인용하는 문서: [...]"
  annotation 추가.
- agent가 `follow(node, "references")`를 쓰도록 tool hint 강화.

## 2. Phase 2 — 관계 강화 일반화 (측정 1회)

finreg-전용 `eval/datasets/link_finreg_references.py`를 재사용 가능한
범용 메커니즘으로 승격.

### WS-A: StructuralReferenceLinker

- 범용 extension: DomainProfile이 `reference_patterns`(정규식) +
  `target_resolver`(clean target index 빌더) 스펙을 주입.
- **clean-target gate** — target inventory가 깨끗하지 않으면 no-op.
  ReferenceLinker(v0.23) measured negative의 교훈을 코드로 강제.
- 추가 관계종:
  - 별표·서식 참조 (ANNEX)
  - cross-law 참조 (「은행법」 제X조 → 타 법령 조문)
  - 법률 ↔ 시행령 ↔ 시행규칙 위임 관계 (IMPLEMENTS)
- GraphExpander `_expand_references`는 이미 범용 — 재사용.
- v0.23 ReferenceLinker는 이 메커니즘으로 흡수/대체 검토.

## 3. Phase 3 — 규모 & 일반성 검증 (측정 1회)

### WS-E: corpus 확장 + 타 corpus 적용

- finreg에 금융위 소관 행정규칙(고시·훈령·예규) 추가 → 1만+ 조문.
- REFERENCES 강화를 KRRA 등 기존 corpus에 적용해 일반성 검증.
- ~10만 노드 SQLite 백엔드 스케일 실측.

## 4. 측정 규율

- Phase 단위로만 측정 (Phase 종료 시 1회). 변경마다 측정하지 않는다.
- 측정 시 `eval/rag_baseline.py` + `eval/run_all.py --agent`로 항상
  RAG vs agent를 병기. agent 단독 숫자는 무의미.
- 하베스는 `--agent-concurrency`로 병렬 (기본 12, finreg는 16).

## 5. 산출물 (이번 세션에서 이미 만든 것)

- `eval/datasets/build_finreg.py` — law.go.kr 헤드리스 스크래퍼
- `eval/datasets/ingest_finreg.py` — 조문 → 그래프
- `eval/datasets/gen_finreg_queries.py` — 검증 가능 GT 생성 (근거+정답 포함,
  multi-hop은 FTS-verified RAG-hard)
- `eval/datasets/link_finreg_references.py` — REFERENCES 엣지 추출
- `eval/rag_baseline.py` — vanilla RAG head-to-head 러너
- `EdgeKind.REFERENCES` + GraphExpander `_expand_references`
- `run_agent_benchmark` 병렬화 + multi-hop strict 채점
