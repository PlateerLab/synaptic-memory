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

## 1. Phase 1 — 엔진 개선

목표: finreg multi-hop 73% → 90%+. 현재 실패 33/120 공략.

### 측정 결과 (2026-05-17)

| 단계 | finreg multi-hop (120q) |
|---|---:|
| vanilla RAG | 0% |
| agent — REFERENCES 엣지 기본 (Step 7 확장만) | 73% |
| agent — **WS-B 완성** | **83%** (+10pp, 커밋 8ce9827) |
| agent — WS-B + WS-D | 82% (−1, 노이즈) |

- **WS-B 채택.** GraphExpander Step 2 우선순위 + reranker 참조-동반 lift +
  aggregator 참조-동반 묶음 선택. 73→83% 검증됨.
- **WS-D 기각 — measured null.** tool 결과에 REFERENCES 명시 노출 +
  get_document citation 힌트. 100→98/120, 노이즈 범위. 효과 없음 →
  코드 되돌림. (CONCEPTS §13 측정 규율: 효과 없는 mechanism 은 ship 안 함.)
- **WS-C 보류.** 남은 20 실패의 양상이 셋으로 분기 (search-level 진단):
  agent-loop 7 (A·B 둘 다 검색되나 agent 실패) / 능동 추종 필요 7 / 진입
  검색 실패 6. 단일 근본원인이 없어 추측 없이 고칠 수 없음. multi-hop
  0%→83% 로 헤드라인 목표는 이미 달성 — 83→90 은 수익체감 구간.

**결론**: Phase 1 = WS-B 단독 채택. multi-hop RAG 0% vs Synaptic 83%,
single-hop 94% 동률. "agent 가 RAG 를 압도" 명제 측정 완료.

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

## 2. Phase 2 — 관계 강화 일반화 ✅ 완료

finreg-전용 `link_finreg_references.py`를 corpus-agnostic 메커니즘으로 승격.

### WS-A: StructuralReferenceLinker ✅ (commit 5069a06, ee785eb)

- 범용 extension `structural_reference_linker.py`. DomainProfile이
  `reference_key_property` (+ optional `reference_scope_property`)만
  선언하면 동작.
- **매처 자동 도출** — corpus의 실제 key 값들("제1조"…"제561조")에서
  alternation 매처를 빌드. corpus별 정규식 손수 작성 불요. (하드코딩 회피)
  `reference_token_pattern`은 surface-form 불일치 시의 optional override.
- **clean-target gate** — key 충돌률 > 10%면 자기 차단 + no-op.
  ReferenceLinker(v0.23) measured negative의 교훈을 코드로 강제.
- `DocumentIngester.ingest()`에 hook — `from_data()` 포함 모든 인제스트
  경로가 자동 적용. v0.23 ReferenceLinker 대체.

**실측 검증 (실제 그래프, end-to-end)**:
- finreg (clean target inventory): 8,427 REFERENCES 엣지 생성. 손으로 짠
  전용 스크립트(8,393)와 동등.
- KRRA (clean citation-key 없음): 게이트가 3가지 경우 모두 안전하게
  no-op — 설정 없음 / key=category (99% 충돌) / key=doc_id (94% 충돌).

→ "프로파일에 식별자 속성 1~2줄 + `from_data()` → multi-hop 관계 자동
구축; 부적합 corpus 엔 무해한 no-op" 가 코드로 참.

추가 관계종 (별표 ANNEX, cross-law, 법률↔시행령 IMPLEMENTS)은 후속 과제.

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
