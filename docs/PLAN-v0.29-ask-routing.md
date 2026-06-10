# PLAN v0.29 — 정직한 라우팅: `graph.ask()`

상태: draft-r2 (적대적 리뷰 반영) · 목표: "싼 single-shot으로 충분하면 에이전트를 부르지 않는다"를 측정 가능한 제품 기능으로 만든다.

## 1. 방향 재정립 — 왜 지금인가

측정된 사실 (2026-06-10 기준, 전부 반복 측정 또는 결정론적):

- **답변품질 동률, 비용 7.5×.** rag_vs_agent_answer **finreg_multihop 120q**×3런: naive-RAG 69.7/120 (0.581) vs agent 67.0/120 (0.558), Δ−2.2pp — 노이즈 플로어 ±8/120 내 동률. 비용은 ~220s vs ~1650s/run. **"agent가 답변품질에서도 이긴다" 가설 기각** (commit 169c863의 +7.5pp는 단일 40q 런 노이즈로 철회). 단, 이 parity의 측정 범위는 finreg_multihop 120q 한정 — `finreg.json` 120q는 agent 측정이 아직 없고, 기존 3런 산출물은 run별 aggregate 요약뿐이라 per-query 데이터가 없다 (E1에서 재실행으로 해소).
- **agent의 가치는 single-shot 0점 영역에 국한·압도적.** assort Hard 0.0→91%, X2BEE Hard 0.379→100% — 노이즈 플로어 수 배. 정형 집계/필터/FK조인과 진짜 bridge multi-hop.
- **그래프는 single-shot 랭킹 부스트가 아니다.** 시맨틱 신호 7연속 기각 (decomposer −10.6%@3.8×, inline phrase −6.6%, entity-linker R@5 −0.020, ReferenceLinker 정밀도 ~50%, calibration 4/5 회귀, phrase bridge Δ0@3.5×, expansion single-shot net-neg 15/16벤치). 그래프의 실측 가치 = (a) 정형 도구의 기반(typed nodes + FK), (b) agent 보행로 (+8.3pp, 94 vs 84/120).
- **MuSiQue 추격 종결.** R@5 0.453 vs HippoRAG2 0.747 — 해소엔 LLM triple = "LLM-free 인덱싱" 원칙 충돌. known limitation.

기각된 narrative 3건과 대체:

| 버린다 | 채택한다 |
|---|---|
| "그래프가 single-shot 점수를 올린다" | "그래프 = 정형 도구 기반(0.0→91%) + agent 보행로(+8.3pp)" |
| "agent가 항상 낫다" | "agent-required 영역만 승급 — **라우팅 자체가 제품**" |
| "multi-hop 정확도로 HippoRAG2와 경쟁" | "비용($0 인덱싱)·신선도(CDC)·결정론으로 경쟁" |

선언: synaptic-memory는 **에이전트의 데이터 측 메모리**다 — 운영 DB·CSV·문서를 LLM 비용 $0·결정론적 ID로 단일 그래프에 넣어 CDC로 살아있게 유지하고, 정직한 라우팅(`graph.ask()`)을 내장한 MCP 도구 서버. 시장 공백(P1 "운영 DB와 함께 사는 그래프", P2 "정형+비정형 단일 MCP 서버")과 측정 원장이 일치하는 유일한 포지션이다.

## 2. Stop / Keep / Start

**Stop**
- single-shot 시맨틱 신호 연구 전면 동결 (위 7건).
- orphan 5종 삭제 — `extensions/dual_level_search.py`, `embedder_hyde.py`, `reranker_colbert.py`, `reranker_llm.py`, `query_decomposer_llm.py` (src 참조 0 + measured negative).
- MuSiQue 추격 공식 종결 — `docs/PLAN-v0.18-architecture.md` Q2 트랙 닫음, README known-limitation 유지.
- "agent 항상 우월" 서사·agent-default(v0.18 Q1) 폐기.
- MCP 죽은 도구 deprecated 표기 (journaling 4, resonance 3, hebbian 2, ontology 2 등). **주의: MCP tool `agent_search`는 목록에서 제외** — `src/synaptic/mcp/server.py:1483`의 살아있는 Agent v1 주 진입점이며 1군(실측 9) 도구다. 죽은 것은 legacy *모듈* `src/synaptic/agent_search.py`(intent-based strategies)이므로 모듈 단위로 표기. 착수 전 deprecated 목록 전체를 server.py 실물과 1:1 대조해 확정. 표기 작업 자체는 표면 정렬(E5)과 함께 **v0.29.1**, 실제 제거는 v0.30.
- AutoRAG −0.100 자동 해소 시도 중단 — blend 0.1 수동 설정 문서화로 종결.

**Keep (무수정 동결)**
- `extensions/evidence_search.py` + `backends/sqlite.py` — 14벤치 MRR 0.615/0.647, ask() cheap 경로 본체. **비트 단위 무회귀가 v0.29 전체의 하드 게이트.**
- `agent_tools_structured.py` 4도구 + `extensions/table_ingester.py`/`db_ingester.py` + deterministic ID — assort Hard 0.0→91%의 실체.
- `agent_loop.py` 9도구 + efficiency directive(default-on) + agent 한정 graph expansion — 멀티턴 81.4%.
- `extensions/cdc/` (full↔cdc top-k 동일성 regression-lock) + `extensions/connectivity.py` backbone — P1 포지션의 기술 실체.
- sufficiency gate default-on (`agent_loop.py:1302 _judge_sufficiency`, **+3.2pp — 단일 A/B 69→72/93, 노이즈 플로어 내 참고치**, fail-open이라 무해) — 라우터 tier-1 트리거로 재사용. 단, tier-1이 cheap 경로의 RAG 합성 답변을 판정하는 것은 judge가 튜닝된 분포(agent 중간답변)와 다르므로, cheap-sufficient corpus에서의 false-escalation rate를 E2에서 측정한다.
- Kiwi/HWP/의존성 0, finreg REFERENCES + clean-target gate (73% vs HippoRAG2 31%).
- ≥3런/±8/120 측정 규율.

**Start (순서 고정 — 코드보다 측정 먼저, 측정보다 계측 먼저)**
1. [1주차] T1: token usage 집계 + `--out-jsonl` per-query 출력 머지 → **finreg{,_multihop} 240q × RAG/agent × 3런 재실행**으로 per-query JSONL 신규 생성 (H100 예산 명기, E1 전반)
2. [2주차] 라우팅 GT 빌더 + 신규 JSONL 위 McNemar + proxy 타당성 검사 (E1 후반 — 라벨 재설계 루프 1회 버퍼 포함)
3. [3주차] tier-0 신호별 AUC 하니스 — go/no-go 게이트 (E2 — 미달 시 보수 라우팅 확정 루프 버퍼 포함)
4. [4주차] `graph.ask()` + MCP `knowledge_ask` (E3)
5. [5주차] keystone 혼합 풀 검증 (E4)

컷라인: **E1+E2+E3+E4가 릴리스 최소선** — keystone 검증 없는 릴리스는 이 방향 전환의 의미가 없다. E2 go/no-go 미달 시 E3는 보수 라우팅(고확신 양성만 tier-0 승급, 나머지 tier-1 위임)을 **기본값**으로 축소해 일정을 지킨다. E5(표면 정렬)는 통째로 **v0.29.1로 이월** — 이미 독립 커밋 설계라 이월 비용 0.

## 3. Epics

### E1 — 라우팅 GT + 비용 계측 + 3런 per-query 데이터 재생성 (keystone 전제, M+)

목표: "이 쿼리는 agent가 필요한가"의 라벨과 tokens/query 비용 축을 확보한다. 전제 사실: 기존 3런 산출물(`examples/ablation/diagnostics/rag_vs_agent_answer_multirun_20260610.log`)은 run별 합계 12줄짜리 aggregate 요약뿐이고, `rag_vs_agent_answer.py`는 per-query 출력 자체가 없다 (`eval/unified.py:513 load_bench_log`의 `[qid] turns=N found=M hit=` 형식과도 무관). **따라서 기존 로그 소급은 불가능하며, per-query 데이터는 재실행으로 새로 만든다.**

태스크 (순서 고정):
- **T1 — 계측 먼저.** `src/synaptic/agent_loop.py` — `AgentSearchResult`(L407)에 `prompt_tokens: int = 0`, `completion_tokens: int = 0` 추가. `run_agent_loop`(L1341) 내 모든 `client.chat.completions.create` 응답에서 `resp.usage` 누적 (`_judge_sufficiency` L1302의 judge 호출 포함, usage 없으면 0 fail-open). 같은 커밋에서 `examples/ablation/rag_vs_agent_answer.py`에 RAG arm 합성 호출 usage 집계 + `--out-jsonl` per-query 결과(qid, arm, run, hit, judge, tokens) 출력 추가.
- **3런 재실행 (T1 머지 후, H100 예산 태스크).** `finreg_multihop.json` 120q + `finreg.json` 120q × {naive-RAG, agent} × 3런을 `--out-jsonl`로 재실행해 per-query JSONL 생성. 예산: agent arm 실측 ~1650s/120q-run 기준, 240q×3런 + judge ≈ **반나절 H100 점유 (~4–5h)** — 1주차 일정에 명시 반영. judge는 temp=0 + 응답 캐싱으로 실행. agent_solve confirmed 라벨(3런 중 ≥2)은 이 JSONL에서만 산출.
- `eval/routing_gt.py` 신설 — `build_routing_gt(out: Path) -> None`. 입력: (a) finreg 240q — **위 재실행 JSONL** (confirmed 축), (b) gt_datasets 200q의 `type` 라벨 — **타입은 데이터에서 enumerate** (하드코딩 금지; 실측 파일 전체 25종, assort_hard 단독 18종), agent 축은 v0.17.1 단일 런 로그(`eval/baselines/agent_20260419_020747.log`) 기반 **provisional 플래그**, (c) `krra_graph.json` 15q (`topk_inadequacy`/`current_answerable` — 라우팅 GT로 이미 설계된 셋), (d) AutoRAG 720q — agent 측정 없음, **hit-only 플래그** (cheap-sufficient 음성 대량 소스). 라벨: `single_shot_hit@5`(결정론) × `agent_solve` 2×2 → {cheap_sufficient, agent_required, unsolved, both} + 계층 플래그 {confirmed(3런), provisional(단일런), hit-only, unmeasured}.
- **assort Hard OOM 7건 처리** — v0.17.1 측정에서 16k context로 실행 자체가 안 된 7q는 history compaction/turn 예산 조정으로 재실행 시도. 실패 시 GT에 `unmeasured` 플래그로 분리하고 agent 축 분모에서 제외 (라벨 조작 금지).
- `eval/unified.py` — `mcnemar_paired(a, b) -> tuple[int, int, float]` 추가 + 신규 per-query JSONL 로더. **McNemar는 신규 JSONL 위에서만 수행** — −2.2pp parity의 per-query 분해(discordant pair) 리포트.
- **proxy 타당성 검사**: finreg confirmed 부분에서 hit@5 라벨 vs judge 정답 일치율 리포트. (id-reach ~80% vs 답변 55.8% 괴리 대응.)
- **held-out 분할**: GT를 qid-hash로 train/held-out 분할 (corpus 층화, 50/50). E2의 신호 어휘·임계값 튜닝은 train에서만, E2 go/no-go와 E3 머지 게이트 수치는 **held-out에서만** 보고. assort Easy 15q는 튜닝에 쓰지 않고 전량 held-out(precision 게이트 전용).

수용 게이트:
- GT ≥1,100q 라벨 생성, 계층 플래그(confirmed/provisional/hit-only/unmeasured) 전수 부착. 라벨 산출 자체는 입력 JSONL이 주어지면 결정론.
- assort Hard 40q: **전부 `single_shot_hit@5=0` AND ≥30q agent_required**(provisional — v0.17.1 실측 solved 30/33과 정합). unmeasured는 분모 제외.
- proxy 일치율 ≥80% — 미달 시 라벨 재설계(hit@5 대신 judge 기반)하고 E2 착수 보류 (2주차 버퍼에서 흡수).
- `tests/test_agent_token_usage.py`, `tests/test_routing_gt.py` green + 기존 1088+ 테스트 무회귀.

### E2 — tier-0 신호 AUC 하니스 (go/no-go 게이트, S)

목표: 결정론적 신호가 agent-required를 예측하는지 단일런으로 판정. 라우터 코드를 쓰기 *전에* 신호의 가치를 측정한다.

태스크:
- `eval/routing_signal_auc.py` 신설 — E1 GT 위에서 신호별 AUC + recall@precision 곡선. **AUC/recall 판정은 confirmed 라벨 부분집합에서만, 튜닝은 train 분할에서만, 보고 수치는 held-out에서만.** provisional/hit-only는 참고 리포트로 병기. 신호 (전부 코드에 이미 존재, 배선만):
  - s1 (1선, 임베더-독립): 정형 어휘 × typed-node 보유 — `eval/unified.py:classify_query`(L216)의 enumeration/aggregation/multi_hop 휴리스틱 + backend의 `_table_name` kind 존재 여부.
  - s2: single-shot hit=0 / top-k score 신호 — rerank std (`sweep_rerank_deadzone.py` 재활용), top1−top2 margin. **보조 신호** (임베더 의존 — deadzone 전례).
  - s3: `_table_name` row가 top-k에 등장 (structured corpus 감지).
- **tier-1 judge false-escalation 측정** — sufficiency judge(temp=0, 캐싱)를 cheap-sufficient corpus(AutoRAG 표본 + assort Easy)의 RAG 합성 답변에 적용해 false-escalation rate 산출. judge가 agent 중간답변 분포로 튜닝된 점에 대한 분포-이동 검사 (E3 end-to-end escalation 게이트의 입력).

수용 게이트 (결정론, 단일런, held-out):
- 신호 조합으로 **agent-required recall ≥0.90 (confirmed 한정) AND tier-0 escalation: AutoRAG ≤15% + assort Easy ≤20%** 달성 가능 여부 판정.
- 미달 시: 보수 라우팅(고확신 양성만 승급, 나머지는 tier-1 gate에 위임)으로 E3 스코프 축소 — 이 결정 자체가 E2의 산출물.

### E3 — `graph.ask()` + MCP `knowledge_ask` (keystone, M)

목표: 단일 진입점. tier-0 결정론 신호 → cheap 경로(search + 1회 합성) → tier-1 sufficiency gate 불충분 시 agent 승급.

태스크:
- `src/synaptic/router.py` 신설 — `@dataclass RouteDecision(route: Literal["single_shot","agent"], reasons: list[str], signals: dict)`; `decide_route(query, *, has_table_nodes, anchor=None, search_scores=None) -> RouteDecision`. E2에서 살아남은 신호만. LLM 호출 0.
- `src/synaptic/graph.py` — `async def ask(self, question: str, *, mode: str = "auto", k: int = 10, max_turns: int = 5) -> AskResult` 신설 (chat L1696 / search L1816 옆). cheap 경로: `self.search()` → `rag_vs_agent_answer.py`의 `_RAG_SYSTEM` 합성 프롬프트 이식 → `_judge_sufficiency` 재사용한 tier-1 게이트 → 불충분 시 `self.chat()` 승급. `AskResult`: answer, route, route_reasons, escalated, prompt_tokens, completion_tokens, evidence. `mode="search"|"agent"`로 강제 우회 제공.
- **tier-1 결정론화** — judge 호출에 temperature=0 + prompt-hash 응답 캐시 옵션. end-to-end escalation 게이트의 전제조건 (아래).
- `src/synaptic/mcp/server.py` — `@server.tool() knowledge_ask` 추가 (route 근거·토큰 비용을 응답에 명시).
- `src/synaptic/agent_loop.py` L21-24 docstring 정정 (36→9 도구, compare_search 제거).
- **conversational corpus별 정책** — KRRA Conv는 cheap 잔류 (agent −23pp 실측), assort Conv는 승급 허용 (+9pp 실측). 라우터에 corpus 단위 정책이 아니라 신호 기반으로 자연 분리되는지 GT에서 확인하고, 분리 안 되면 보수 정책(둘 다 tier-1 위임)을 채택.

수용 게이트:
- [결정론] `uv run python eval/run_all.py --quick --compare eval/baselines/qa_latest.json` — 14벤치 MRR **비트 단위 무회귀** (ask()는 신규 층, cheap 경로 무수정).
- [결정론, 단일런, held-out] **tier-0 라우팅** (LLM 무관 층):
  - assort Hard: **구조 연산 타입(aggregation*·filter·cross_table·ontology_*·temporal·exhaustive 등 — routing_gt가 enumerate) 전수 승급 + 전체 ≥90% 승급** (40q 기준 ≥36/40 상당). conversational/paraphrase류 11q는 승급 미요구 (승급해도 벌점 없음 — assort Conv는 agent 이득 영역).
  - agent-required recall ≥0.90 (confirmed 라벨 한정), krra_graph 15q 중 agent-required 전부 승급.
  - **escalation 예산 (양방향)**: AutoRAG 720q ≤15% **AND assort Easy 15q ≤20%** — 문서 corpus와 정형 corpus 양쪽에 cheap-sufficient precision 게이트를 대칭으로 걸어 "_table_name이면 전부 승급"류 퇴화 라우터를 차단.
- **end-to-end escalation** (tier-0 + tier-1 judge 합성 — LLM 개입이므로 결정론 표기 금지): judge temp=0 + 응답 캐싱을 전제로 단일런 판정. 캐싱 불가 환경이면 3런 밴드(±)로 판정. 예산은 tier-0와 동일 한도.
- [결정론] empty answer 0 유지 (169c863 강제 합성 경로 보존).
- `tests/test_router.py`(신호별 라우팅 단위), `tests/test_graph_ask.py`(mock client로 escalation/우회/토큰 집계) green.

### E4 — keystone 검증: 혼합 풀 cost-at-quality 리포트 (M)

목표: 방향 전체의 입증 또는 기각. **이 리포트 하나가 v0.29의 존재 이유다.** 설계 원칙: **퇴화 라우터(always-RAG, always-agent)가 둘 다 가시적으로 패배할 수 있는 풀**이어야 한다 — finreg 단독은 agent≈RAG parity 영역이라 always-RAG가 게이트를 전부 통과해버리므로, agent-required 질량을 풀에 명시적으로 섞는다.

태스크:
- `examples/ablation/cost_at_quality.py` 신설 — **혼합 풀**: finreg 240q (cheap-sufficient 질량, `eval/data/finreg_graph.sqlite`) + **assort Hard 40q** (agent-required 질량, `eval/data/assort_graph.sqlite`, 기존 LLM-judge 골격 재사용) + 가능하면 X2BEE Hard 20q. 4 arm: always-RAG / always-agent / **oracle router**(GT 라벨로 상한 계산, 실행 불요) / `ask()`. x=tokens/query, y=solve. finreg의 always-RAG/always-agent arm은 E1 재실행 JSONL 재사용 — 신규 GPU 비용은 assort/X2BEE arm + ask() arm × 3런.

수용 게이트:
- [3런 + per-query McNemar] ask() solve ≥ max(always-RAG, always-agent) − 2q.
- [3런 + McNemar] **혼합 풀에서 ask() solve > always-RAG solve — agent-required 클래스 기여분을 discordant pair로 입증.** 이 게이트가 "라우팅이 제품"의 직접 검증: always-RAG는 assort Hard에서 품질로 패배하고(single-shot 0.0), always-agent는 finreg에서 비용으로 패배해야 하며, ask()는 양쪽 모두와 분리돼야만 통과한다.
- [결정론] 평균 비용 ≤0.35× always-agent (tokens/query).
- oracle 대비 격차 공개. 클레임 한정: agent-required 클래스 내 델타만 노이즈 플로어 수 배(0→91%급)로 주장. 도메인 일반화(finreg+assort 2도메인)는 미입증으로 명시.

### E5 — 표면 정렬 + P1 증명물 (M, **v0.29.1로 이월** — 독립 커밋, 본 릴리스 게이트와 무관)

태스크 (이월 후에도 명세 유지):
- `README.md` — `ask()`/`chat()` 승격 (81.4%를 만든 chat()이 현재 0회 등장), 7.5×·±8/120 정직 표기, L331-344 파이프라인 다이어그램에서 PPR/GraphExpander/MENTIONS의 single-shot 랭킹 광고 제거 → "agent 보행로" 재배치, "언제 search/언제 ask" 기준 명시.
- `pyproject.toml` keywords — hebbian-learning/memory-consolidation/evidence-chain 제거.
- `src/synaptic/mcp/server.py` — 42도구 description에 계층 표기: 1군(실측 9 — **`agent_search` 포함**) / 운영(ingest·CDC·edit ~18) / deprecated. **착수 전 deprecated 목록을 server.py 실물과 1:1 대조해 확정** (Stop 절 정정 반영: legacy 모듈 `src/synaptic/agent_search.py`는 모듈 단위 표기, MCP tool과 혼동 금지). 시그니처는 외부 계약이므로 유지.
- orphan 5종 삭제 + `examples/quickstart.py`의 `graph._backend` private close → `graph.close()` 수정.
- `examples/cdc_live_demo.py` 신설 — `from_database(mode="cdc")` → 소스 변경 → `sync_from_database` → `ask()`, 인덱싱 $0 + full↔cdc top-k 동일성 측정표 출력. **P1 포지션의 증명물.**
- `docs/COMPARISON.md` — vs mem0/Zep/GraphRAG/LightRAG 대비표 (비용·신선도·결정론·정형+비정형 축. multi-hop 정확도 주장 금지).

수용 게이트: 전체 테스트 green, README 수치는 전부 측정 원장 인용 가능, CDC 데모 결정론적 재현.

## 4. 첫 스프린트 (오늘 착수)

**T1 (오늘): token usage 집계 + per-query 출력.** `src/synaptic/agent_loop.py:407` `AgentSearchResult`에 `prompt_tokens: int = 0` / `completion_tokens: int = 0` 필드 추가 → `run_agent_loop` 메인 루프와 `_judge_sufficiency`(L1302) 호출부에서 `getattr(resp, "usage", None)` 누적 (없으면 0, fail-open) → `tool_log` 패턴(L439)과 동일하게 "항상 채움, 저비용". 테스트: `tests/test_agent_token_usage.py` 신설 — **`tests/test_agent_efficiency.py`의 fake-client stub 패턴 재사용**, usage 객체 2턴 반환 → 합산 검증 + usage 부재 시 0 검증. 같은 커밋에서 `examples/ablation/rag_vs_agent_answer.py`에 `--out-jsonl` per-query 출력(qid, arm, run, hit, judge, tokens) 추가.

**T2: 3런 per-query 재실행 (T1 머지 직후, H100 예산).** finreg_multihop 120q + finreg 120q × {RAG, agent} × 3런 → JSONL. 기존 multirun 로그는 aggregate 12줄뿐이라 소급 불가 — 이것이 신규 측정인 이유. 예산: agent arm 실측 ~1650s/120q-run × 6런 + judge(temp=0, 캐싱) ≈ **H100 반나절(~4–5h)**, RAG arm은 ~4분/run으로 무시 가능. 1주차에 배타 슬롯으로 확보.

**T3: `eval/routing_gt.py`** — E1 명세대로 (타입 enumerate, 계층 플래그, OOM 7건 처리 포함). 검증: `tests/test_routing_gt.py` (소형 fixture JSONL로 2×2 라벨·플래그 정확성).

**T4: McNemar** — `eval/unified.py`에 `mcnemar_paired` + T2 신규 JSONL 적용, −2.2pp parity의 per-query 분해 리포트. 테스트는 `tests/test_eval_unified.py`에 케이스 추가.

**T5: proxy 타당성 검사** 실행 → 결과에 따라 E2 진입 또는 라벨 재설계 (2주차 버퍼).

H100 경합 주의: **T2가 유일한 1주차 GPU 소비처 (~4–5h)** — 나머지 T1/T3/T4/T5는 결정론/저비용. E4의 신규 arm 측정(assort/X2BEE + ask() × 3런)이 5주차의 두 번째 GPU 슬롯.

## 5. 측정 계획

신규 하니스 3개 + 기존 재사용:

| 측정 | 하니스 | 노이즈 | 판정 방식 |
|---|---|---|---|
| per-query 3런 JSONL (finreg 240q) | `rag_vs_agent_answer.py --out-jsonl` (T1/T2) | judge+agent | 3런, agent_solve = ≥2/3 |
| 라우팅 GT 라벨 | `eval/routing_gt.py` (신설) | 없음 (입력 JSONL 고정 시 결정론) | 단일런, 계층 플래그 명시 |
| 신호 AUC / recall@precision | `eval/routing_signal_auc.py` (신설) | 없음 | 단일런 go/no-go — train 튜닝 / **held-out 판정**, confirmed 한정 |
| tier-0 escalation | GT 위 결정론 평가 | 없음 | 단일런 (AutoRAG ≤15%, assort Easy ≤20%) |
| end-to-end escalation | tier-0 + tier-1 judge | judge (LLM) | temp=0+캐싱 전제 시 단일런, 불가 시 3런 밴드 |
| tier-1 false-escalation (cheap corpus) | E2 1행 측정 | judge | temp=0+캐싱, 단일런 |
| cost-at-quality 4점 | `examples/ablation/cost_at_quality.py` (신설, 혼합 풀) | judge+agent | 3런 + McNemar |
| 14벤치 무회귀 | `eval/run_all.py --quick --compare` (기존) | 없음 | 비트 동일 |
| 비용 | tokens/query (T1 신설 — 현재 wall-clock뿐) | 없음 | 결정론 |

judge 안정화는 보조가 아니라 **게이트 전제조건**: temp=0 + 응답 캐싱 + judge-vs-id-hit 일치율 리포트 — naive-RAG arm spread 3q가 judge 노이즈 하한 추정치.

## 6. 명시적 비목표 (v0.29.1 / v0.30+ 이월)

- **E5 표면 정렬 전체 (README·keywords·도구 계층 표기·orphan 삭제·CDC 데모·COMPARISON.md)** — v0.29.1로 이월. 독립 커밋 설계라 이월 비용 0, 본 릴리스 게이트와 무관.
- **deprecated MCP 도구 실제 제거 + legacy `search.py`/`resonance.py`/`hebbian.py`/`synonyms.py`/`agent_search.py` 모듈 삭제** — v0.29.x는 표기만. 외부 계약 파괴는 별도 메이저 정리로.
- **KRRA Conv −23pp 원인 규명** — v0.29 라우터의 conversational 정책은 corpus별로 분리: **KRRA Conv만 cheap 경로 잔류** (승급 시 악화가 측정으로 확인됨), assort Conv는 승급 허용 (+9pp 실측). 원인 규명 자체는 이월.
- **connectivity backbone → agent solve 효과 측정** — ≥3런 필수, H100 경합으로 이월. 구조 보장(28.9%→1.4%)은 이미 증명, 정확도 주장은 안 함.
- **per-embedder calibration 자동화** — deadzone/adaptive 신호는 보편이나 효과가 임베더 의존 (dz≈2는 Qwen 한정).
- **KRRA 잔여 1.4% islands entity embedding** — v0.27 phrase-dense 중립 전례로 기대값 불확실.
- **외부 도메인 answer GT 구축** — finreg+assort 2도메인 종속 해소. E4 클레임 한정의 해제 조건.
- **MuSiQue/triple 정책 재론** — "LLM-free 인덱싱" 원칙 변경 결정이 선행돼야 함. 원칙 유지 시 영구 known-limitation.
- **graph.py(2515L) god-module 분해** — ask() 추가로 더 커지지만, facade 분해는 라우팅 입증 후의 구조 작업.
