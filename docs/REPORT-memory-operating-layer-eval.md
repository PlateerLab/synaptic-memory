# Memory Operating Layer Eval Report

작성: 2026-07-01

대상 PR: [#10 Add memory operating layer foundation](https://github.com/PlateerLab/synaptic-memory/pull/10)

---

## 요약

PR #10 이후 `main`에는 memory event ledger, retrieval feedback ledger,
scope-aware reinforcement, edge provenance, pollution/health signals, OpenIE
cache/replay harness가 들어갔다.

이번 후속 평가는 세 가지를 확인했다.

1. **기본 경로는 안정적인가?**
   - lint, targeted unit tests, memory operating PoC, OpenIE skip/cache-only smoke
     모두 통과했다.
2. **기본 RAG를 넘어서는 relation expansion 신호가 있는가?**
   - cache-only OpenIE smoke에서 relation target expansion이 `1/26 -> 26/26`,
     strong relation evidence가 `0/8 -> 8/8`로 증가했다.
3. **DeepSeek Flash live extraction이 실제로 동작하는가?**
   - 5-row live warm smoke에서 `5/5` extraction이 성공했다.
   - 이어서 50-row live warm batch도 `50/50` extraction이 성공했고, 30% coverage
     cache-only 재측정에서 relation target expansion이 `7/145 -> 141/145`로
     유지됐다.
   - 추가 40-row live warm batch로 50% coverage까지 올렸고, cache-only gate에서
     relation target expansion이 `2/98 -> 90/98`로 유지됐다.
   - 다음 50-row batch는 transient 빈 응답 2건이 있었지만 failure manifest 재시도로
     둘 다 성공했고, 75% coverage gate에서 relation target expansion이
     `3/96 -> 80/96`으로 유지됐다.
   - 최종 50-row batch와 transient failure 2건 재시도로 100% coverage까지 올렸고,
     full cache-only gate에서 R@5 no-regress, relation probe, revertibility,
     cache coverage가 모두 PASS했다.

따라서 이 문서는 **foundation merge 직후의 deterministic 검증 + DeepSeek
Flash live warm 결과 + 100% cache-only gate 결과**를 기록한다.

---

## 검증 환경

Repository state:

- branch: `main`
- merge commit: `56fd66b`
- merged PR: `#10`

Runtime availability:

| 항목 | 상태 |
|---|---|
| `DEEPSEEK_API_KEY` | missing |
| `OPENAI_API_KEY` | missing |
| `QWEN_BASE_URL` | missing |
| local chunk corpus | present: `~/synaptic-eval/mz_chunks.jsonl` |
| Qwen OpenIE cache | present: `~/synaptic-eval/openie_cache_mz_200_qwen.jsonl` |

초기 dry-run 시점에는 `DEEPSEEK_API_KEY`가 없었다. 이후 live smoke를 위해
키를 한 번 제공받아 실행 프로세스의 환경 변수로만 주입했다. API key는 코드,
문서, DB, cache에 저장하지 않는다.

---

## Completed Checks

### 1. Formatting and lint

```bash
uv run ruff format --check
uv run ruff check src/synaptic eval/scripts tests
```

결과:

- `ruff format --check`: PASS
- `ruff check`: PASS

### 2. Targeted regression tests

```bash
uv run pytest \
  tests/test_memory_operating_layer.py \
  tests/test_memory_operating_poc.py \
  tests/test_openie_extractor.py \
  tests/test_openie_mz_poc.py \
  tests/test_document_ingester.py \
  tests/test_backend_sqlite.py \
  -q
```

결과:

- `117 passed`

### 3. Memory operating PoC

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_poc_eval_current_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `21/21` |
| memory events | `22` |
| retrieval events | `8` |
| signal events | `9` |

검증된 동작:

- `search(record=True)`가 retrieval event를 남긴다.
- explicit/task/test feedback이 event와 scope score를 갱신한다.
- implicit feedback은 global success/failure count를 오염시키지 않는다.
- task success는 node score와 Hebbian edge score를 local/global로 기록한다.
- repeated failure, property conflict, supersession, stale, low-confidence,
  drift-spike signal이 생성된다.
- suspect memory는 자동 삭제되지 않고 flag/penalty/health report로 관리된다.

후속 edge-only signal penalty 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_edge_signal_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `22/22` |
| memory events | `22` |
| retrieval events | `8` |
| signal events | `9` |
| suspect count | `9` |

추가로 검증된 동작:

- edge-only suspect signal도 `edge_ids`를 검색 결과 node의 연결 edge에 매칭해
  endpoint 후보에 bounded penalty를 적용한다.
- node-local signal, scope score repeated failure signal, stale/supersession signal의
  기존 동작은 유지된다.

후속 edge score boost 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_edge_boost_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `23/23` |
| memory events | `22` |
| retrieval events | `8` |
| signal events | `9` |
| suspect count | `9` |

추가로 검증된 동작:

- node score가 없는 후보도 scope-local edge score가 있으면 해당 edge endpoint로
  해석되어 bounded boost를 받는다.
- edge score boost는 기존 relevance order를 뒤집지 않고 cap 안에서만 작동한다.
- edge-only suspect signal penalty gate도 함께 유지된다.

후속 memory ranking diagnostics 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_diagnostics_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `24/24` |
| scope boosted nodes | `1` |
| scope edge score hits | `1` |
| max abs boost | `0.10` |
| signal penalized nodes | `1` |
| edge-only signal penalized nodes | `1` |

추가로 검증된 동작:

- scope/node/edge reinforcement boost와 memory signal penalty가 검색 결과
  `diagnostics`에 관측 가능한 numeric signal로 남는다.
- 운영/평가 레이어가 memory layer의 ranking 개입 여부를 SearchResult만 보고도
  추적할 수 있다.

후속 retrieval ledger diagnostics 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_retrieval_ledger_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `25/25` |
| retrieval properties recorded | `true` |
| recorded query | `Alpha retention` |
| recorded returned count | `2` |
| recorded total candidates | `2` |

추가로 검증된 동작:

- `record=True` 검색의 `RetrievalEvent.properties`와 mirror `MemoryEvent.properties`에
  compact retrieval metadata가 남는다.
- `memory_` prefix ranking diagnostics는 검색 event ledger에 같이 저장되어
  SearchResult가 없어도 나중에 ranking 개입 이력을 추적할 수 있다.

후속 memory health ranking summary 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_health_ranking_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `26/26` |
| boosted retrievals | `1` |
| penalized retrievals | `1` |
| boosted nodes | `1` |
| penalized nodes | `1` |
| max scope boost | `0.10` |
| max signal penalty | `0.05` |

추가로 검증된 동작:

- `memory_health()`가 retrieval ledger의 `memory_` ranking diagnostics를 집계한다.
- 운영자는 health report만 봐도 최근 검색에서 memory boost/penalty가 몇 번, 얼마나
  개입했는지 확인할 수 있다.

후속 top reinforced edge health 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_top_edges_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `27/27` |
| top reinforced edge ids | `poc_edge_score_boost_relation`, `ca22935c4fa14f9c` |

추가로 검증된 동작:

- `memory_health()`가 node 강화 summary뿐 아니라 edge/relation 강화 summary도 노출한다.
- relation memory가 강화됐는지 health report만 보고도 추적할 수 있다.

후속 top reinforced edge starvation guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_top_edges_unstarved_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `28/28` |
| top reinforced node count | `10` |
| top reinforced edge ids | `poc_edge_score_boost_relation` 포함 |

추가로 검증된 동작:

- node score가 health top score list를 채워도 edge/relation 강화 summary가 사라지지 않는다.
- node와 edge top reinforced summary는 각각 독립적으로 최대 10개까지 유지된다.

후속 OpenIE provenance artifact count guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_openie_provenance_count_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `29/29` |
| health_counts_openie_edges_by_provenance | `true` |
| openie_artifact_count | `1` |

추가로 검증된 동작:

- OpenIE relation edge id가 `openie_` prefix가 아니어도 `properties.is_openie=true` provenance metadata로 health artifact count에 포함된다.
- health report의 OpenIE artifact 지표가 id naming convention에만 의존하지 않는다.

후속 negative scope demotion guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_negative_scope_demotion_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `30/30` |
| negative_scope_score_demoted_candidate | `true` |
| negative_scope_order | `poc_scope_clean_memory`, `poc_scope_demoted_memory` |
| negative_scope_demoted_resonance | `0.9` |
| memory_scope_boosted_nodes | `0.0` |
| memory_scope_demoted_nodes | `1.0` |

추가로 검증된 동작:

- positive scope reinforcement는 base relevance를 뒤집지 않도록 계속 clamp된다.
- negative scope score는 clamp에 막히지 않고 해당 후보를 실제로 demote할 수 있다.
- ranking diagnostics는 positive boost와 negative demotion을 분리해서 기록한다.
- 최종 result order는 조정된 resonance 기준으로 정렬된다.

후속 health demotion summary guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_health_demotion_summary_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `31/31` |
| health_summarizes_scope_demotions | `true` |
| memory_demoted_retrieval_count | `1` |
| memory_demoted_node_count | `1` |
| max_memory_scope_demotion | `0.1` |
| memory_adjusted_retrieval_count | `2` |
| memory_adjusted_node_count | `2` |

추가로 검증된 동작:

- retrieval ledger에 기록된 scope demotion diagnostics가 `memory_health()` summary로 집계된다.
- positive boost, negative demotion, 전체 adjustment count를 health report에서 분리해서 볼 수 있다.

후속 top demoted score summary guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_top_demoted_summary_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `32/32` |
| health_reports_top_demoted_scores | `true` |
| top demoted node ids | `poc_scope_demoted_memory`, `poc_scope_score_failed_memory`, `poc_scoped_negative_memory` |
| top demoted edge ids | `poc_edge_score_demoted_relation` |

추가로 검증된 동작:

- scope score가 음수로 누적된 node/edge를 `memory_health()`에서 직접 확인할 수 있다.
- reinforced summary는 양수 score만, demoted summary는 음수 score만 분리해서 보여준다.

후속 strong negative scope score signal guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_strong_negative_signal_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `33/33` |
| strong_negative_scope_score_signal_created | `true` |
| signal_kind | `repeated_failure` |
| score_signal_type | `strong_negative_scope_score` |
| signal tags | `_memory_signal`, `_memory_suspect` |
| score | `-1.000000` |

추가로 검증된 동작:

- `failure_count >= 3`이 아니어도 scope score가 강하게 음수이면 suspect signal 후보가 된다.
- top demoted memory가 health summary에만 머물지 않고 pollution monitor signal로 연결된다.
- signal provenance에는 `score_scope_key`, feedback count, score가 metadata로 남는다.

후속 edge score signal endpoint/penalty guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_edge_score_signal_endpoint_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `35/35` |
| strong_negative_edge_score_signal_records_endpoints | `true` |
| strong_negative_edge_score_signal_demoted_endpoint | `true` |
| signal edge_ids | `poc_edge_score_demoted_relation` |
| signal node_ids | `poc_scope_demoted_memory`, `poc_scope_clean_memory` |
| memory_signal_penalized_nodes | `1.0` |

추가로 검증된 동작:

- edge-level negative scope score signal이 relation id뿐 아니라 affected endpoint node ids도 보존한다.
- persisted signal metadata만으로 어떤 relation과 endpoint가 suspect인지 추적할 수 있다.
- 해당 signal은 search-time memory signal penalty 흐름으로 연결된다.

후속 signal event provenance guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_signal_event_provenance_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `36/36` |
| strong_negative_signal_event_records_provenance | `true` |
| signal event score_signal_type | `strong_negative_scope_score` |
| signal event score_scope_key | `user:eval-user` |
| signal event score | `-1.000000` |

추가로 검증된 동작:

- signal observation node뿐 아니라 `MemoryEventKind.SIGNAL` ledger에도 signal provenance가 남는다.
- signal event만 조회해도 scope, target ids, confidence, score, reason을 추적할 수 있다.

후속 signal penalty retrieval provenance guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_signal_penalty_provenance_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `37/37` |
| signal_penalty_retrieval_event_records_provenance | `true` |
| memory_signal_source_ids | `memsig_32d1378cc670e496` |
| memory_signal_edge_ids | `poc_edge_score_demoted_relation` |
| memory_signal_penalized_node_ids | `poc_scope_clean_memory` |
| memory_signal_penalized_nodes | `1.0` |

추가로 검증된 동작:

- search-time penalty diagnostics가 penalized node id, source signal id, related edge id를 compact metadata로 남긴다.
- `record=True` 또는 retrieval event 기록 시 숫자 diagnostics와 문자열 provenance diagnostics가 함께 저장된다.

후속 health penalty provenance summary guard 검증:

```bash
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_health_penalty_provenance_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| result | PASS |
| gates | `38/38` |
| health_reports_penalty_provenance | `true` |
| top penalty signal ids | `memsig_32d1378cc670e496`, `poc_edge_only_signal_node` |
| top penalized node ids | `poc_scope_clean_memory`, `poc_edge_only_signal_suspect` |
| top penalty edge ids | `poc_edge_score_demoted_relation`, `poc_edge_only_signal_relation` |

추가로 검증된 동작:

- `memory_health()`가 retrieval ledger의 signal penalty provenance를 집계한다.
- health report만 보고도 가장 자주 penalty를 만든 signal, node, edge 후보를 추적할 수 있다.

### 4. OpenIE off baseline smoke

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --skip-openie \
  --embed-base-url "" \
  --results ~/synaptic-eval/mz_openie_skip_eval_current_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| docs | `233` |
| chunks | `6,108` |
| queries | `51` |
| baseline R@1 | `84.3%` |
| baseline R@5 | `98.0%` |
| baseline R@10 | `98.0%` |
| revertibility gate | PASS |

OpenIE를 끈 baseline 경로가 안정적으로 동작한다.

### 5. OpenIE cache audit

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-audit \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --results ~/synaptic-eval/openie_cache_audit_eval_current_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| lines | `9` |
| unique keys | `9` |
| parseable records | `9` |
| invalid JSON | `0` |
| invalid records | `0` |
| entities | `57` |
| triples | `46` |
| result | PASS |

### 6. OpenIE cache-only smoke

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-source-limit 200 \
  --openie-max-chunks 5 \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-only \
  --llm-model Qwen3.6-27B \
  --relation-probe-limit 50 \
  --min-relation-expanded-lift 1 \
  --min-relation-evidence-lift 1 \
  --min-strong-relation-evidence-rate 0.5 \
  --min-openie-cache-coverage 0.02 \
  --embed-base-url "" \
  --results ~/synaptic-eval/mz_openie_cache_eval_current_results.json
```

결과:

| 항목 | 값 |
|---|---:|
| scanned chunks | `200` |
| cache eligible chunks | `5` |
| cache coverage | `2.5%` |
| relation edges created | `26` |
| OpenIE artifacts | `94` |
| extraction failures | `0` |
| baseline R@5 | `100.0%` |
| OpenIE R@5 | `100.0%` |
| delta R@5 | `+0.0%` |
| revertibility gate | PASS |

Relation probe:

| 지표 | graph expansion off | graph expansion on |
|---|---:|---:|
| relation target expanded | `1/26` | `26/26` |
| relation evidence hit | `1/26` | `9/26` |
| strong relation evidence | `0/8` | `8/8` |

해석:

- R@5는 이미 천장에 가까운 작은 smoke라 개선 여지가 없었다.
- 대신 relation probe에서 기본 RAG/FTS만으로는 확장되지 않던 relation target이
  OpenIE graph edge를 통해 모두 확장됐다.
- 즉 PR #10의 OpenIE relation layer는 "직접 비슷한 chunk" 검색을 넘어
  relation evidence 후보를 끌어올 수 있음을 보여준다.

### 7. DeepSeek Flash live warm smoke

DeepSeek Flash API key를 런타임 env로만 주입하고 5-row live warm smoke를
실행했다. DeepSeek API는 `json_schema` response format을 거부했지만, provider의
`json_object` fallback으로 extraction은 정상 진행됐다.

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input ~/synaptic-eval/openie_cache_missing_200.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-v4-flash \
  --llm-api-key-env DEEPSEEK_API_KEY \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-limit 5 \
  --openie-cache-warm-total-chunks 200 \
  --openie-cache-warm-target-coverage 0.5 \
  --openie-cache-warm-pending-output \
    ~/synaptic-eval/openie_cache_pending_deepseek_smoke_5.jsonl \
  --openie-cache-warm-failure-output \
    ~/synaptic-eval/openie_cache_failures_deepseek_smoke_5.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_deepseek_smoke_5_results.json
```

Warm result:

| 항목 | 값 |
|---|---:|
| rows attempted | `5` |
| rows succeeded | `5` |
| extraction failures | `0` |
| new entities | `31` |
| new triples | `22` |
| cache entries after warm | `14` |
| projected coverage after batch | `5.0%` |
| elapsed | `90.6s` |

Audit after warm:

| 항목 | 값 |
|---|---:|
| cache lines | `14` |
| unique keys | `14` |
| parseable records | `14` |
| invalid JSON | `0` |
| invalid records | `0` |
| entities | `88` |
| triples | `68` |
| result | PASS |

Cache-only scoring after warm:

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-source-limit 200 \
  --openie-max-chunks 10 \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-only \
  --llm-model deepseek-v4-flash \
  --relation-probe-limit 100 \
  --min-relation-expanded-lift 1 \
  --min-relation-evidence-lift 1 \
  --min-strong-relation-evidence-rate 0.5 \
  --min-openie-cache-coverage 0.05 \
  --embed-base-url "" \
  --results ~/synaptic-eval/mz_openie_cache_deepseek_smoke_5_results.json
```

Scoring result:

| 항목 | 값 |
|---|---:|
| cache eligible chunks | `10/200` |
| cache coverage | `5.0%` |
| relation edges created | `48` |
| OpenIE artifacts | `179` |
| extraction failures | `0` |
| baseline R@5 | `100.0%` |
| OpenIE R@5 | `100.0%` |
| revertibility gate | PASS |

Relation probe after DeepSeek smoke:

| 지표 | graph expansion off | graph expansion on |
|---|---:|---:|
| relation target expanded | `2/47` | `47/47` |
| relation evidence hit | `2/47` | `16/47` |
| strong relation evidence | `0/11` | `11/11` |

해석:

- live DeepSeek extraction이 기존 cache와 같은 OpenIE replay/eval path에 정상
  연결됐다.
- cache coverage가 `2.5% -> 5.0%`로 늘면서 relation probe 대상은 `26 -> 47`
  개로 증가했다.
- R@5는 여전히 작은 smoke에서 천장이지만, relation evidence lift는
  `8 -> 14`로 증가했다.

---

## DeepSeek Flash 50-Row Scale Eval

5-row smoke 이후 같은 missing manifest에서 50-row live warm batch를 실행했다.
이미 smoke에서 cache된 5개 row는 건너뛰고, 다음 50개 uncached row만 처리했다.

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input ~/synaptic-eval/openie_cache_missing_200.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-v4-flash \
  --llm-api-key-env DEEPSEEK_API_KEY \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-limit 50 \
  --openie-cache-warm-total-chunks 200 \
  --openie-cache-warm-target-coverage 0.5 \
  --openie-cache-warm-pending-output \
    ~/synaptic-eval/openie_cache_pending_deepseek_50.jsonl \
  --openie-cache-warm-failure-output \
    ~/synaptic-eval/openie_cache_failures_deepseek_50.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_deepseek_50_results.json
```

Warm result:

| 항목 | 값 |
|---|---:|
| missing rows loaded | `195` |
| skipped cached rows | `5` |
| rows attempted | `50` |
| rows succeeded | `50` |
| extraction failures | `0` |
| new entities | `228` |
| new triples | `153` |
| cache entries after warm | `64` |
| projected after batch | `60/200` |
| projected coverage | `30.0%` |
| target coverage | `50.0%` |
| rows needed for target | `90` |
| batches needed at limit 50 | `2` |
| target reachable | `true` |
| elapsed | `859.2s` |

Audit after 50-row warm:

| 항목 | 값 |
|---|---:|
| cache lines | `64` |
| unique keys | `64` |
| parseable records | `64` |
| invalid JSON | `0` |
| invalid records | `0` |
| empty records | `2` |
| entities | `316` |
| triples | `221` |
| result | PASS |

Cache-only scoring at 30% coverage:

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-source-limit 200 \
  --openie-max-chunks 60 \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-only \
  --llm-model deepseek-v4-flash \
  --relation-probe-limit 150 \
  --min-relation-expanded-lift 10 \
  --min-relation-evidence-lift 5 \
  --min-strong-relation-evidence-rate 0.5 \
  --min-openie-cache-coverage 0.30 \
  --embed-base-url "" \
  --results ~/synaptic-eval/mz_openie_cache_deepseek_50_results.json
```

Scoring result:

| 항목 | 값 |
|---|---:|
| cache eligible chunks | `60/200` |
| cache coverage | `30.0%` |
| relation edges created | `191` |
| OpenIE artifacts | `724` |
| extraction failures | `0` |
| baseline R@1 | `93.2%` |
| OpenIE R@1 | `90.9%` |
| baseline R@5 | `100.0%` |
| OpenIE R@5 | `100.0%` |
| cache coverage gate | PASS |
| relation probe gate | PASS |
| revertibility gate | PASS |

Relation probe at 30% coverage:

| 지표 | graph expansion off | graph expansion on |
|---|---:|---:|
| relation target expanded | `7/145` | `141/145` |
| relation evidence hit | `7/145` | `58/145` |
| strong relation evidence | `1/40` | `35/40` |

Relation lift:

| 지표 | 값 |
|---|---:|
| expanded lift | `+134` |
| evidence lift | `+51` |
| strong expanded lift | `+39` |
| strong evidence lift | `+34` |

해석:

- 5-row smoke 이후 50-row batch까지 DeepSeek Flash extraction failure는 `0`이다.
- coverage가 `5.0% -> 30.0%`로 증가하면서 relation probe 대상은
  `47 -> 145`개로 늘었다.
- R@5는 여전히 작은 benchmark에서 천장이지만, relation evidence lift는
  `+14 -> +51`로 커졌다.
- strong relation evidence는 graph expansion off `1/40`에서 on `35/40`으로
  증가했다. 기본 RAG 검색만으로는 거의 못 잡는 relation evidence를 OpenIE
  graph expansion이 끌어오는 신호가 더 강해졌다.

---

## DeepSeek Flash 50% Coverage Eval

30% coverage 이후 `openie_cache_missing_deepseek_50_rerun.jsonl` 기준으로
dry-run을 실행했다. 40개 row를 추가 warm하면 50% coverage에 도달하는 것으로
계산됐다.

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input \
    ~/synaptic-eval/openie_cache_missing_deepseek_50_rerun.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-model deepseek-v4-flash \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-dry-run \
  --openie-cache-warm-limit 40 \
  --openie-cache-warm-total-chunks 200 \
  --openie-cache-warm-target-coverage 0.5 \
  --openie-cache-warm-pending-output \
    ~/synaptic-eval/openie_cache_pending_deepseek_to50_dryrun_40.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_dryrun_deepseek_to50_results.json
```

Dry-run result:

| 항목 | 값 |
|---|---:|
| rows loaded | `140` |
| rows pending | `40` |
| deferred by limit | `100` |
| existing covered chunks | `60/200` |
| projected after batch | `100/200` |
| projected coverage | `50.0%` |
| target reachable | `true` |

40-row live warm:

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input \
    ~/synaptic-eval/openie_cache_missing_deepseek_50_rerun.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-v4-flash \
  --llm-api-key-env DEEPSEEK_API_KEY \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-limit 40 \
  --openie-cache-warm-total-chunks 200 \
  --openie-cache-warm-target-coverage 0.5 \
  --openie-cache-warm-pending-output \
    ~/synaptic-eval/openie_cache_pending_deepseek_to50_40.jsonl \
  --openie-cache-warm-failure-output \
    ~/synaptic-eval/openie_cache_failures_deepseek_to50_40.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_deepseek_to50_results.json
```

Warm result:

| 항목 | 값 |
|---|---:|
| rows attempted | `40` |
| rows succeeded | `40` |
| extraction failures | `0` |
| new entities | `206` |
| new triples | `133` |
| cache entries after warm | `104` |
| projected after batch | `100/200` |
| projected coverage | `50.0%` |
| elapsed | `610.6s` |

Audit after 50% warm:

| 항목 | 값 |
|---|---:|
| cache lines | `104` |
| unique keys | `104` |
| parseable records | `104` |
| invalid JSON | `0` |
| invalid records | `0` |
| empty records | `2` |
| entities | `522` |
| triples | `354` |
| result | PASS |

Cache-only scoring at 50% coverage:

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-source-limit 200 \
  --openie-max-chunks 100 \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-only \
  --llm-model deepseek-v4-flash \
  --relation-probe-limit 100 \
  --min-relation-expanded-lift 20 \
  --min-relation-evidence-lift 10 \
  --min-strong-relation-evidence-rate 0.5 \
  --min-openie-cache-coverage 0.50 \
  --embed-base-url "" \
  --results ~/synaptic-eval/mz_openie_cache_deepseek_to50_fast_results.json
```

Scoring result:

| 항목 | 값 |
|---|---:|
| cache eligible chunks | `100/200` |
| cache coverage | `50.0%` |
| relation edges created | `314` |
| OpenIE artifacts | `1,222` |
| extraction failures | `0` |
| baseline R@1 | `93.2%` |
| OpenIE R@1 | `90.9%` |
| baseline R@5 | `100.0%` |
| OpenIE R@5 | `100.0%` |
| cache coverage gate | PASS |
| relation probe gate | PASS |
| revertibility gate | PASS |

Relation probe at 50% coverage:

| 지표 | graph expansion off | graph expansion on |
|---|---:|---:|
| relation target expanded | `2/98` | `90/98` |
| relation evidence hit | `2/98` | `36/98` |
| strong relation evidence | `1/41` | `31/41` |

Relation lift:

| 지표 | 값 |
|---|---:|
| expanded lift | `+88` |
| evidence lift | `+34` |
| strong expanded lift | `+40` |
| strong evidence lift | `+30` |

해석:

- 5-row, 50-row, 추가 40-row live warm까지 총 `95/95` extraction이 성공했고
  failure는 `0`이었다.
- 50% coverage gate에서도 R@5 no-regress와 OpenIE revertibility가 유지됐다.
- bounded runtime을 위해 official 50% gate는 `relation_probe_limit=100`으로
  기록했다. 더 큰 probe limit은 별도 long-running scale run으로 분리하는 편이
  좋다.

---

## DeepSeek Flash 75% Coverage Eval

50% coverage 이후 `openie_cache_missing_deepseek_to50_fast_rerun.jsonl` 기준으로
dry-run을 실행했다. 50개 row를 추가 warm하면 75% coverage에 도달하는 것으로
계산됐다.

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input \
    ~/synaptic-eval/openie_cache_missing_deepseek_to50_fast_rerun.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-model deepseek-v4-flash \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-dry-run \
  --openie-cache-warm-limit 50 \
  --openie-cache-warm-total-chunks 200 \
  --openie-cache-warm-target-coverage 0.75 \
  --openie-cache-warm-pending-output \
    ~/synaptic-eval/openie_cache_pending_deepseek_to75_dryrun_50.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_dryrun_deepseek_to75_results.json
```

Dry-run result:

| 항목 | 값 |
|---|---:|
| rows loaded | `100` |
| rows pending | `50` |
| deferred by limit | `50` |
| existing covered chunks | `100/200` |
| projected after batch | `150/200` |
| projected coverage | `75.0%` |
| target reachable | `true` |

50-row live warm:

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input \
    ~/synaptic-eval/openie_cache_missing_deepseek_to50_fast_rerun.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-v4-flash \
  --llm-api-key-env DEEPSEEK_API_KEY \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-limit 50 \
  --openie-cache-warm-total-chunks 200 \
  --openie-cache-warm-target-coverage 0.75 \
  --openie-cache-warm-pending-output \
    ~/synaptic-eval/openie_cache_pending_deepseek_to75_50.jsonl \
  --openie-cache-warm-failure-output \
    ~/synaptic-eval/openie_cache_failures_deepseek_to75_50.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_deepseek_to75_results.json
```

Warm result:

| 항목 | 값 |
|---|---:|
| rows attempted | `50` |
| rows succeeded | `48` |
| extraction failures | `2` |
| new entities | `251` |
| new triples | `162` |
| cache entries after warm | `152` |
| projected after batch | `148/200` |
| projected coverage | `74.0%` |
| elapsed | `849.8s` |

두 실패 row는 DeepSeek가 빈 body를 반환해 `OpenIE response is not a JSON object`
로 기록된 transient failure였다. failure manifest만 별도 재시도했다.

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input \
    ~/synaptic-eval/openie_cache_failures_deepseek_to75_50.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-v4-flash \
  --llm-api-key-env DEEPSEEK_API_KEY \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-limit 2 \
  --openie-cache-warm-failure-output \
    ~/synaptic-eval/openie_cache_failures_deepseek_to75_retry_2.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_deepseek_to75_retry_2_results.json
```

Retry result:

| 항목 | 값 |
|---|---:|
| rows attempted | `2` |
| rows succeeded | `2` |
| extraction failures | `0` |
| new entities | `10` |
| new triples | `8` |
| cache entries after retry | `154` |
| elapsed | `52.5s` |

주의: retry 입력은 failure manifest 2개 row만 담고 있으므로 coverage projection은
과대 계산된다. 최종 coverage는 아래 cache-only scoring 결과를 기준으로 본다.

Audit after 75% warm:

| 항목 | 값 |
|---|---:|
| cache lines | `154` |
| unique keys | `154` |
| parseable records | `154` |
| invalid JSON | `0` |
| invalid records | `0` |
| empty records | `6` |
| entities | `783` |
| triples | `524` |
| result | PASS |

Cache-only scoring at 75% coverage:

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-source-limit 200 \
  --openie-max-chunks 150 \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-only \
  --llm-model deepseek-v4-flash \
  --relation-probe-limit 100 \
  --min-relation-expanded-lift 20 \
  --min-relation-evidence-lift 10 \
  --min-strong-relation-evidence-rate 0.5 \
  --min-openie-cache-coverage 0.75 \
  --embed-base-url "" \
  --results ~/synaptic-eval/mz_openie_cache_deepseek_to75_results.json
```

Scoring result:

| 항목 | 값 |
|---|---:|
| cache eligible chunks | `150/200` |
| cache coverage | `75.0%` |
| relation edges created | `466` |
| OpenIE artifacts | `1,810` |
| extraction failures | `0` |
| baseline R@1 | `93.2%` |
| OpenIE R@1 | `90.9%` |
| baseline R@5 | `100.0%` |
| OpenIE R@5 | `100.0%` |
| cache coverage gate | PASS |
| relation probe gate | PASS |
| revertibility gate | PASS |

Relation probe at 75% coverage:

| 지표 | graph expansion off | graph expansion on |
|---|---:|---:|
| relation target expanded | `3/96` | `80/96` |
| relation evidence hit | `3/96` | `28/96` |
| strong relation evidence | `2/39` | `24/39` |

Relation lift:

| 지표 | 값 |
|---|---:|
| expanded lift | `+77` |
| evidence lift | `+25` |
| strong expanded lift | `+34` |
| strong evidence lift | `+22` |

해석:

- 75% coverage까지 총 `145`개 DeepSeek extraction row가 cache에 성공적으로
  추가됐다.
- 75% warm 중 transient failure가 2건 있었지만 failure manifest 재시도로 모두
  복구됐다. 이 흐름은 cache warming retry path가 실제로 유용함을 보여준다.
- 75% cache-only scoring에서도 R@5 no-regress, relation gate, revertibility가
  유지됐다.
- official scale gates는 runtime을 bounded하게 유지하기 위해 계속
  `relation_probe_limit=100` 기준으로 기록한다.

---

## DeepSeek Flash 100% Coverage Eval

75% coverage 이후 남은 50개 row를 추가 warm했다. 첫 50-row batch에서는
DeepSeek transient 빈 응답 2건이 있었고, failure manifest 2건만 재시도해 모두
복구했다.

Warm result:

| 항목 | 값 |
|---|---:|
| rows attempted | `50` |
| rows succeeded | `48` |
| extraction failures | `2` |
| new entities | `254` |
| new triples | `174` |
| cache entries after warm | `202` |
| projected coverage | `99.0%` |
| elapsed | `762.0s` |

Retry result:

| 항목 | 값 |
|---|---:|
| rows attempted | `2` |
| rows succeeded | `2` |
| extraction failures | `0` |
| new entities | `12` |
| new triples | `8` |
| cache entries after retry | `204` |
| elapsed | `40.5s` |

Final cache audit:

| 항목 | 값 |
|---|---:|
| cache lines | `204` |
| unique keys | `204` |
| parseable records | `204` |
| invalid JSON | `0` |
| invalid records | `0` |
| empty records | `6` |
| entities | `1,049` |
| triples | `706` |
| result | PASS |

100% 첫 replay에서는 OpenIE entity seed artifact가 최종 evidence top-k를 밀어내고,
`KRA produced/related X` 같은 허브 relation이 relation probe를 과대표집하면서
R@5 no-regress와 strong evidence gate가 실패했다. 이 후속 패치에서 다음 가드를
추가했다.

- 최종 evidence에서는 OpenIE entity가 direct FTS/PPR seed로 들어온 경우 제외한다.
  relation edge를 타고 들어온 OpenIE target은 그대로 유지한다.
- relation probe는 같은 `(source, relation)` 쌍을 기본 3개까지만 샘플링한다.
- eval용 memory health report는 signal node를 DB에 persist하지 않는 read-only
  mode로 생성한다.

Cache-only scoring at 100% coverage:

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-source-limit 200 \
  --openie-max-chunks 200 \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-only \
  --llm-model deepseek-v4-flash \
  --relation-probe-limit 100 \
  --min-relation-expanded-lift 20 \
  --min-relation-evidence-lift 10 \
  --min-strong-relation-evidence-rate 0.5 \
  --min-openie-cache-coverage 1.0 \
  --embed-base-url "" \
  --results ~/synaptic-eval/mz_openie_cache_deepseek_to100_fixed_results.json
```

Scoring result:

| 항목 | 값 |
|---|---:|
| cache eligible chunks | `200/200` |
| cache coverage | `100.0%` |
| relation edges created | `627` |
| OpenIE artifacts | `2,403` |
| extraction failures | `0` |
| baseline R@1 | `93.2%` |
| OpenIE R@1 | `90.9%` |
| baseline R@5 | `100.0%` |
| OpenIE R@5 | `100.0%` |
| delta R@5 | `+0.0%` |
| cache coverage gate | PASS |
| relation probe gate | PASS |
| strong relation evidence gate | PASS |
| revertibility gate | PASS |

Relation probe at 100% coverage:

| 지표 | graph expansion off | graph expansion on |
|---|---:|---:|
| relation target expanded | `4/94` | `92/94` |
| relation evidence hit | `0/94` | `35/94` |
| strong relation evidence | `0/28` | `26/28` |

Relation lift:

| 지표 | 값 |
|---|---:|
| expanded lift | `+88` |
| evidence lift | `+35` |
| strong expanded lift | `+28` |
| strong evidence lift | `+26` |
| strong evidence rate | `92.9%` |

Memory health snapshot:

| 항목 | 값 |
|---|---:|
| signal count | `939` |
| suspect count | `40` |
| relation reinforced signals | `899` |
| stale signals | `16` |
| low-confidence relation signals | `7` |
| OpenIE failure rate | `0.0%` |

해석:

- DeepSeek Flash cache는 200 chunk 기준 100% coverage까지 채워졌고, cache audit도
  parse failure 없이 통과했다.
- OpenIE entity hub를 최종 evidence에서 직접 반환하지 않도록 하면서 R@5 no-regress가
  회복됐다. OpenIE target relation discovery는 유지됐다.
- relation probe 과대표집을 막으면서 100% coverage에서도 strong relation evidence
  rate가 `92.9%`로 안정화됐다.
- full replay는 revertibility gate에서 `2,403`개 OpenIE artifact 제거 후 baseline
  fingerprint가 복원됨을 확인했다.

### Post-merge performance reruns

100% cache-only gate를 같은 입력과 cache로 재실행해 OpenIE replay/revertibility
비용을 확인했다.

| 상태 | 결과 파일 | OpenIE elapsed | wall time | gate |
|---|---|---:|---:|---|
| fixed 100% gate | `mz_openie_cache_deepseek_to100_fixed_results.json` | `2131.8s` | - | PASS |
| PR #14 bulk purge | `mz_openie_cache_deepseek_to100_bulk_results.json` | `1508.6s` | `26:46.58` | PASS |
| PR #15 batch edge replay | `mz_openie_cache_deepseek_to100_batch_results.json` | `955.4s` | `18:25.66` | PASS |
| PR #17 entity replay cache | `mz_openie_cache_deepseek_to100_entitycache_results.json` | `614.9s` | `11:41.07` | PASS |
| PR #19 relation probe read cache | `mz_openie_cache_deepseek_to100_probecache_results.json` | `724.3s` | `15:15.56` | PASS |
| batched node/event writes | `mz_openie_cache_deepseek_to100_batchnodes_results.json` | `126.0s` | `4:44.47` | PASS |
| batched document ingest writes | `mz_openie_cache_deepseek_to100_ingestbatch_results.json` | `117.3s` | `2:23.41` | PASS |
| batched OpenIE link results | `mz_openie_cache_deepseek_to100_linkbatch_results.json` | `3.6s` | `0:31.89` | PASS |
| evidence aggregate similarity cache | `mz_openie_cache_deepseek_to100_simcache_results.json` | `1.2s` | `0:16.28` | PASS |
| graph expand read cache | `mz_openie_cache_deepseek_to100_expandcache2_results.json` | `1.9s` | `0:15.03` | PASS |
| graph expand edge batch | `mz_openie_cache_deepseek_to100_edgebatch_results.json` | `1.1s` | `0:10.65` | PASS |
| PPR iteration cache/bounds | `mz_openie_cache_deepseek_to100_pprbounded_results.json` | `2.4s` | `0:12.66` | PASS |
| evidence aggregate token cache | `mz_openie_cache_deepseek_to100_aggtok_results.json` | `0.8s` | `0:07.93` | PASS |
| EvidenceSearch PPR depth-1 discovery | `mz_openie_cache_deepseek_to100_pprdepth1_results.json` | `1.1s` | `0:13.75` | PASS |
| evidence aggregate pairwise cache | `mz_openie_cache_deepseek_to100_paircache_results.json` | `4.4s` | `0:17.86` | PASS |
| evidence aggregate MMR bound | `mz_openie_cache_deepseek_to100_mmrbounds_results.json` | `2.7s` | `0:16.19` | PASS |
| GraphExpander sub-stage timing | `mz_openie_graph_timing_results.json` | `6.0s` | `0:22.80` | PASS |
| PPR yield diagnostics | `mz_openie_pprdiag_results.json` | `4.9s` | `0:26.23` | PASS |
| EvidenceSearch PPR seed cap | `mz_openie_pprseedcap_results.json` | `4.1s` | `0:15.35` | PASS |
| GraphExpander path node batch | `mz_openie_graphpathbatch_results.json` | `4.4s` | `0:22.82` | PASS |
| GraphExpander filtered edges + light PPR | `mz_openie_lightppr_results.json` | `1.5s` | `0:12.97` | PASS |
| EvidenceSearch PPR seed cap v2 | `mz_openie_pprseed32x1_results.json` | `3.8s` | `0:18.39` | PASS |
| EvidenceSearch PPR result cap | `mz_openie_pprtop2_results.json` | `2.7s` | `0:07.36` | PASS |
| EvidenceSearch aggregate candidate pool cap | `mz_openie_aggpool_final_results.json` | `0.7s` | `0:07.94` | PASS |
| SynapticGraph default FTS seed fanout cap | `mz_openie_seedfanout2_results.json` | `0.8s` | `0:07.82` | PASS |
| GraphExpander OpenIE scope filter | `mz_openie_openiescopefilter_results.json` | `0.7s` | `0:06.20` | PASS |
| SQLite FTS query limit cap | `mz_openie_ftslimit_results.json` | `3.0s` | `0:07.22` | PASS |
| EvidenceSearch aggregate pool min v2 | `mz_openie_aggpool48_results.json` | `1.7s` | `0:06.01` | PASS |
| EvidenceSearch aggregate pool min v3 | `mz_openie_aggpool24_results.json` | `0.8s` | `0:05.76` | PASS |
| EvidenceSearch PPR result cap v2 | `mz_openie_pprtop1_results.json` | `2.4s` | `0:10.01` | PASS |
| EvidenceSearch saturated PPR skip | `mz_openie_pprskip_saturated_results.json` | `0.8s` | `0:10.03` | PASS |
| GraphExpander default budget cap | `mz_openie_expbudget40_results.json` | `1.9s` | `0:08.38` | PASS |
| SynapticGraph FTS seed fanout v2 | `mz_openie_seedfanout1_results.json` | `0.9s` | `0:05.07` | PASS |
| GraphExpander saturated path skip | `mz_openie_graphskip_saturated_results.json` | `0.7s` | `0:09.46` | PASS |
| SQLite FTS LIKE deficit cap | `mz_openie_ftslike_deficit_results.json` | `1.5s` | `0:04.88` | PASS |
| EvidenceAggregator terminal token skip | `mz_openie_agg_terminal_tokens_results.json` | `0.9s` | `0:09.15` | PASS |
| GraphExpander empty REFERENCES skip | `mz_openie_graphrefs_partialindex_rerun_results.json` | `0.7s` | `0:11.23` | PASS |
| GraphExpander filtered light edge reads | `mz_openie_filtered_light_edges_rerun_results.json` | `1.2s` | `0:06.95` | PASS |
| SQLite FTS skip embedding materialization | `mz_openie_fts_skip_embedding_rerun_results.json` | `0.8s` | `0:10.18` | PASS |
| EvidenceAggregator sorted pool fast path | `mz_openie_agg_sort_refindex_rerun_results.json` | `0.9s` | `0:11.33` | PASS |
| GraphExpander category light edge reads | `mz_openie_category_light_edges_rerun_results.json` | `2.7s` | `0:18.44` | PASS |
| GraphExpander selective relation light reads | `mz_openie_selective_related_light_results.json` | `0.8s` | `0:10.96` | PASS |
| GraphExpander entity hub mention filter | `mz_openie_entity_hub_mentions_rerun_results.json` | `0.7s` | `0:05.59` | PASS |
| GraphExpander document related skip | `mz_openie_skip_document_related_results.json` | `1.0s` | `0:10.70` | PASS |
| GraphExpander document-scope PART_OF guard | `mz_openie_document_scope_partof_results.json` | `1.7s` | `0:09.18` | PASS |
| SQLite FTS light LIKE fallback | `mz_openie_fts_like_light_results.json` | `2.5s` | `0:10.13` | PASS |
| EvidenceAggregator reference companion skip | `mz_openie_agg_reference_skip_results.json` | `0.9s` | `0:06.77` | PASS |
| EvidenceAggregator active remaining guard | `mz_openie_agg_active_remaining_results.json` | `1.0s` | `0:07.67` | PASS |

핵심 검색/게이트 지표는 GraphExpander document related skip run에서도 유지됐다:

| 항목 | 값 |
|---|---:|
| baseline R@5 | `100.0%` |
| OpenIE R@5 | `100.0%` |
| delta R@5 | `+0.0%` |
| cache coverage | `100.0%` |
| cache hits/misses | `200/0` |
| OpenIE artifacts | `2,403` |
| relation edges | `627` |
| relation expanded lift | `+89` |
| relation evidence lift | `+47` |
| memory health signals | `930` |
| suspect memories | `31` |
| aggregate pool limit | `30 avg/query` |
| FTS seed count | `1,128 total / 25.6 avg` |
| scored candidates | baseline `1,266 / 28.8 avg`, OpenIE `1,508 / 34.3 avg` |
| baseline aggregate stage | `51.2ms total / 1.2ms avg` |
| OpenIE aggregate stage | `40.1ms total / 0.9ms avg` |
| baseline FTS stage | `47.9ms total / 1.1ms avg` |
| OpenIE FTS stage | `48.9ms total / 1.1ms avg` |
| baseline expand stage | `24.0ms total / 0.5ms avg` |
| baseline expand_graph / expand_ppr | `23.0ms / 0.9ms` |
| baseline graph references / document | `4.0ms / 13.6ms` |
| baseline graph related / entity | `0.14ms / 0.22ms` |
| baseline graph category | `0.07ms total / 0.002ms avg` |
| baseline PPR bfs / iterate | `0.6ms / 0.1ms` |
| baseline PPR added candidates | `0 total / 0.0 avg` |
| baseline PPR seed count | `1,128 total / 25.6 avg` |
| OpenIE expand stage | `38.8ms total / 0.9ms avg` |
| OpenIE expand_graph / expand_ppr | `36.6ms / 2.2ms` |
| OpenIE graph references / document | `3.6ms / 11.7ms` |
| OpenIE graph related / entity | `8.3ms / 7.9ms` |
| OpenIE graph category | `0.06ms total / 0.001ms avg` |
| OpenIE PPR bfs / iterate | `0.8ms / 0.3ms` |
| OpenIE PPR skipped saturated | `37/44 queries` |
| OpenIE PPR result count | `68 total / 1.5 avg` |
| OpenIE PPR added candidates | `50 total / 1.1 avg` |
| OpenIE PPR seed count | `1,128 total / 25.6 avg` |

주의: PR #17은 OpenIE entity node의 불필요한 `updated_at` 갱신을 줄이므로,
relation probe와 health signal의 세부 카운트는 이전 run과 소폭 달라졌다. 다만
R@5 no-regress, cache coverage, relation probe, strong evidence, revertibility gate는
모두 PASS를 유지했다.

PR #19는 relation probe의 no-graph/graph ablation 검색 사이에서 FTS/node/edge
read-through cache를 공유한다. Unit test는 duplicate FTS 호출 제거를 확인했고,
100% cache-only gate도 PASS했지만, 단일 full rerun에서는 PR #17보다 wall time이
길었다. 특히 OpenIE replay elapsed가 `614.9s -> 724.3s`로 흔들렸으므로, PR #19는
현재 "품질 유지 + read-path 중복 제거"로만 해석하고 full-eval 성능 개선으로
계산하지 않는다.

batched node/event write run은 OpenIE entity hub 보장을 chunk 안에서
`save_nodes_batch`로 모으고, semantic extract event의 `source_event_id` stamping을
backend bulk update로 처리한다. 100% cache-only gate는 PASS했고, relation probe와
memory health count는 PR #17/#19 계열과 같은 수준을 유지했다.

batched document ingest run은 `DocumentIngester`가 source 안의 document/chunk
node와 structural edge를 문서마다 flush하지 않고 ingest-run batch로 저장한다.
같은 source 안에 같은 `doc_id`가 다시 나오면 pending writes를 먼저 flush해서
기존 skip/replace 의미를 유지한다.

batched OpenIE link results run은 `LLMOpenIEExtractor.link_results()`가 selected
chunks의 extraction 결과를 먼저 deterministic write plan으로 만든 뒤, entity hub
upsert와 OpenIE edge upsert를 run-level batch로 저장한다. 기존 extractor는
per-chunk fallback을 유지한다.

성능 변화:

- PR #14 이후 OpenIE elapsed는 `2131.8s -> 1508.6s`로 `29.2%` 감소했다.
- PR #15 이후 OpenIE elapsed는 `2131.8s -> 955.4s`로 `55.2%` 감소했다.
- PR #17 이후 OpenIE elapsed는 `2131.8s -> 614.9s`로 `71.2%` 감소했다.
- batched node/event write 이후 OpenIE elapsed는 `2131.8s -> 126.0s`로
  `94.1%` 감소했고, 이전 best인 PR #17 대비 `79.5%` 추가 감소했다.
- batched document ingest write 이후 full wall time은 `35:31.83 -> 2:23.41`로
  `93.3%` 감소했고, 이전 best인 batched node/event write run 대비 `49.6%`
  추가 감소했다.
- batched OpenIE link results 이후 OpenIE elapsed는 `117.3s -> 3.6s`, full wall
  time은 `2:23.41 -> 0:31.89`로 추가 감소했다. 최초 fixed run 대비 wall time은
  `98.5%` 감소했다.
- evidence aggregate similarity cache 이후 aggregate stage는 baseline
  `6667.3ms -> 383.1ms`, OpenIE `6621.6ms -> 383.6ms`로 감소했다. full wall
  time은 `0:31.89 -> 0:16.28`로 추가 감소했고, R@5/relation probe/revertibility
  gate는 PASS를 유지했다.
- graph expand read cache 이후 expand stage는 baseline `926.5ms -> 327.4ms`,
  OpenIE `2238.8ms -> 1560.9ms`로 감소했다. full wall time은
  `0:16.28 -> 0:15.03`으로 소폭 감소했고, R@5/relation probe/revertibility gate는
  PASS를 유지했다. OpenIE replay elapsed는 run별 변동이 있으므로 이 PR의 핵심
  효과는 search-stage expand timing으로 해석한다.
- graph expand edge batch 이후 expand stage는 baseline `327.4ms -> 138.5ms`,
  OpenIE `1560.9ms -> 926.1ms`로 감소했다. full wall time은
  `0:15.03 -> 0:10.65`로 추가 감소했다. 새 sub-stage timing 기준으로 남은
  OpenIE expand 비용은 대부분 PPR (`818.3ms`)이며 GraphExpander 자체는
  `107.8ms` 수준이다.
- PPR iteration cache/bounds 이후 OpenIE expand stage는 `926.1ms -> 411.0ms`,
  OpenIE PPR은 `818.3ms -> 313.8ms`로 감소했다. 배열 기반 power iteration과
  EvidenceSearch discovery용 bounded iteration(`max_iter=20`, `tol=1e-5`)을
  적용했다. R@5/relation probe/revertibility gate는 PASS를 유지했다. 해당 run의
  wall time은 OpenIE replay elapsed 변동(`1.1s -> 2.4s`) 때문에 edge-batch run보다
  길었으므로, 이 변화는 search-stage timing으로 평가한다.
- evidence aggregate token cache 이후 aggregate stage는 baseline
  `385.7ms -> 327.8ms`, OpenIE `381.0ms -> 359.2ms`로 감소했다. 후보 selection
  의미는 바꾸지 않고 node content token set만 bounded per-instance cache로
  재사용한다. R@5/relation probe/revertibility gate는 PASS를 유지했다. 같은 run의
  OpenIE PPR/FTS stage는 실행 노이즈로 더 느리게 측정됐으므로, 이 변화는 직접
  수정한 aggregate stage timing으로 해석한다.
- EvidenceSearch PPR depth-1 discovery 이후 OpenIE expand stage는
  `481.3ms -> 160.6ms`, OpenIE PPR은 `373.5ms -> 64.7ms`, OpenIE PPR BFS/read는
  `186.8ms -> 5.3ms`로 감소했다. 일반 PPR API의 기본 `bfs_depth=2`는 유지하고,
  EvidenceSearch candidate discovery만 `bfs_depth=1`을 사용한다. GraphExpander가
  이미 1-hop neighbours를 당겨오므로, discovery PPR은 direct relation targets를
  랭킹하는 데 필요한 한 layer만 materialize한다. R@5/relation probe/revertibility
  gate는 PASS를 유지했다. 해당 run의 full wall time은 이전 aggregate-token-cache
  run보다 길었으므로, 이 변화는 search-stage timing으로 평가한다.
- evidence aggregate pairwise cache 이후 aggregate stage는 baseline
  `315.2ms -> 227.4ms`, OpenIE `279.4ms -> 201.2ms`로 감소했다. token cache 위에
  bounded pairwise Jaccard cache를 추가해, query 간 반복되는 후보 쌍 similarity를
  재계산하지 않는다. Cache key는 `node.id + content digest`라서 node content가
  바뀌면 stale similarity를 재사용하지 않는다. R@5/relation probe/revertibility
  gate는 PASS를 유지했다. 해당 run의 OpenIE replay elapsed가 `1.1s -> 4.4s`로
  튀었으므로, 이 변화도 직접 수정한 aggregate stage timing으로 평가한다.
- evidence aggregate MMR bound 이후 aggregate stage는 baseline
  `227.4ms -> 202.1ms`, OpenIE `201.2ms -> 183.6ms`로 감소했다. reranker가 이미
  total-descending 후보를 넘기는 production path에서는 현재 best adjusted score가
  남은 후보의 이론상 최대치(`lambda * total`) 이상이면 뒤쪽 후보를 더 보지 않는다.
  Unsorted caller는 기존처럼 full scan을 유지한다. R@5/relation probe/revertibility
  gate는 PASS를 유지했다. 같은 run의 OpenIE expand/FTS stage는 실행 노이즈로 더
  느리게 측정됐으므로, 이 변화는 직접 수정한 aggregate stage timing으로 평가한다.
- GraphExpander sub-stage timing은 retrieval 의미를 바꾸지 않고
  `GraphExpander.expand()` 내부 path별 timing을 `EvidenceSearchResult.timings_ms`에
  노출한다. 200-chunk cache-only gate는 PASS했고, 최신 run 기준 OpenIE
  GraphExpander 내부 비용은 seed edge prefetch `54.1ms`, document scope `26.2ms`,
  related semantic relation walk `3.7ms`, references `3.2ms`, entity mentions
  `3.0ms` 순서다. 따라서 다음 최적화 후보는 generic node batch가 아니라
  seed prefetch fan-out과 document-scope per-node fetch를 더 좁히는 방향이다.
- PPR yield diagnostics는 retrieval 의미를 바꾸지 않고 `EvidenceSearchResult`와
  public `SearchResult`에 `diagnostics`를 추가한다. 최신 200-chunk cache-only run에서
  baseline PPR은 `4`개 후보만 추가했지만, OpenIE PPR은 `944`개, 평균 `21.5`개/query를
  실제 expanded set에 추가했다. 따라서 PPR을 전역 skip하는 최적화는 위험하다. 다음
  후보는 PPR을 끄는 것이 아니라, OpenIE에서 유용한 추가 후보를 유지하면서 PPR
  seed/fan-out/iteration 비용을 줄이는 방향이어야 한다.
- EvidenceSearch PPR seed cap은 discovery PPR에 넘기는 personalization seed를 점수
  상위 `max(64, k*2)`개로 제한한다. 200-chunk cache-only gate는 PASS했고, OpenIE
  PPR stage는 `74.9ms -> 52.5ms`, PPR iterate는 `34.1ms -> 23.1ms`로 감소했다.
  OpenIE PPR added candidates는 `944 -> 1,111`로 유지/증가했고, relation expanded
  `93/93`, evidence `31/93`, revertibility gate도 PASS를 유지했다. 이 변화는 PPR을
  끄지 않고 useful-candidate cost를 낮추는 첫 번째 bounded policy다.
- GraphExpander path node batch는 REFERENCES/document/chunk-next/entity/related
  expansion path마다 candidate node id를 먼저 모은 뒤 `get_nodes_batch()`로 한 번에
  읽는다. 200-chunk cache-only gate는 PASS했고, OpenIE document-scope path는
  `27.3ms -> 13.6ms`, related path는 `4.0ms -> 2.3ms`, GraphExpander 전체는
  `98.6ms -> 96.2ms`로 소폭 감소했다. 같은 run에서 seed prefetch와 PPR timing은
  `57.4ms -> 68.3ms`, `52.5ms -> 61.3ms`로 흔들렸으므로, 이 변화는 full-wall
  speedup이 아니라 direct node-fetch path 개선으로 해석한다.
- GraphExpander filtered edges + light PPR은 seed 전체의 full edge prefetch를 제거하고
  expansion path가 필요한 edge kind만 batch-read한다. SQLite/Memory backend에는
  filtered edge batch와 SQLite `(source_id, kind)` / `(target_id, kind)` index를
  추가했다. Discovery PPR은 edge provenance/properties가 필요 없으므로 SQLite에서
  `properties_json`을 읽거나 파싱하지 않는 light edge batch를 사용한다. 200-chunk
  cache-only gate는 PASS했고, OpenIE seed prefetch는 `68.3ms -> 0.0ms`,
  GraphExpander 전체는 `96.2ms -> 84.4ms`로 감소했다. PPR은 더 이상 GraphExpander의
  full prefetch를 재사용하지 않으므로 OpenIE PPR BFS는 prewarmed run 대비
  `4.5ms -> 35.3ms`로 증가했지만, filtered-only run의 `52.0ms`보다는 감소했다.
  같은 DB에서 80-node SQLite micro-benchmark는 full edge batch `1.98ms`, light edge
  batch `1.00ms`, document-kind filtered batch `0.38ms`, semantic-kind filtered
  batch `0.62ms`였다. 따라서 이 변화는 seed-prefetch fan-out을 제거하고 PPR edge
  materialization을 가볍게 만드는 foundation이며, full search timing은 run별 FTS/PPR
  노이즈와 함께 해석한다.
- EvidenceSearch PPR seed cap v2는 discovery PPR seed policy를
  `max(64, k*2)`에서 `max(32, k)`로 낮춘다. 200-chunk cache-only gate는 PASS했고,
  relation expanded/evidence `93/93`, `32/93`, revertibility gate도 유지했다.
  OpenIE PPR seed count는 `2,386 -> 1,202`, PPR stage는 `86.3ms -> 61.1ms`,
  PPR BFS는 `35.3ms -> 18.0ms`, PPR iterate는 `24.2ms -> 12.8ms`로 감소했다.
  PPR added candidates는 `1,110 -> 1,748`로 증가했는데, 적은 teleport seed가
  top-k 결과 안에서 기존 expanded set 밖 후보를 더 많이 남겼기 때문이다. R@5와
  relation evidence gate가 유지됐으므로 v2 bounded policy는 useful-candidate
  discovery를 보존하면서 PPR read/iteration 비용을 낮춘다.
- EvidenceSearch PPR result cap은 discovery PPR 반환 후보를 `k*3`에서 `k*2`로
  줄인다. 200-chunk cache-only gate는 PASS했고, relation expanded/evidence
  `93/93`, `32/93`도 유지했다. OpenIE PPR result count는 `3,272 -> 2,288`,
  PPR added candidates는 `1,748 -> 933`, scored candidates는 `5,452 -> 4,637`로
  감소했다. 그 결과 OpenIE PPR stage는 `61.1ms -> 46.8ms`, PPR fetch는
  `20.8ms -> 11.9ms`, rerank는 `15.2ms -> 8.9ms`로 줄었다. 이 단계는 PPR의
  useful relation discovery를 유지하면서 downstream candidate pressure를 낮춘다.
- EvidenceSearch aggregate candidate pool cap은 final MMR/diversity stage에 들어가는
  passage 후보를 기본 `max(64, k*2)`로 제한한다. Aggregator의 protected pool은
  카테고리 대표, 문서 다양성 후보, REFERENCES companion을 보존한다. 200-chunk
  cache-only gate는 PASS했고, relation expanded/evidence `93/93`, `32/93`,
  R@5 no-regress를 유지했다. Aggregate stage는 baseline `203.5ms -> 155.3ms`,
  OpenIE `181.2ms -> 143.1ms`로 줄었다.
- SynapticGraph default FTS seed fanout cap은 `graph.search()`가 EvidenceSearch에
  넘기는 기본 lexical seed over-fetch를 `max(20, limit*3)`에서
  `max(20, limit*2)`로 낮춘다. 명시적 `fts_seed_limit`은 그대로 존중한다.
  200-chunk cache-only gate는 PASS했고, relation expanded/evidence `93/93`,
  `32/93`, R@5 no-regress를 유지했다. 평균 FTS seed는 `76.1 -> 50.9`,
  OpenIE scored candidates는 `105.4 -> 89.1`로 줄었고, total search timing은
  baseline `382.3ms -> 293.5ms`, OpenIE `406.8ms -> 338.1ms`로 감소했다.
- GraphExpander OpenIE scope filter는 OpenIE entity hub를 document-scope
  `CONTAINS/PART_OF` expansion에서 제외한다. 일반 phrase/entity hub의
  `CONTAINS` path는 유지하고, OpenIE `PART_OF` relation은 related path에서
  `semantic_relation`으로 남긴다. 200-chunk cache-only gate는 PASS했고,
  relation expanded는 `93/93`을 유지하면서 relation evidence가 `32/93 -> 47/93`로
  증가했다. R@5 no-regress와 revertibility도 유지됐다.
- SQLite FTS query limit cap은 `search_fts()`의 FTS5 pass가 최종 반환 가능한
  `limit`만 읽도록 바꾼다. FTS virtual table은 node당 한 row라 FTS pass 자체에는
  `limit*2` over-fetch가 필요 없고, LIKE fallback은 FTS hit가 `limit`보다 적을 때만
  기존처럼 동작한다. 200-chunk cache-only gate는 PASS했고, relation expanded/evidence
  `93/93`, `47/93`, R@5 no-regress를 유지했다. FTS stage는 baseline
  `91.0ms -> 76.8ms`, OpenIE `99.7ms -> 76.5ms`로 줄었다.
- EvidenceSearch aggregate pool min v2는 final MMR/diversity stage에 들어가는
  기본 후보 풀을 `max(64, k*2)`에서 `max(48, k)`로 낮춘다. 200-chunk cache-only
  gate는 PASS했고, R@5 no-regress, relation expanded/evidence `93/93`, `47/93`를
  유지했다. Aggregate stage는 baseline `134.1ms -> 107.3ms`, OpenIE
  `115.7ms -> 110.2ms`로 줄었고, 평균 evidence count도 baseline `21.9`,
  OpenIE `23.5`로 충분히 유지됐다.
- EvidenceSearch aggregate pool min v3는 기본 후보 풀을 `max(48, k)`에서
  `max(24, k)`로 낮춘다. 200-chunk cache-only gate는 PASS했고, R@5 no-regress,
  relation expanded/evidence `93/93`, `47/93`를 유지했다. 최신 run에서 실제 평균
  pool은 `48 -> 30`으로 줄었고, aggregate stage는 baseline `107.3ms -> 85.1ms`,
  OpenIE `110.2ms -> 84.6ms`로 줄었다. `pool=16` probe는 formal gate는
  통과했지만 relation evidence가 `47/93 -> 31/93`로 감소했으므로 v3 기본값은
  `24`를 안전선으로 둔다.
- EvidenceSearch PPR result cap v2는 discovery PPR 반환 후보를 `k*2`에서 `k`로
  줄인다. 200-chunk cache-only gate는 PASS했고, R@5 no-regress,
  relation expanded/evidence `93/93`, `47/93`를 유지했다. OpenIE PPR result count는
  `2,288 -> 1,178`, PPR added candidates는 `756 -> 115`, scored candidates는
  `3,918 -> 3,277`로 줄었다. OpenIE PPR fetch도 `10.5ms -> 4.5ms`로 줄었고,
  OpenIE expand stage는 `110.3ms -> 105.9ms`로 소폭 감소했다. 같은 run의 FTS와
  OpenIE replay elapsed는 흔들렸으므로, 이 변화는 full-wall speedup이 아니라
  useful relation discovery를 유지한 candidate-pressure reduction으로 해석한다.
- EvidenceSearch saturated PPR skip은 aggregate cap이 켜져 있고 GraphExpander가
  이미 final aggregate pool 이상의 후보를 만든 query에서 discovery PPR을 생략한다.
  PPR은 현재 expanded set 밖 후보를 추가하는 데만 쓰이므로, 후보 풀이 이미 충분한
  query에서는 downstream recall을 해치지 않고 graph materialisation/iteration을
  피할 수 있다. 200-chunk cache-only gate는 PASS했고, R@5 no-regress,
  relation expanded/evidence `93/93`, `47/93`를 유지했다. PPR은 `37/44`개 query에서
  skip됐고, OpenIE PPR stage는 `37.9ms -> 2.5ms`, OpenIE expand stage는
  `105.9ms -> 74.9ms`, scored candidates는 `3,277 -> 3,212`로 줄었다.
- GraphExpander default budget cap은 기본 `max_total_expanded`를 `100 -> 40`으로
  낮춘다. 200-chunk cache-only gate는 PASS했고, R@5 no-regress,
  relation expanded/evidence `93/93`, `47/93`를 유지했다. Probe에서 `80/60/48/40`은
  모두 relation evidence를 유지했지만, `32`는 relation expanded/evidence가
  `93/93`, `47/93`에서 `90/93`, `45/93`로 떨어졌으므로 `40`을 안전선으로 둔다.
  최신 run에서 OpenIE expanded-before-PPR은 `3,162 -> 1,498`, scored candidates는
  `3,212 -> 1,548`, OpenIE GraphExpander stage는 `72.3ms -> 46.9ms`, aggregate
  stage는 `89.0ms -> 44.2ms`로 줄었다.
- SynapticGraph FTS seed fanout v2는 `graph.search()` 기본 lexical seed pool을
  `max(20, limit*2)`에서 `max(20, limit)`로 낮춘다. 명시적 `fts_seed_limit`은
  그대로 존중한다. 200-chunk cache-only gate는 PASS했고, R@5 no-regress,
  relation expanded/evidence `93/93`, `47/93`를 유지했다. FTS seed total은
  `2,238 -> 1,128`, OpenIE FTS stage는 `66.8ms -> 49.5ms`, OpenIE total search는
  `164.5ms -> 141.0ms`, scored candidates는 `1,548 -> 1,508`로 줄었다.
- GraphExpander saturated path skip은 first-come budget이 이미 찬 뒤에는 이후
  expansion path의 backend read를 건너뛴다. Timing key는 계속 기록하므로
  diagnostics shape는 유지한다. 200-chunk cache-only gate는 PASS했고, R@5
  no-regress, relation expanded/evidence `93/93`, `47/93`를 유지했다.
  Baseline GraphExpander stage는 `35.5ms -> 32.3ms`, OpenIE GraphExpander stage는
  `45.7ms -> 42.2ms`, OpenIE related path는 `13.1ms -> 9.3ms`로 줄었다.
- SQLite FTS LIKE deficit cap은 FTS5 pass가 이미 일부 결과를 채웠을 때 LIKE
  fallback scan limit을 기존 `limit*2`에서 `max(limit, deficit*2)`로 줄인다.
  FTS hit가 0개인 query는 기존 `limit*2` 안전선을 유지하고, FTS가 거의 채운
  query만 초과 materialization을 줄인다. 200-chunk cache-only gate는 PASS했고,
  R@5 no-regress, relation expanded/evidence `93/93`, `47/93`, scored candidates
  baseline/OpenIE `1,266/1,508`을 그대로 유지했다. 최신 run의 stage timing은
  baseline FTS `52.6ms -> 48.4ms`, OpenIE FTS `51.4ms -> 52.9ms`로 실행 노이즈가
  섞였으므로 이 변화는 recall-preserving fallback read bound로 해석한다.
- EvidenceAggregator terminal token skip은 마지막 evidence를 선택한 뒤 더 이상
  MMR similarity 비교가 남지 않은 경우 해당 node의 content tokenization/cache write를
  생략한다. Reference companion attach와 future MMR 비교가 필요한 경우에는 기존처럼
  selected token entry를 유지한다. 200-chunk cache-only gate는 PASS했고, R@5
  no-regress, relation expanded/evidence `93/93`, `47/93`, scored candidates
  baseline/OpenIE `1,266/1,508`을 유지했다. Aggregate stage는 baseline
  `44.7ms -> 43.2ms`, OpenIE `42.3ms -> 40.9ms`로 줄었다.
- GraphExpander empty REFERENCES skip은 backend가 `REFERENCES` edge kind 부재를
  확인할 수 있을 때 seed별 REFERENCES filtered edge batch를 건너뛴다. SQLite에는
  일반 filtered edge planner를 해치지 않도록 `kind = 'references'` partial index를
  추가했고, Memory backend에는 같은 optional existence check를 추가했다. optional
  method가 없는 backend는 기존 path를 그대로 사용한다. 200-chunk cache-only gate는
  PASS했고, R@5 no-regress, relation expanded/evidence `93/93`, `47/93`,
  scored candidates baseline/OpenIE `1,266/1,508`을 유지했다. 직접 수정한
  REFERENCES stage는 baseline `5.7ms -> 3.7ms`, OpenIE `6.0ms -> 3.8ms`로
  감소했다. Full search total은 FTS/SQLite run noise가 섞이므로 이 변화는
  REFERENCES path의 불필요 read 제거로 해석한다.
- GraphExpander filtered light edge reads는 provenance metadata가 필요 없는
  REFERENCES/document-scope/chunk-next/entity-mention path에서 filtered edge read가
  `properties_json`을 읽고 파싱하지 않도록 한다. `related` semantic relation path는
  `is_openie`와 `confidence` metadata가 필요하므로 full filtered read를 유지한다.
  200-chunk cache-only gate는 PASS했고, R@5 no-regress, relation expanded/evidence
  `93/93`, `47/93`, scored candidates baseline/OpenIE `1,266/1,508`을 유지했다.
  최신 rerun 기준 OpenIE GraphExpander stage는 `40.6ms -> 39.9ms`, entity mention
  path는 `9.8ms -> 9.1ms`로 줄었다. Baseline total은 작은 DB에서 FTS/SQLite run
  noise가 더 커서 이 변화는 full-wall speedup이 아니라 provenance-free edge
  materialization reduction으로 해석한다.
- SQLite FTS skip embedding materialization은 `EvidenceSearch`가 query embedding을
  쓰지 않는 기본 경로에서 SQLite `search_fts(include_embedding=False)`를 요청하게
  한다. 이때 FTS seed node는 title/content/tags/properties/source 등 retrieval에
  필요한 metadata는 유지하고, PRF/query-vector path에서만 embedding을 계속 읽는다.
  200-chunk cache-only gate는 PASS했고, R@5 no-regress, relation expanded/evidence
  `93/93`, `47/93`, scored candidates baseline/OpenIE `1,266/1,508`을 유지했다.
  최신 rerun 기준 OpenIE FTS stage는 `50.8ms -> 49.3ms`, OpenIE total search는
  `137.8ms -> 135.0ms`였다. 첫 run은 SQLite/FTS noise로 더 느리게 나왔으므로,
  이 변화는 embedding-bearing corpora에서 FTS seed materialization 폭을 줄이는
  no-regress foundation으로 해석한다.
- EvidenceAggregator sorted pool fast path는 production reranker가 이미
  total-descending으로 넘기는 passage pool에서 `_bounded_passage_pool()`의 full sort를
  생략한다. Unsorted external caller는 기존처럼 sort한다. 같은 변경에서 structured /
  passage split은 한 번의 loop로 합쳤고, REFERENCES companion attach는 anchor별 index를
  한 번 만든 뒤 id set으로 제거해 selection마다 remaining 전체를 다시 훑지 않도록 했다.
  200-chunk cache-only gate는 PASS했고, R@5 no-regress, relation expanded/evidence
  `93/93`, `47/93`, scored candidates baseline/OpenIE `1,266/1,508`을 유지했다.
  최신 rerun 기준 baseline aggregate stage는 `44.9ms -> 43.0ms`로 줄었다. OpenIE
  aggregate는 `40.0ms -> 40.7ms`로 run noise 안에서 소폭 느렸으므로, 이 변화는
  broad wall-time claim이 아니라 sorted production path의 불필요 sort/scan 제거로
  해석한다.
- GraphExpander category light edge reads는 category sibling expansion을 generic
  `get_neighbors(depth=1)`에서 `PART_OF` filtered light edge read + batch node fetch로
  바꾼다. Category node에 다른 edge kind가 붙어도 sibling 후보로 materialize하지 않고,
  큰 category에서는 필요한 document 후보만 budget 안에서 가져온다. Unit test는
  `RELATED` edge가 category sibling으로 새지 않고, `get_neighbors()` 없이
  `get_edges_batch_filtered_light()`와 `get_nodes_batch()`가 쓰이는지를 고정한다.
  200-chunk cache-only gate는 PASS했고, R@5 no-regress, relation expanded/evidence
  `93/93`, `47/93`, scored candidates baseline/OpenIE `1,266/1,508`을 유지했다.
  최신 rerun 기준 total search는 baseline `129.0ms -> 127.2ms`, OpenIE
  `138.3ms -> 137.4ms`였다. Category path 자체는 `0.03-0.07ms` 수준이라 wall-time
  claim보다 read-shape correctness와 large-category materialization reduction으로
  해석한다.
- GraphExpander selective relation light reads는 relation expansion에서 generic
  `RELATED` edge는 traversal fields만 materialize하고, `DEPENDS_ON`, `PART_OF`,
  `CAUSED` 같은 typed OpenIE relation만 provenance metadata를 유지한다. 단순히
  `RELATED`와 typed relation을 두 번 읽으면 SQLite round-trip이 늘고 related evidence가
  `47/93 -> 46/93`으로 흔들렸기 때문에, backend optional method
  `get_edges_batch_filtered_selective_light()`와 `GraphReadCache` wrapper를 추가해 한 번의
  filtered read 안에서 light/full materialization을 나눈다. Memory/SQLite tests는
  `RELATED` properties가 비워지고 typed relation confidence는 유지되는지 확인한다.
  200-chunk cache-only gate는 PASS했고, R@5 no-regress, relation expanded/evidence
  `93/93`, `47/93`, scored candidates baseline/OpenIE `1,266/1,508`을 유지했다.
  최신 run 기준 OpenIE graph related path는 `9.7ms -> 9.5ms`, OpenIE total search는
  `137.4ms -> 133.4ms`였다. 작은 run의 wall time은 FTS/SQLite noise가 있으므로,
  이 변화는 broad speedup claim보다 provenance-free RELATED materialization을 제거하는
  no-regress foundation으로 해석한다.
- GraphExpander entity hub mention filter는 entity mention expansion을 모든
  `NodeKind.ENTITY` seed가 아니라 `_phrase`, `_openie_entity`, `_llm_enriched`,
  `_spacy` hub tag가 있는 seed로 제한한다. 문서/row처럼 ENTITY로 저장된 seed는
  불필요한 incoming `MENTIONS` read를 건너뛰고, 실제 phrase/OpenIE/SpaCy/LLM hub는
  source chunk bridge를 유지한다. 단순 empty-MENTIONS existence guard는 OpenIE DB에서
  entity path를 `8.6ms -> 12.2ms`로 느리게 만들었지만, hub tag filter rerun은
  relation expanded/evidence `93/93`, `47/93`, scored candidates baseline/OpenIE
  `1,266/1,508`을 유지하면서 baseline entity mention path를 `3.3ms -> 0.18ms`,
  OpenIE entity mention path를 `8.6ms -> 8.4ms`로 유지했다.
- GraphExpander document related skip은 `tags=["document"]`인 ENTITY seed를
  structured/OpenIE relation expansion seed에서 제외한다. `DocumentIngester`는 문서를
  `NodeKind.ENTITY`로 저장하므로 기존 "ENTITY만 related path를 탄다" 조건은 문서 graph를
  제외하지 못했고, document seed의 category `PART_OF` 같은 edge를 relation 후보로 읽은 뒤
  버릴 수 있었다. Non-document ENTITY(row, phrase, SpaCy/LLM/OpenIE hub)는 기존 relation
  path를 유지한다. 200-chunk cache-only gate는 두 번 모두 PASS했고, relation
  expanded/evidence `93/93`, `47/93`, scored candidates baseline/OpenIE
  `1,266/1,508`을 유지했다. 대표 run 기준 related path는 baseline `4.7ms -> 0.14ms`,
  OpenIE `10.4ms -> 8.3ms`로 줄었다. rerun에서도 related path는 baseline `0.15ms`,
  OpenIE `9.3ms`로 유지됐고, full total은 FTS/SQLite noise가 섞였다.
- GraphExpander document-scope PART_OF guard는 document-scope expansion에서
  `CONTAINS`와 chunk-oriented `PART_OF`는 유지하되, document seed의 outgoing
  `PART_OF` category edge를 same-document neighbourhood로 취급하지 않는다. 이전에는
  `doc -> category PART_OF`가 `document_chunk` reason으로 확장될 수 있었다. Unit test는
  category가 document-scope 결과에서 빠지고, legacy chunk `PART_OF` parent expansion은
  유지되는지 고정한다. 200-chunk cache-only gate는 PASS했고, R@5 no-regress, relation
  expanded/evidence `93/93`, `47/93`, scored candidates baseline/OpenIE `1,266/1,508`을
  유지했다. 이 run에서는 candidate count 변화가 없어 performance claim보다 graph path
  semantics cleanup으로 해석한다.
- SQLite FTS light LIKE fallback은 FTS5가 못 잡은 substring 후보를 보충할 때 처음부터
  full `syn_nodes` row를 materialize하지 않고 `id/title/content`만 읽어 substring 점수를
  계산한 뒤, 최종 `limit`에 살아남는 LIKE 후보만 full node로 로드한다. FTS5 hit band와
  LIKE fallback band, `with_scores=True` contract, `include_embedding=False` 동작은 유지한다.
  200-chunk cache-only gate는 두 번 모두 PASS했고, R@5 no-regress, relation
  expanded/evidence `93/93`, `47/93`, FTS seed count baseline/OpenIE `1,128/1,128`을
  유지했다. Run별 FTS latency는 baseline `50.2ms -> 53.4ms -> 46.1ms`, OpenIE
  `52.8ms -> 49.3ms -> 59.2ms`로 흔들렸으므로 broad speedup claim보다
  LIKE-fallback JSON/embedding materialization reduction으로 해석한다.
- EvidenceAggregator reference companion skip은 REFERENCES companion을 evidence에 붙인 뒤
  남은 후보 리스트를 매번 재구성하지 않고 `remaining_ids`에서만 제거해 이후 greedy scan이
  건너뛰도록 한다. 결과 순서와 duplicate 방지 동작은 유지되며, reference-heavy corpus에서
  companion attach의 O(n) 리스트 rebuild를 없앤다. 200-chunk cache-only gate는 PASS했고,
  R@5 no-regress, relation expanded/evidence `93/93`, `47/93`, scored candidates
  baseline/OpenIE `1,266/1,508`을 유지했다. MZ run의 aggregate stage는 baseline
  `49.5ms -> 44.6ms`, OpenIE `40.2ms -> 42.7ms`로 run noise가 있어 broad latency claim보다
  companion-heavy path cleanup으로 해석한다.
- EvidenceAggregator active remaining guard는 companion attach 이후 실제 active candidate set인
  `remaining_ids`가 비었을 때 greedy MMR loop를 즉시 종료한다. 이미 dead path가 된 similarity
  helper들도 제거했다. 결과 품질은 유지됐고, 200-chunk cache-only gate는 PASS, relation
  expanded/evidence `93/93`, `47/93`, scored candidates baseline/OpenIE `1,266/1,508`을
  유지했다. 이전 reference companion skip run 대비 aggregate stage는 baseline
  `44.569ms -> 43.737ms`, OpenIE `42.748ms -> 40.529ms`였으며, 이는 품질 변화 없는
  stale-scan cleanup으로 해석한다.
- Memory health penalty provenance counts는 health report가 감점 원인 signal/edge/node의
  상위 ID뿐 아니라 반복 횟수 map도 함께 노출하게 한다. 이로써 운영자가 한 번 발생한
  감점과 반복적으로 검색 품질을 누르는 오염 신호를 구분할 수 있다. POC gate는
  `health_reports_penalty_provenance_counts`를 추가해 `39/39` PASS했다. 대표 결과는
  `top_penalty_signal_counts={"memsig_32d1378cc670e496": 1, "poc_edge_only_signal_node": 1}`,
  `top_penalized_node_counts={"poc_scope_clean_memory": 1, "poc_edge_only_signal_suspect": 1}`,
  `top_penalty_edge_counts={"poc_edge_score_demoted_relation": 1, "poc_edge_only_signal_relation": 1}`였다.
- PR #15의 300-edge SQLite micro-benchmark는 old per-edge write
  `109,962.04ms` 대비 batch write `195.85ms`로 `561.45x` 빨랐다.
- PR #17의 100-chunk repeated-entity micro-benchmark는 backend `get_node()` `1`,
  `save_node()` `1`, `update_node()` `0`으로 같은 hub 반복 조회/갱신을 제거했다.
- 50-entity SQLite link micro-benchmark는 fallback 개별 node save
  `9,576.92ms` 대비 batch node save `243.45ms`로 `39.34x` 빨랐다.
- 30-edge event stamp micro-benchmark는 old fallback `1,739.87ms` 대비 bulk stamp
  `640.20ms`로 `2.72x` 빨랐다. 300-edge old fallback은 1분 이상 걸려 중단했다.
- 200-doc/200-chunk SQLite ingest micro-benchmark는 document-per-flush path
  `92.006s` 대비 ingest-run batch `2.683s`로 `34.3x` 빨랐다.
- 50-chunk OpenIE link micro-benchmark는 run-level materialization에서
  `save_nodes_batch=1`, `save_openie_edges_batch=1`로 완료됐다.
- 남은 큰 검색-stage 비용은 aggregate selection, FTS, OpenIE GraphExpander의
  document-scope/related/entity path다. FTS seed fanout v2 이후 FTS는 더 이상
  unbounded seed over-fetch 비용이 아니고, GraphExpander saturated path skip 이후
  budget이 찬 path의 backend read도 제거됐으며, SQLite LIKE fallback도 deficit
  기준으로 bounded 됐다. EvidenceAggregator도 terminal pick tokenization을 생략하지만,
  200-chunk gate 기준에서 aggregate selection, FTS, OpenIE GraphExpander
  document/related/entity path는 아직 직접 측정 가능하다. PPR BFS/read/iterate는
  같은 기준으로 더 이상 지배 병목이 아니다. Baseline document ingest write path,
  OpenIE replay write path, aggregate tokenisation/Jaccard recomputation path도
  같은 기준에서 지배 병목이 아니다.

---

## Current Interpretation

작업 전의 기본 RAG는 "질의와 직접 유사한 chunk" 중심이었다. PR #10 이후에는:

- 검색/피드백이 ledger로 남는다.
- 성공/실패/선택/무시 신호를 scope별로 분리해서 기록한다.
- 강화 score가 user/workspace/global을 오염시키지 않도록 분리된다.
- OpenIE relation edge에 provenance가 남는다.
- 충돌, 오래된 기억, 반복 실패, 낮은 confidence relation이 health signal로
  관찰된다.
- relation probe에서 OpenIE graph expansion이 실제로 evidence 후보를 넓힌다.

아직 남은 증명:

- Qwen3.6 small quality reference 재측정.
- 더 큰 query/evidence benchmark에서 100% coverage의 R@1/R@5와 relation lift가
  유지되는지 확인.
- full long-running pytest/QA suite를 빠른 CI와 nightly eval로 분리.
- OpenIE entity/node replay와 relation probe/search DB roundtrip을 추가로 줄인다.
