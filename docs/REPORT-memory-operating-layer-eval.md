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

핵심 검색/게이트 지표는 최신 GraphExpander filtered light edge reads rerun에서도 유지됐다:

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
| baseline aggregate stage | `45.0ms total / 1.0ms avg` |
| OpenIE aggregate stage | `40.8ms total / 0.9ms avg` |
| baseline FTS stage | `49.8ms total / 1.1ms avg` |
| OpenIE FTS stage | `50.8ms total / 1.2ms avg` |
| baseline expand stage | `31.2ms total / 0.7ms avg` |
| baseline expand_graph / expand_ppr | `30.1ms / 1.0ms` |
| baseline graph references / document | `4.0ms / 13.7ms` |
| baseline PPR bfs / iterate | `0.7ms / 0.1ms` |
| baseline PPR added candidates | `0 total / 0.0 avg` |
| baseline PPR seed count | `1,128 total / 25.6 avg` |
| OpenIE expand stage | `42.2ms total / 1.0ms avg` |
| OpenIE expand_graph / expand_ppr | `39.9ms / 2.3ms` |
| OpenIE graph references / document | `3.8ms / 12.1ms` |
| OpenIE graph related / entity | `9.5ms / 9.1ms` |
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
