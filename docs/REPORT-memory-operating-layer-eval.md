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

따라서 이 문서는 **foundation merge 직후의 deterministic 검증 + DeepSeek
Flash live warm 결과 + 다음 scale eval 계획**을 기록한다.

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

- DeepSeek Flash live extraction 100/200 chunk batch.
- Qwen3.6 small quality reference 재측정.
- cache coverage가 올라간 상태에서 R@1/R@5와 relation evidence lift가 유지되는지
  확인.
- full long-running pytest/QA suite를 빠른 CI와 nightly eval로 분리.
