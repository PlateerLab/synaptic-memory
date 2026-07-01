# Memory Operating Layer Eval Report

작성: 2026-07-01

대상 PR: [#10 Add memory operating layer foundation](https://github.com/PlateerLab/synaptic-memory/pull/10)

---

## 요약

PR #10 이후 `main`에는 memory event ledger, retrieval feedback ledger,
scope-aware reinforcement, edge provenance, pollution/health signals, OpenIE
cache/replay harness가 들어갔다.

이번 후속 평가는 두 가지를 확인했다.

1. **기본 경로는 안정적인가?**
   - lint, targeted unit tests, memory operating PoC, OpenIE skip/cache-only smoke
     모두 통과했다.
2. **기본 RAG를 넘어서는 relation expansion 신호가 있는가?**
   - cache-only OpenIE smoke에서 relation target expansion이 `1/26 -> 26/26`,
     strong relation evidence가 `0/8 -> 8/8`로 증가했다.

다만 live DeepSeek/Qwen extraction은 이 런타임에 API key/endpoint가 없어
실행하지 못했다. 따라서 이 문서는 **foundation merge 직후의 deterministic
검증 + live eval 준비 상태**를 기록한다.

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

API key는 코드, 문서, DB, cache에 저장하지 않는다. live eval을 실행할 때만
프로세스 환경 변수로 주입한다.

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

---

## DeepSeek Flash Live Eval Readiness

현재 live eval은 `DEEPSEEK_API_KEY`가 없어 실행하지 못했다. 대신 같은 입력에서
cache warming dry-run을 실행해 다음 batch 크기와 coverage projection을 계산했다.

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input ~/synaptic-eval/openie_cache_missing_200.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-model deepseek-v4-flash \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-dry-run \
  --openie-cache-warm-limit 50 \
  --openie-cache-warm-total-chunks 200 \
  --openie-cache-warm-target-coverage 0.5 \
  --openie-cache-warm-pending-output \
    ~/synaptic-eval/openie_cache_pending_eval_next_50_target50.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_dry_run_eval_next_results.json
```

Dry-run result:

| 항목 | 값 |
|---|---:|
| missing rows loaded | `195` |
| pending batch rows | `50` |
| deferred by limit | `145` |
| existing covered chunks | `5/200` |
| projected after one 50-row batch | `55/200` |
| projected coverage | `27.5%` |
| target coverage | `50.0%` |
| rows needed for target | `95` |
| batches needed at limit 50 | `2` |
| target reachable | `true` |

즉 DeepSeek Flash로 다음 batch 50개를 warm하면 coverage는 `2.5% -> 27.5%`로
오르고, 50% coverage에는 총 95개 추가 row가 필요하다.

Live run command:

```bash
export DEEPSEEK_API_KEY=...

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
    ~/synaptic-eval/openie_cache_pending_eval_next_50_target50.jsonl \
  --openie-cache-warm-failure-output \
    ~/synaptic-eval/openie_cache_failures_eval_next_50.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_deepseek_50_results.json
```

After warming, rerun cache-only scoring with a higher coverage gate:

```bash
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-source-limit 200 \
  --openie-max-chunks 50 \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-only \
  --llm-model deepseek-v4-flash \
  --relation-probe-limit 100 \
  --min-relation-expanded-lift 10 \
  --min-relation-evidence-lift 5 \
  --min-strong-relation-evidence-rate 0.5 \
  --min-openie-cache-coverage 0.25 \
  --embed-base-url "" \
  --results ~/synaptic-eval/mz_openie_cache_deepseek_50_results.json
```

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

- DeepSeek Flash live extraction 50/100/200 chunk batch.
- Qwen3.6 small quality reference 재측정.
- cache coverage가 올라간 상태에서 R@1/R@5와 relation evidence lift가 유지되는지
  확인.
- full long-running pytest/QA suite를 빠른 CI와 nightly eval로 분리.

