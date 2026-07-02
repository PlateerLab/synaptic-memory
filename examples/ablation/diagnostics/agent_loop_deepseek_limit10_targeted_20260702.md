# DeepSeek agent loop targeted limit-10 check

- Run at: 2026-07-02 20:47 KST
- Model: `deepseek-v4-flash`
- Base corpus: `tests/benchmark/data/msmarco_passage_full.json`
- SQLite DB: `tests/benchmark/data/msmarco_full.db`
- Corpus limit: 8,841,823 passages
- Target set: six high-call misses from the prior 50-query DeepSeek run
- Change under test: expose `limit`/`read_top_k` to the agent tool schema and widen the default `deep_search` evidence pool from 5 to 10.

## Result

The targeted high-call miss set improved from `0/6` reached in the prior 50-query run to `2/6` reached after the wider `deep_search` evidence pool.

| QID | Query | Prior 50-query run | Limit-10 targeted run | First relevant |
| --- | --- | ---: | ---: | --- |
| 54544 | blood diseases that are sexually transmitted | no | no | - |
| 293992 | how many product lines does coca cola have | no | yes | turn 2 / call 3 |
| 208145 | how bicycle tire tubes are sized | no | no | - |
| 14151 | age requirements for name change | no | yes | turn 1 / call 1 |
| 91711 | child psychiatrist salary 2016 | no | no | - |
| 237373 | how is soil created from rocks | no | no | - |

## Interpretation

This is not a full replacement for the 50-query gate, but it validates the specific bottleneck seen in high-call misses: some questions needed a wider first evidence pool rather than more repeated follow-up searches. The change keeps a hard runtime cap (`limit <= 20`, `read_top_k <= 5`) so broad searches can recover more candidates without letting the LLM request unbounded context.

