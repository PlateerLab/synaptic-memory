# DeepSeek agent loop auto-rewrite check

- Run at: 2026-07-02 21:34 KST
- Model: `deepseek-v4-flash`
- Base corpus: `tests/benchmark/data/msmarco_passage_full.json`
- SQLite DB: `tests/benchmark/data/msmarco_full.db`
- Corpus limit: 8,841,823 passages
- Target set: two high-call misses whose gold evidence is reachable through simple deterministic rewrites.
- Change under test: `deep_search` runs bounded deterministic rewrite hints internally and merges the rewrite evidence before returning to the LLM.

## Deterministic Check

| QID | Original query | Auto rewrite | Gold rank in `deep_search` |
| --- | --- | --- | ---: |
| 91711 | child psychiatrist salary 2016 | child psychiatrist salary | 1 |
| 237373 | how is soil created from rocks | making soil rock pieces; small pieces of rock form soil | 1 |

## DeepSeek Live Smoke

| QID | Before auto-run hints | After auto-run hints | First relevant |
| --- | ---: | ---: | --- |
| 91711 | no | yes | turn 1 / call 1 |
| 237373 | no | yes | turn 1 / call 1 |

## Interpretation

Prompt-visible hints alone were not enough: DeepSeek often generated nearby but non-gold rewrites. Running the deterministic rewrite hints inside `deep_search` removes that planning variance for cheap, bounded patterns such as dropping noisy numeric years and rewriting "created from" process questions into answer-shaped phrases.

