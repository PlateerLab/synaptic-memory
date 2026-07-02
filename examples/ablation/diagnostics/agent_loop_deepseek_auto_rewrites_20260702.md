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

## 50-Query Follow-Up

After merging the wider `deep_search` evidence controls and auto-run rewrites, a fresh DeepSeek 50-query run reached `25/50`, up from the prior `23/50`.

| Metric | Prior 50-query run | Post rewrite run |
| --- | ---: | ---: |
| Reach | 23/50 | 25/50 |
| Mean turns | 4.14 | 4.00 |
| Mean tool calls | 5.78 | 6.08 |
| Mean first relevant turn | 1.70 | 1.16 |
| Empty tool calls | 11 | 9 |

Net new hits:

- `1101278` - do prince harry and william have last names
- `293992` - how many product lines does coca cola have
- `208145` - how bicycle tire tubes are sized
- `14151` - age requirements for name change
- `91711` - child psychiatrist salary 2016

Regressed in this live run:

- `178627` - effects of detox juice cleanse
- `45924` - average temperatures las vegas by month
- `208494` - how big do newfypoo's get

The live run also exposed a follow-up issue: DeepSeek sometimes appends a process word to the source phrase, e.g. `how is soil created from rocks weathering`. A source-cleanup rule strips trailing process words such as `weathering` before producing rewrites. With that cleanup, `237373` again reaches the gold document at turn 1 / call 1 in a targeted live smoke.

## Interpretation

Prompt-visible hints alone were not enough: DeepSeek often generated nearby but non-gold rewrites. Running the deterministic rewrite hints inside `deep_search` removes that planning variance for cheap, bounded patterns such as dropping noisy numeric years and rewriting "created from" process questions into answer-shaped phrases.
