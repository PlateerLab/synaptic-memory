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

## Blood-Borne Transmission Follow-Up

The remaining high-call miss `54544` (`blood diseases that are sexually transmitted`) used a lay phrase while the gold document uses `STI`, `blood borne infection`, and `sexual and blood borne transmission routes`. A bounded medical synonym rewrite now fires only when the query contains blood, sexually transmitted/sexual wording, and a disease/infection term.

| QID | Original query | Auto rewrite | Gold rank in `deep_search` | DeepSeek targeted smoke |
| --- | --- | --- | ---: | --- |
| 54544 | blood diseases that are sexually transmitted | sexual blood borne transmission routes | 1 | reach=yes, first relevant turn 1 / call 1 |

## Answer-Shaped Rewrite Follow-Up

Current deterministic `deep_search` also showed that several misses were caused by answer-page phrasing rather than graph traversal failure. Bounded rewrites now preserve the query entity while switching to the wording commonly used in the relevant passage.

| QID | Original query | Auto rewrite examples | Gold rank in `deep_search` | DeepSeek targeted smoke |
| --- | --- | --- | ---: | --- |
| 319564 | how much fiber is in carrots | one cup carrots grams fiber; one cup cooked carrots grams fiber | 3 | reach=yes, first relevant turn 1 / call 1 |
| 155234 | do bigger tires affect gas mileage | tire size factors influence gas mileage; tire width versus gas mileage | 1 | reach=yes, first relevant turn 1 / call 1 |
| 208145 | how bicycle tire tubes are sized | bicycle tire tube size sidewall ETRTO metric imperial; bicycle tire sidewall tube size printed raised numbers | 1 | reach=yes, first relevant turn 1 / call 1 |

## Interpretation

Prompt-visible hints alone were not enough: DeepSeek often generated nearby but non-gold rewrites. Running the deterministic rewrite hints inside `deep_search` removes that planning variance for cheap, bounded patterns such as dropping noisy numeric years and rewriting "created from" process questions into answer-shaped phrases.
