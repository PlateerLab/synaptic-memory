# Tier-0 routing-signal AUC report (v0.29 E2)

- GT: `eval/data/routing_gt_v029.jsonl` — 624 records (train 305 / heldout 319 / other-split 0)
- Retrieval results: `assort=eval/data/hits/assort_hits_fts_k5.jsonl, assort_hard=eval/data/hits/assort_hard_hits_fts_k5.jsonl, assort_conversational=eval/data/hits/assort_conversational_hits_fts_k5.jsonl, x2bee=eval/data/hits/x2bee_hits_fts_k5.jsonl, x2bee_hard=eval/data/hits/x2bee_hard_hits_fts_k5.jsonl, x2bee_conversational=eval/data/hits/x2bee_conversational_hits_fts_k5.jsonl, krra=eval/data/hits/krra_hits_fts_k5.jsonl, krra_hard=eval/data/hits/krra_hard_hits_fts_k5.jsonl, krra_conversational=eval/data/hits/krra_conversational_hits_fts_k5.jsonl, autorag=eval/data/hits/autorag_hits_fts_k5.jsonl`
- Tiers: {'confirmed': 240, 'hit_only': 169, 'provisional': 187, 'unmeasured': 28}
- Labels: {'agent_required': 136, 'both': 168, 'cheap_sufficient': 40, 'single_shot_hit': 166, 'single_shot_miss': 3, 'unlabeled': 28, 'unsolved': 83} (positive=agent_required; negative=['both', 'cheap_sufficient']; unsolved excluded from AUC/recall)

## Signal AUC — confirmed tier, held-out split (verdict basis)

| signal | AUC | n_pos | n_neg | NaN | note |
|---|---:|---:|---:|---:|---|
| s1_structured_lexicon | 0.500 | 17 | 74 | 0 |  |
| s2_zero_results | — | 0 | 0 | 91 | requires retrieval pass |
| s2_score_flatness | — | 0 | 0 | 91 | requires retrieval pass |
| s2_margin_deficit | — | 0 | 0 | 91 | requires retrieval pass |
| s3_table_row_in_topk | — | 0 | 0 | 91 | requires retrieval pass |

## recall @ precision — confirmed tier, held-out split

| signal | R@P≥0.90 | R@P≥0.80 | R@P≥0.70 | R@P≥0.50 |
|---|---:|---:|---:|---:|
| s1_structured_lexicon | — | — | — | — |
| s2_zero_results | — | — | — | — |
| s2_score_flatness | — | — | — | — |
| s2_margin_deficit | — | — | — | — |
| s3_table_row_in_topk | — | — | — | — |

## Reference — provisional / hit-only tiers, held-out split (no verdict weight)

| signal | AUC | n_pos | n_neg | NaN | note |
|---|---:|---:|---:|---:|---|
| s1_structured_lexicon | 0.623 | 53 | 113 | 0 |  |
| s2_zero_results | 0.587 | 46 | 113 | 7 | 7 NaN dropped |
| s2_score_flatness | 0.618 | 38 | 113 | 15 | 15 NaN dropped |
| s2_margin_deficit | 0.664 | 38 | 113 | 15 | 15 NaN dropped |
| s3_table_row_in_topk | 0.608 | 46 | 113 | 7 | 7 NaN dropped |

## Tuned OR-combo (thresholds selected on train split only)

| signal | threshold |
|---|---:|
| s1_structured_lexicon | disabled |
| s2_zero_results | disabled |
| s2_score_flatness | disabled |
| s2_margin_deficit | disabled |
| s3_table_row_in_topk | disabled |

- train recall (confirmed): 0.000 (0/15)
- train escalation autorag: 0.000 (0/56)
- train escalation assort_easy: 0.000 (0/0)

## Held-out verdict

- agent-required recall (confirmed): 0.000 (0/17) — gate ≥ 0.90
- precision on confirmed labels: —
- escalation autorag: 0.0% (0/58) — budget ≤ 15%
- escalation assort_easy: 0.0% (0/15) — budget ≤ 20%

**VERDICT: NO-GO** — adopt conservative routing for E3: promote high-confidence positives only; defer the rest to the tier-1 sufficiency gate.
