# cost-at-quality — mixed pool (v0.29 E4 keystone)

common paired pool: 120 queries (finreg_multihop=120)
oracle labels: derived_from_pool

| arm | solve | rate | tokens/query | cost vs always-agent | min runs |
|---|---:|---:|---:|---:|---:|
| always-RAG | 69/120 | 0.575 | 3,239 | 0.09x | 3 |
| always-agent | 68/120 | 0.567 | 34,677 | 1.00x | 3 |
| oracle-router | 86/120 | 0.717 | 8,066 | 0.23x | 3 |
| ask() | — | — | — | — | — (unmeasured) |

## acceptance gates (plan §E4)

- **quality**: NOT EVALUABLE — ask arm unmeasured — run the live arm (--run-live) or load --ask-jsonl
- **cost**: NOT EVALUABLE — ask arm unmeasured — run the live arm (--run-live) or load --ask-jsonl
- **separation**: NOT EVALUABLE — ask arm unmeasured — run the live arm (--run-live) or load --ask-jsonl

oracle ceiling: 86/120 solve at 8,066 tok/q — the routing headroom ask() competes for

claim scope: per plan §E4 — only the agent-required-class delta is claimed (noise floor multiples); domain generalization (finreg+assort, 2 domains) is NOT established by this report.
