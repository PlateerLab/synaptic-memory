# cost-at-quality — mixed pool (v0.29 E4 keystone)

common paired pool: 240 queries (finreg=120, finreg_multihop=120)
oracle labels: routing_gt

| arm | solve | rate | tokens/query | cost vs always-agent | min runs |
|---|---:|---:|---:|---:|---:|
| always-RAG | 152/240 | 0.633 | 3,216 | 0.09x | 3 |
| always-agent | 149/240 | 0.621 | 35,170 | 1.00x | 3 |
| oracle-router | 184/240 | 0.767 | 7,903 | 0.22x | 3 |
| ask() | — | — | — | — | — (unmeasured) |

## acceptance gates (plan §E4)

- **quality**: NOT EVALUABLE — ask arm unmeasured — run the live arm (--run-live) or load --ask-jsonl
- **cost**: NOT EVALUABLE — ask arm unmeasured — run the live arm (--run-live) or load --ask-jsonl
- **separation**: NOT EVALUABLE — ask arm unmeasured — run the live arm (--run-live) or load --ask-jsonl

oracle ceiling: 184/240 solve at 7,903 tok/q — the routing headroom ask() competes for

claim scope: per plan §E4 — only the agent-required-class delta is claimed (noise floor multiples); domain generalization (finreg+assort, 2 domains) is NOT established by this report.
