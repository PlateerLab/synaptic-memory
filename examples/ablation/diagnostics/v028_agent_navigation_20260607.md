# v0.28 — Agent navigation: measurement log (2026-06-07)

Durable record of every evaluation run in the "agent navigation upgrade" track.
Raw run logs were in `/tmp` (ephemeral); this file is the permanent record.

Infra: vLLM Qwen3.6-27B LLM @ :8012, Qwen3-Embedding-4B @ :8013 (dim 2560),
SqliteGraphBackend, custom graphs in `eval/data/*_graph.sqlite`. Agent runs are
`eval/run_all.py --agent --agent-only`, id-reach only (no judge) → `solved` =
`found_ids ∩ relevant_docs` (pure retrieval reach, not answer quality).

---

## 1. Corpus navigability scan (LLM-free, `examples/ablation/graph_navigability.py`)

Is the graph reachable as one component, or fragmented into islands?

| graph | nodes | edges | largest component | isolated | components | verdict |
|---|---:|---:|---:|---:|---:|---|
| finreg | 4,476 | 12,550 | **97.0%** | 0.0% | 8 | navigable ✓ |
| assort | 13,909 | 20,450 | **100.0%** | 0.0% | 1 | navigable ✓ |
| **KRRA** | 90,125 | 292,429 | **71.1%** | **28.9%** | 26,076 | **fragmented** |
| **x2bee** | 19,843 | 18,744 | **0.4%** | 0.5% | 1,099 | **shattered** (largest=72) |

Hop distance (BFS, largest component): finreg median 5, assort 5, KRRA 6, all
heavy-tailed hubs (max/avg degree 100–175x). → where connected, it IS small-world;
the problem is fragmentation, not hop count.

---

## 2. Single-shot: graph expansion ON vs OFF (`SYNAPTIC_GRAPH_EXPANSION`, FTS-only, `--quick`)

| Dataset | ON MRR | OFF MRR | Δ |
|---|---:|---:|---:|
| KRRA Easy | 0.967 | 0.975 | −0.008 |
| KRRA Hard | 0.583 | 0.579 | −0.004 |
| KRRA Conv | 0.146 | 0.196 | **−0.050** |
| assort Easy/Hard | 0.856 / 0.000 | 0.856 / 0.000 | 0 |
| X2BEE Easy/Hard/Conv | 1.000 / 0.368 / 0.164 | = | 0 |
| assort Conv | 0.472 | 0.472 | 0 |
| finreg | 0.693 | 0.735 | **−0.042** |
| finreg multihop | 0.613 | 0.707 | **−0.094** (but R@10 0.662→0.504: graph *helps recall*) |
| HotPotQA-24 | 0.909 | 0.909 | 0 |
| Allganize ko / Eval | 0.959 / 0.927 | = | 0 |
| PublicHealthQA | 0.611 | 0.611 | 0 |
| AutoRAG | 0.895 | 0.895 | 0 |
| **latency** | 11.5 / 26 / 14s | **3.7 / 7 / 2.7s** | **2–4x faster OFF** |

**Conclusion:** graph expansion is net-**negative** on single-shot ranking
(15/16 equal-or-better OFF) and 2–4x slower. Only multi-hop *recall* benefits.
→ An index-time connectivity backbone would NOT raise single-shot quality.

---

## 3. Agent mode: graph expansion ON vs OFF (finreg multihop, 5 turns, concurrency 6)

| | id-reach | time |
|---|---:|---:|
| graph **ON** | **94/120 (0.783)** | 1274s |
| graph **OFF** | **84/120 (0.700)** | 1350s |
| **Δ** | **+10 (+8.3pp) with graph** | |

**Conclusion (the gate result):** OPPOSITE sign from single-shot. Graph traverse
**helps the agent +8.3pp**. The graph's value is realized through the agent's
iterative traverse, not single-shot ranking → improving traversal is the right
lever. Recorded in memory `project_v028_graph_helps_agent_not_singleshot`.

---

## 4. Navigation upgrade validation (`expand` query-aware ranking + island fallback, commit 80aba7c)

A/B via `SYNAPTIC_NAV_UPGRADE` (1=shipped, 0=pre-upgrade behaviour).

| corpus | upgrade ON | upgrade OFF | Δ | semantic-fallback fires |
|---|---:|---:|---:|---:|
| finreg multihop | 94/120 (0.783) | (=baseline ON §3) | **0** | **0** (finreg has no islands) |
| KRRA Hard | 29/39 (0.744) | **30/39 (0.769)** | **−1 (noise, not a win)** | **0** |

**Verdict: the explicit-`expand` upgrade is NEUTRAL-to-noise — not a win.** Even
on KRRA (29% islands) the semantic fallback fired 0 times. → demoted to opt-in
(`SYNAPTIC_NAV_UPGRADE`, default OFF). The measurement infra (navigability scan,
nav_metrics, `SYNAPTIC_GRAPH_EXPANSION` toggle) stays — it's what produced these
findings. The query-aware-ranking + island ideas move to EvidenceSearch's
internal expansion next (the path that actually fires, §3).

**Observation (important):** semantic fallback fired **0 times even on KRRA
(29% islands)**. The agent reaches evidence via the `search` tool and rarely
calls the explicit `expand` tool on island nodes → the upgrade landed on a
code path the agent barely uses. The graph's +8.3pp (§3) comes from the
`search` tool's INTERNAL GraphExpander, not the explicit `expand` tool.

**Implication:** if KRRA Δ ≈ 0 → the explicit-`expand` upgrade is inert in
practice; the same ideas (query-aware neighbour selection, island handling)
should move INTO EvidenceSearch's internal expansion, which fires on every
search and is what produced the §3 gain. (Next development target.)

---

## Status
- Shipped (commit 80aba7c): nav_metrics + navigability scan +
  `SYNAPTIC_GRAPH_EXPANSION` toggle (measurement infra — keep) and the `expand`
  query-aware ranking + island fallback (measured neutral → demoted to opt-in,
  `SYNAPTIC_NAV_UPGRADE` default OFF).
- Decided: explicit-`expand` upgrade is not a win on the agent's actual path.
- Next: port query-aware neighbour selection into EvidenceSearch's internal
  GraphExpander step (fires on every `search` — the §3 +8.3pp path), measure on
  the agent benches the same way.
