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

## 5. Relevance-aware expansion budget (`SYNAPTIC_EXPAND_RELEVANCE`, commit c2360a5)

When the expander budget is full, keep the most query-relevant non-seed
neighbours instead of the first-visited ones (seeds protected). Hypothesis: on
KRRA's 90k neighbourhoods, relevant neighbours were being dropped before the
reranker saw them.

| corpus | relevance ON | baseline (default) | Δ |
|---|---:|---:|---:|
| KRRA Hard | 29/39 (0.744), 551s | 30/39 (0.769), 586s | **−1 (noise, not a win)** |

**Verdict: neutral.** Together with §4 this is now TWICE that tweaking the
expansion's neighbour *selection* did nothing. Mechanism: the reranker
re-scores every candidate, so given any reasonable candidate pool, *which*
neighbours won the budget doesn't change the final ranking.

### The load-bearing learning
- Graph **ON vs OFF** = **+8.3pp** (§3): the agent gain is from REACH —
  whether the graph reaches evidence the seeds missed at all.
- Graph neighbour **selection** (§4 explicit-expand, §5 relevance-budget) =
  **0**: once reached, the reranker handles relevance.
- ⇒ **The live lever is REACH/connectivity, not selection.** The 29% isolated
  nodes in KRRA (§1) are pure reach-absence — the +8.3pp can't touch them. The
  next development is adding reach to islands (the opposite of selection-tuning),
  measured ON THE AGENT (where reach paid off), not single-shot (where the §2
  ablation said an index backbone won't help ranking).
- Both selection tweaks kept opt-in (default off). Legible city-map (commit
  3e91181) is a DIFFERENT mechanism (priming/wayfinding, not expansion) — still
  to be measured.

## 6. Sufficiency gate — re-verified before default-on (commit 3941e82)

The +3.2pp gate result lived only in memory from a prior code state, and the
eval harness's inline loop doesn't exercise the gate. So before flipping the
default, re-measured on CURRENT code via `run_agent_loop` (the real gated path),
`examples/ablation/gate_ab.py`, KRRA Hard, gate OFF vs ON:

| | id-reach | time |
|---|---:|---:|
| gate **OFF** | 29/39 (0.744) | 719s |
| gate **ON** | **32/39 (0.821)** | 864s |
| **Δ** | **+3 (+7.7pp)** | +20% latency |

Reproduces the recorded win (was +5.1pp on KRRA Hard) on current code → default-on
justified. Minor: 2 enumeration queries hit the model's 32k context at turn 10
(both arms; fail-open) — a separate context-budget limit, not the gate.

**Caveat added after §7's noise-floor finding:** this single-run KRRA Hard
+7.7pp (29→32/39) carries the SAME single-run weakness — on KRRA the gate-ON
arm later read 31–32 across runs, and gate-OFF could vary too, so +3/39 here is
only borderline above noise. The gate's default-on does NOT rest on THIS run:
it rests on the original multi-bench aggregate (+3.2pp over 93 queries across
benches, 0 regressions — much harder to dismiss as noise than one 39q run) plus
the gate being fail-open/harmless. Treat §6 as "consistent with the win, not
independent proof of it." See `[[project_v028_agent_bench_noise_floor]]`.

## 7. Bridge-aware gate (L29b) — multi-hop chaining (commit pending)

The §5 learning was: the live lever is REACH, not selection. The query-time
answer to reach on *multi-hop* questions: when the sufficiency gate fires, don't
just say "find the missing evidence" — have the judge name the concrete
follow-up search query with the **bridge entity from the evidence spelled out**,
and relay it as an explicit chained search. No index-time OpenIE triples; the
agent loop does the hop. Opt-in `gate_bridge` / `SYNAPTIC_GATE_BRIDGE=1`.

A/B via `gate_ab.py --compare bridge` on **finreg_multihop** (120q, all
`multi_hop`, strict `relevant.issubset(found)` scoring — both hops required),
gate ON plain vs gate ON + bridge:

**TWO runs each (raw bridge, then grounded bridge — see §8 for grounding),
gate ON plain vs gate ON + bridge:**

| bench | run | plain | bridge | Δ |
|---|---|---:|---:|---:|
| finreg multihop | run1 (raw)      | 93/120  | 99/120  | +6 |
| finreg multihop | run2 (grounded) | **101/120** | 101/120 | 0 |
| KRRA Hard       | run1 (raw)      | 32/39   | 31/39   | −1 |
| KRRA Hard       | run2 (grounded) | 31/39   | 32/39   | +1 |

**HONEST CONCLUSION — the bridge effect is inside the noise floor.** The SAME
plain arm (gate_bridge=False, an UNCHANGED code path between the two runs) moved
**93→101 on finreg = 8/120 (6.7pp) of pure run-to-run nondeterminism** (vLLM
sampling at temp>0, concurrency-6 races). Every bridge Δ measured (+6, 0, −1,
+1) sits INSIDE that ±8/120 floor. So:
- the run1 "+5.0pp" was NOT a real win — the plain baseline merely landed low
  (93) that run; it was single-RUN noise mistaken for signal (commit 601b46a's
  claim is **retracted**);
- grounding (§8) neither helped nor hurt above noise either (finreg 101=101,
  KRRA 32 vs 31 — both within ±1).

**Decision: bridge (grounded) stays OPT-IN, default OFF — but with NO measured
accuracy claim.** The mechanism is sound and, in grounded form, provably
harmless (it only relays a next_query whose entity is present in the evidence;
on a hallucinated bridge it falls back to the plain nudge). There is simply no
*measured* win above this rig's noise floor. To ever claim an agent-bench delta
of this size you need repeated trials to estimate variance — a single A/B run
cannot. (Methodology recorded: `[[project_v028_agent_bench_noise_floor]]`.)

## 8. Bridge grounding (self-gating, commit pending)

The hypothesis behind grounding: raw bridge's run1 KRRA −1 came from the judge
*hallucinating* a bridge entity on non-multihop queries and sending the agent
after it. Fix (corpus-agnostic, query-time, no index LLM): before relaying the
judge's `next_query`, require its NOVEL tokens (those not already in the
question) to appear verbatim in the gathered evidence — the bridge premise is
"the evidence already names the entity". Ungrounded → fall back to the plain
nudge. `_bridge_is_grounded()` in `agent_loop.py`; tiny universal stoplist, no
per-corpus words.

Outcome: grounding is the right *safety* refinement (it can only ever turn a
relay into the plain nudge, never the reverse), but its accuracy effect — like
the bridge itself — is inside the noise floor (finreg 101=101, KRRA 31→32). Kept
because it makes the opt-in knob strictly safer at zero measured cost, NOT
because it was shown to help. Both the raw and grounded bridge are the same
opt-in flag (`SYNAPTIC_GATE_BRIDGE=1`); grounded is the only shipped behaviour.

## 9. Agent context-overflow guard (deterministic fix, commit pending)

Every long agent run logged `agent LLM call failed at turn 12/13: ... maximum
context length is 32768 tokens` — the unbounded message history overflowed the
window and the loop BROKE, discarding all remaining retrieval. 2 such hard
failures in the §7 KRRA canary (enumeration queries, max_turns=15). This is the
opposite of a noise-floor delta: a hard, reproducible 400 you can COUNT.

Fix (`agent_loop.py`): fold the oldest tool-result contents to a stub before
each LLM call (proactive, `SYNAPTIC_AGENT_HISTORY_BUDGET` chars, default 48k);
on a context-length 400 slipping through, compact to half-budget and retry once
(reactive) instead of breaking. Order + tool_call_id pairing preserved (only the
content string shrinks), recent evidence kept (oldest-first), `found_ids`
untouched.

Validated (`examples/ablation/overflow_check.py`, KRRA Hard, gate default-on):

| | count |
|---|---:|
| escaped (hard 400s that broke a turn) | **0** (was 2) |
| reactive retries that fired | 0 (proactive caught all) |
| solved | 32/39 (unchanged vs gate-ON baseline) |

A deterministic before/after (2→0 hard failures), immune to the §7 noise floor.
The reactive net never fired — proactive 48k budget sufficed — but it stays as
the estimate-independent backstop. This is the right KIND of agent work given
§7: not chasing sub-noise accuracy deltas, but removing a hard failure that
silently capped long-session reach (a query dying at turn 12 can't reach
anything regardless of how good the retrieval levers are).

## Status
- Shipped (commit 80aba7c): nav_metrics + navigability scan +
  `SYNAPTIC_GRAPH_EXPANSION` toggle (measurement infra — keep) and the `expand`
  query-aware ranking + island fallback (measured neutral → demoted to opt-in,
  `SYNAPTIC_NAV_UPGRADE` default OFF).
- Decided: explicit-`expand` upgrade is not a win on the agent's actual path.
- Next: port query-aware neighbour selection into EvidenceSearch's internal
  GraphExpander step (fires on every `search` — the §3 +8.3pp path), measure on
  the agent benches the same way.
