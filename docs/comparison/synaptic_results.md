# Synaptic Memory — reproducible benchmark numbers

Every number on this page is regenerable from source on a laptop. When
a number has a private corpus attached, that's flagged.

The companion file [published_numbers.md](published_numbers.md) lists
what competitors have claimed about their own systems; this one lists
what we can demonstrate about ours.

Last updated: 2026-04-17 (Synaptic v0.15.0).

---

## Tier 1 — public, reproducible in under 2 seconds on a laptop

Run::

    pip install "synaptic-memory[korean]"
    python examples/benchmark_allganize.py

```
Dataset                  Corpus  Queries      MRR     R@10        Hit     Time
--------------------------------------------------------------------------------
Allganize RAG-ko            200      200    0.947    1.000   200/200     9.3s
Allganize RAG-Eval          300      300    0.911    0.950   285/300     5.9s
```

- Engine: `graph.search()` defaults to **EvidenceSearch** in v0.16.0+
  (BM25 + PPR + MMR + graph expansion). **No embedder, no
  cross-encoder, no LLM at any point.**
- Two releases of cumulative gain from the v0.15.0 legacy baseline
  (0.621 / 0.615):
  * v0.15.1 — query-mode Kiwi → 0.743 / 0.695 (+0.12 / +0.08)
  * v0.16.0 — engine default flipped to evidence → 0.947 / 0.911
    (+0.20 / +0.22)
  Full ablation per release: `examples/ablation/run_ablation.py`.
- Deterministic. Re-running from a clean checkout produces bitwise
  identical numbers.
- Data source:
  [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)
  (MIT) — snapshot shipped under
  [`tests/benchmark/data/`](../../tests/benchmark/data/).
- Source code: [`examples/benchmark_allganize.py`](../../examples/benchmark_allganize.py)
  (~100 lines, no external services).

**This is the floor.** Any hypothesis-testing should start here.

---

## Release-over-release ablation (v0.15.0 → v0.16.0)

Locked in by [`examples/ablation/run_ablation.py`](../../examples/ablation/run_ablation.py)
on 2026-04-17 (v0.16.0):

| Dataset | Lang | Queries | v0.15.0 | v0.15.1 (kiwi) | **v0.16.0 (evidence + kiwi)** | Total Δ |
|---------|------|---------|---------|----------------|-------------------------------|---------|
| Allganize RAG-ko | ko | 200 | 0.621 | 0.743 | **0.947** | +0.326 |
| Allganize RAG-Eval | ko | 300 | 0.615 | 0.695 | **0.911** | +0.296 |
| PublicHealthQA KO | ko | 77 | 0.318 | 0.466 | **0.546** | +0.228 |
| AutoRAG KO | ko | 114 | 0.592 | 0.692 | **0.906** | +0.314 |
| HotPotQA-24 | en | 24 | 0.727 | 0.727 | **0.875** | +0.148 |

Two independent changes compose:

1. **v0.15.1** — ``_normalize_korean(text, query_mode=True)`` drops
   Kiwi-surviving interrogative / copular noise forms at query time
   only. Index-time normalisation is unchanged, so existing graph
   files do not need to be rebuilt. Kiwi runs only when the query is
   ≥50 % Hangul, which is why HotPotQA (English) is mathematically
   untouched on this step.
2. **v0.16.0** — ``SynapticGraph.search()`` default engine flipped
   from legacy ``HybridSearch`` to ``EvidenceSearch``. This was the
   biggest single lever; both Korean and English benchmarks improve.

Streaming invariance also sharpens on the new default — see
[docs/paper/theorem.md](../paper/theorem.md) Section 3.4 for the
empirical numbers (96 % bit-wise top-10 identical, 100 % top-1
identical, |ΔMRR| exactly zero on the v0.16.0 streaming rerun).

---

## Tier 1.5 — English multi-hop standard benchmarks (v0.16.0)

Run via::

    python examples/ablation/download_benchmarks.py
    python examples/ablation/run_tier1_benchmarks.py --subset 500

Datasets are regenerated from public HuggingFace releases
(see [`examples/ablation/download_benchmarks.py`](../../examples/ablation/download_benchmarks.py));
the JSON snapshots under `tests/benchmark/data/` are gitignored.

| Dataset | Source | Docs | Queries (subset) | MRR @ 10 | R@5 | R@10 | Hit @ 10 | Reference (not head-to-head) |
|---------|--------|------|------------------:|---------:|----:|-----:|---------:|------------------------------|
| HotPotQA dev (distractor) | `hotpot_qa/distractor` | 66,635 | 500 / 7,405 | **0.784** | 0.585 | 0.658 | 459/500 (91.8 %) | HippoRAG2: 56.7 % string acc. |
| MuSiQue-Ans dev | `dgslibisey/MuSiQue` | 21,100 | 500 / 2,417 | 0.590 | 0.379 | 0.440 | 381/500 (76.2 %) | HippoRAG2: F1 51.9, R@5 74.7 % |
| 2WikiMultihopQA dev | `voidful/2WikiMultihopQA` | 56,687 | 500 / 12,576 | **0.795** | 0.501 | 0.552 | 456/500 (91.2 %) | HippoRAG2: R@5 90.4 % |

**Reading the table honestly.**

* **HotPotQA** — hit@10 of 91.8 % at an embedder-free cost is
  competitive with published numbers, but note that HippoRAG2's
  published 56.7 % is *answer string accuracy* after a reader LM
  consumes the retrieved passages, not a retrieval metric. Our
  number is retrieval-only; a reader stage on top would drop it.
* **2WikiMultihopQA** — Synaptic lands at R@5 = 0.501 vs
  HippoRAG2's R@5 = 0.904. Closing that gap with the PPR stage
  alone is unlikely; the gap is the value an LLM-built entity-and-
  relation graph brings on extra-long 2-hop chains.
* **MuSiQue** — by far the hardest dataset for an embedder-free
  system. MuSiQue is 2-4 hops with decomposition, and HippoRAG2's
  R@5 = 0.747 vs. our 0.379 reflects that the PPR seeds derived
  purely from lexical matches cannot chain through intermediate
  entities. Embedder + cross-encoder is the expected next lift,
  tracked for v0.16.1.

**Scope note.** These corpora are far larger than the Allganize /
PublicHealthQA sets above (66 k docs vs. 200-300), so a 500-query
subset is the per-run scale that's comfortably within `uv run` time
budgets. Total wall clock for a 500-query run on a laptop: HotPotQA
~15 min, MuSiQue ~4 min, 2Wiki ~11 min. A full-dataset run is tracked
for v0.16.1 after performance optimisation of the PPR stage
(currently O(corpus) on first hit).

**Scope note.** These corpora are far larger than the Allganize /
PublicHealthQA sets above (66 k docs vs. 200-300), so a 500-query
subset is the per-run scale that's comfortably within `uv run` time
budgets. A full-dataset run is tracked for v0.16.1 after performance
optimisation of the PPR stage (currently O(corpus) on first hit).

## Tier 2 — public datasets, full pipeline (embedder + reranker)

Run (requires GPU-backed embedder + reranker):

```bash
uv run python eval/run_all.py --quick \
    --embed-url http://localhost:11434/v1 \
    --reranker-url http://localhost:8180
```

Embedder: `qwen3-embedding:4b` via Ollama. Reranker:
`bge-reranker-v2-m3` via TEI. Pipeline: `EvidenceSearch` (BM25 +
HNSW + PPR + cross-encoder + MMR).

| Dataset | Language | Corpus | Queries | MRR | Hit rate |
|---------|----------|--------|---------|-----|----------|
| HotPotQA-24 (subset) | English multi-hop | 226 | 24 | **0.964** | 24/24 |
| Allganize RAG-ko | Korean enterprise | 200 | 200 | **0.905** | — |
| Allganize RAG-Eval | Korean finance/medical/legal | 300 | 300 | **0.874** | — |
| PublicHealthQA KO | Korean public health | 77 | 77 | **0.600** | 56/77 |

**Honesty notes:**
- HotPotQA-24 is a 24-question subset used by the Cognee comparison.
  The full HotPotQA-dev (7,405 q) run is planned for v0.16.0. We do
  not claim parity with published HotPotQA numbers yet.
- PublicHealthQA 0.600 is weak. We include it so the picture isn't
  cherry-picked.

---

## Tier 3 — private corpora (internal QA only, not for public claims)

These corpora came from production deployments and can't be
redistributed. They're here to document what the full pipeline
achieves on representative data we own:

| Dataset | Type | Corpus | Queries | MRR | Hit |
|---------|------|--------|---------|-----|-----|
| KRRA Easy | Korean documents (private) | 19,720 | 20 | 0.967 | 20/20 |
| KRRA Hard | Korean documents (private) | 19,720 | 15 | 1.000 | 15/15 |
| X2BEE Easy | PostgreSQL e-commerce (private) | 19,843 | 20 | 1.000 | 20/20 |
| assort Easy | Fashion CSV (private) | 13,909 | 15 | 0.867 | 13/15 |

**Use these internally only.** They're useful for regression
detection, not for external claims — we cannot provide reproduction
steps.

---

## Multi-turn agent (GPT-4o-mini, max 5 turns)

| Dataset | Result |
|---------|--------|
| KRRA Hard | 10–13 / 15 (67–87 %) |
| X2BEE Hard | 17 / 19 (89 %) |
| assort Hard | 12 / 15 (80 %) |

These exercise `deep_search` + `filter_nodes` + `aggregate_nodes` +
`join_related` in concert. The score ranges come from run variance
across the LLM judge — GPT-4o-mini is slightly nondeterministic at
temperature 0.

---

## CDC / streaming invariance

Not a benchmark — a **property** we guarantee by regression test:

```
tests/test_cdc_search_regression.py  (locks top-k equivalence)
```

On X2BEE production Postgres, 19,843 rows:

| Path | Time |
|------|------|
| Initial CDC load | 51 s |
| Full reload (baseline) | 35 s |
| Idempotent resync (no changes) | **6 s** |
| Top-1 match vs full reload | **4 / 4** |

The 6-second idempotent resync is the number that justifies the
whole CDC bet — full reload cost is incurred once per day of reality,
not per query.

---

## What we have not measured and won't claim

- **LoCoMo / LongMemEval.** We haven't run them — they evaluate
  conversational memory (Mem0 / Zep territory), not document
  retrieval. We will run them as part of the Streaming Retrieval
  paper (planned 2026 Q3).
- **MuSiQue / 2Wiki full runs.** Synaptic's PPR descends from
  HippoRAG, so a head-to-head here is fair. Planned v0.16.0.
- **Full HotPotQA-dev (7,405 q).** Planned v0.16.0.
- **Production latency @ 10 M nodes.** We've deployed the Postgres
  backend at that scale internally but haven't published the
  latency distribution. Planned.

Gaps we admit are gaps, not silent. See also the "what we haven't
claimed" section in [README.md](../../README.md).
