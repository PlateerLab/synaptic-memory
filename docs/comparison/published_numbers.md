# Competitor published benchmark numbers

A honest catalogue of numbers the agent-memory / GraphRAG systems have
published themselves, with sources. This is not a head-to-head
comparison (the harness in
[`examples/benchmark_vs_competitors/`](../../examples/benchmark_vs_competitors/)
is that). It's reference data for context.

> **Warning.** Every row in this table is from its own authors'
> self-reported evaluation, on a corpus and metric definition they
> chose. The Zep correction incident (see below) is the main reason
> we keep this file separate from our own measurements.

Last updated: 2026-04-17.

---

## Mem0

Paper: ["Mem0: Building Production-Ready AI Agents with Scalable
Long-Term Memory"](https://arxiv.org/abs/2504.19413) — Chhikara et al.,
ECAI 2025 (arXiv:2504.19413).

| Benchmark | Score | Notes |
|-----------|-------|-------|
| LoCoMo | **91.6** (self-reported, ECAI '25) | metric = LLM-judge + F1 + BLEU blend, not MRR |
| LongMemEval | **93.4** | same blend |
| BEAM 1M | **64.1** | |
| BEAM 10M | **48.6** | |

Mem0 also claims "91% lower response time than full-context
approaches" in the same paper.

### Independent finding
A 2026 dev.to comparison ([Bhardwaj, 2026](https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3))
puts Mem0's **temporal reasoning accuracy at 49.0 %** on LoCoMo's
temporal subset — meaning Mem0's headline 91.6 average masks a weak
spot in time-aware queries.

---

## Zep / Graphiti — the LoCoMo correction incident

Zep's 2025 paper originally claimed **84 %** on LoCoMo. A public
correction ([getzep/zep-papers#5](https://github.com/getzep/zep-papers/issues/5),
raised by the Mem0 team and acknowledged by Zep) revised this to:

| Measurement | Corrected score |
|-------------|-----------------|
| Zep v2 on LoCoMo (4 validated categories) | **58.44 % ± 0.20** |
| Zep previous version | 65.99 % ± 0.16 |

Root cause: Zep's original calculation **included questions from an
adversarial category that the LoCoMo protocol explicitly excludes**.

Zep subsequently rebutted, arguing a reconfigured Zep scores 75.14 %.
The numbers on record depend on who configured the test — hence the
fairness harness.

This is the single most-cited reason in the community why self-reported
agent-memory numbers are treated with suspicion in 2026.

Sources:
- [Revisiting Zep's 84% LoCoMo claim — GitHub issue #5](https://github.com/getzep/zep-papers/issues/5)
- [Atlan "Zep vs Mem0"](https://atlan.com/know/zep-vs-mem0/)

Graphiti (the open-source engine under Zep) does not publish a
separate LoCoMo number. GitHub stars crossed 20k in April 2026.

---

## Cognee

Cognee publishes its own cross-system evaluation — convenient, but
runs on corpora Cognee chose. Main public numbers from
[Cognee AI Memory Benchmarking, Aug 2025](https://medium.com/@cognee/cognee-ai-memory-benchmarking-cognee-lightrag-graphiti-mem0-80b9f62cff36):

| Benchmark | Cognee self-reported |
|-----------|----------------------|
| HotPotQA (multi-hop) | **0.93** (task: exact answer match) |
| TwoWikiMultiHop | (reported in same post, head-to-head with LightRAG, Graphiti, Mem0) |
| MuSiQue | (reported, same post) |

Enterprise traction: **70+ production customers** including Bayer
and University of Wyoming ([Cognee $7.5 M seed, 2025](https://www.cognee.ai/blog/cognee-news/cognee-raises-seven-million-five-hundred-thousand-dollars-seed)).

---

## HippoRAG2

Paper: ["From RAG to Memory: Non-Parametric Continual Learning for
Large Language Models"](https://arxiv.org/html/2502.14802v1) — ICML 2025.
This is the academic baseline — no commercial SaaS, strong research
reputation.

| Benchmark | Metric | HippoRAG2 |
|-----------|--------|-----------|
| MuSiQue | F1 | **51.9** (vs. NV-Embed-v2 + LLM baseline 44.8) |
| MuSiQue | Recall@5 | **74.7 %** (baseline 69.7 %) |
| 2Wiki | Recall@5 | **90.4 %** (baseline 76.5 %) |
| HotpotQA | String accuracy | **56.7 %** |
| MuSiQue | String accuracy | **27.0 %** |

Also: HippoRAG2 claims a later arXiv cross-measurement, EcphoryRAG
(Oct 2025), improves average Exact Match from 0.392 → 0.474 over
HippoRAG across 2Wiki / HotpotQA / MuSiQue.

This is the paper Synaptic's PPR component is descended from — a
*stronger* comparison for us than Mem0, because HippoRAG2 is
trying to solve the same problem (multi-hop retrieval over
documents), not conversational memory.

---

## LightRAG

Paper: ["LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://arxiv.org/html/2410.05779v1)
— EMNLP 2025.

LightRAG reports **win rates against baselines** on UltraDomain
(428-textbook corpus):

| Win-rate vs. | Agriculture | Legal |
|--------------|-------------|-------|
| NaiveRAG | 66.70 % (orig) → 39.06 % (unbiased re-eval) | — |
| MGRAG | 56.38 % → 32.33 % | — |
| Overall retrieval accuracy | 80 %+ | vs. 60-70 % for baselines |

Efficiency (vs. GraphRAG on same corpus):

| Metric | LightRAG | GraphRAG |
|--------|----------|----------|
| Query latency | ~80 ms | ~120 ms |
| Tokens / query | 100 | 610,000 |

**Note on "unbiased re-eval":** a 2026 paper ([How Significant Are
the Real Performance Gains?](https://arxiv.org/html/2506.06331v1))
re-ran LightRAG and several competitors under consistent conditions.
LightRAG's headline win rates were **about half** of the originals
under the independent protocol. Same pattern as Zep.

---

## Microsoft GraphRAG

MS GraphRAG does not publish a single headline number — it publishes
use-case metrics:

- Fortune 500 manufacturer: MTTR 3.2 h → 1.7 h (**47 % reduction**)
- Healthcare partner: diagnostic accuracy **+18 %**

These are production-deployment numbers, not IR benchmarks. Not
directly comparable.

Indexing cost (all reports): LLM tokens are the dominant cost.
GraphRAG spends **~610 k tokens / query-corpus** per LightRAG's
comparison above — the chief motivation for the whole "LLM-free
indexing" direction Synaptic occupies.

---

## Letta (formerly MemGPT)

Letta's focus is **agent framework** more than retrieval, so the
numbers are different in kind:

| Metric | Letta |
|--------|-------|
| LoCoMo (GPT-4o mini, self-reported) | **74 %** |
| ARR (Jun 2025, [Latka](https://getlatka.com/companies/letta.com)) | $1.4 M |
| Seed funding (Sep 2024) | $10 M at $70 M post |

Letta is more naturally a consumer of retrieval than a competitor to
it — in principle it could mount Synaptic as its retrieval layer.

---

## What we can't cleanly compare

| System | Problem |
|--------|---------|
| Mem0 | Mostly measured on LoCoMo (conversational). IR-style MRR/Recall on HotPotQA is not reported. |
| Zep | Numbers disputed, same benchmark, same week. |
| LightRAG | Uses LLM-as-judge "win rate" not MRR. Third-party re-eval halves those numbers. |
| MS GraphRAG | Production-deployment metrics, not benchmark numbers. |
| HippoRAG2 | Uses F1 and string accuracy, not MRR. Closer to comparable but still a gap. |
| Cognee | Own-curated eval, no standard benchmark shared with the others. |

This is why we built the harness in
[`examples/benchmark_vs_competitors/`](../../examples/benchmark_vs_competitors/).
Standard corpus (Allganize RAG-ko, HotPotQA), standard metric (MRR /
Recall@k), same scoring code for everyone, same seed.

---

## How to read this page

- **Do not** cite these numbers against each other. Different
  corpora, different metrics.
- **Do** cite Synaptic's own reproducible numbers alongside these
  when positioning publicly — the contrast is the point. Our numbers
  are in [`eval/results/`](../../eval/results/) and in
  [`examples/benchmark_allganize.py`](../../examples/benchmark_allganize.py)
  (which re-generates them from scratch in under two seconds).
- **Do** re-run competitors in the harness whenever we make a
  positioning claim. Self-reported numbers drift.

Sources for every row are linked inline. If a claim isn't linked, it
was derived from one of the already-cited sources.
