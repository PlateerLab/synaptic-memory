# Synaptic Memory v0.16.0 — release notes

**Release date**: 2026-04-17 · **License**: MIT · **Install**:
`pip install "synaptic-memory[sqlite,korean,vector,mcp]"`

---

## TL;DR

v0.16.0 is the cleanup release that makes Synaptic's benchmark
numbers match what the SDK actually does. Five changes:

1. **`graph.search()` now defaults to `engine="evidence"`** — the
   hybrid BM25 + HNSW + PPR + MMR pipeline that the MCP tool path
   already used. Legacy HybridSearch is deprecated and will be
   removed in v0.17.0.
2. **CDC sync drops 3N round-trips per table to one batch SELECT** —
   `HashTableSyncer` and `TimestampTableSyncer` now issue a single
   `get_pk_index_batch(...)` query instead of three per-row awaits.
3. **Concurrent `_ensure_graph()` is race-safe** — MCP first-turn
   tool dispatches no longer risk constructing two SynapticGraph
   instances.
4. **Korean query-time morphological stripping** (from v0.15.1)
   stays in.
5. **Evaluation coverage grows 30×** — 715 queries across 5
   Korean-heavy corpora → 22,400 queries across 8 corpora that
   now include the standard English multi-hop benchmarks
   (HotPotQA-dev, MuSiQue-Ans-dev, 2WikiMultihopQA-dev).

The net effect on the embedder-free baseline is large:

| Korean benchmark | v0.15.0 | **v0.16.0** | Δ |
|------------------|---------|-------------|---|
| Allganize RAG-ko | 0.621 | **0.947** | +0.326 |
| Allganize RAG-Eval | 0.615 | **0.911** | +0.296 |
| AutoRAG KO | 0.592 | **0.906** | +0.314 |
| PublicHealthQA KO | 0.318 | **0.546** | +0.228 |

English HotPotQA-24 rose 0.727 → 0.875 on the same pipeline.

---

## What shipped

### 1. Engine default: `"evidence"`

Public API: `SynapticGraph.search(query, ...)` — the `engine` kwarg
defaults to `"evidence"`.

**Why now.** v0.14.x–v0.15.x carried `engine="legacy"` as the
default for call-site stability. Meanwhile every benchmark in the
README and the draft arXiv preprint was measured through
EvidenceSearch via the MCP tool path. Keeping SDK callers on the
legacy cascade meant our own documented numbers didn't reflect what
a first-time user of `graph.search()` saw. This release closes that
gap.

**Migration.** Callers that depended on legacy-specific behaviour
— `query_decomposer` hook, `LLMReranker` / `NoOpReranker` injection,
reinforcement-boosted ranking, or the specific `stages_used`
shape — must pass `engine="legacy"` explicitly. A
`DeprecationWarning` fires; the path is scheduled for removal in
v0.17.0. See `tests/test_search_engine_param.py`,
`tests/test_reranker.py`, `tests/test_query_decomposer.py`,
`tests/qa/test_edge_cases.py::TestReinforcementRanking`,
`tests/qa/test_search_quality.py::test_resonance_ordering` for the
six places in-tree that opt-in.

### 2. CDC sync: N+1 → 1 batch SELECT

[`src/synaptic/extensions/cdc/state.py`](../src/synaptic/extensions/cdc/state.py)
grows a `SyncStateStore.get_pk_index_batch(source_url, table, pks)`
method that returns `{pk: (node_id, row_hash, fk_edges_json)}` for
every changed PK in one SQLite query (chunked to 500 host variables
per statement). [`sync.py`](../src/synaptic/extensions/cdc/sync.py)'s
`HashTableSyncer.sync_table()` and `TimestampTableSyncer.sync_table()`
now call this once per table instead of 3 × N times per row.

**Impact.** 100 rows: ~300 ms → ~1 ms of state lookups. 10 k rows:
30 s → under 100 ms.

### 3. Concurrent `_ensure_graph()` race

[`src/synaptic/mcp/server.py`](../src/synaptic/mcp/server.py)'s
lazy initialisation path now holds an `asyncio.Lock` with a
double-checked fast path. Previously, two tool invocations firing
on the same first turn could both see `_graph is None` and
construct two SynapticGraph instances, leaking a backend
connection. Fast path (graph already set) still takes no lock.

### 4. Korean query-time morphological stripping

(Carried from v0.15.1, rebased onto v0.16.0.)
`_normalize_korean(text, query_mode=True)` drops Kiwi-surviving
interrogative / copular noise forms (`무엇`, `어떻`, `대해`,
`설명`, …) at search time only. Index-time normalisation is
unchanged, so existing graph files are not invalidated. Kiwi only
fires when the query is ≥50 % Hangul, which is why English and
code queries are mathematically unchanged.

Net effect on Allganize RAG-ko alone:
0.621 → 0.743 (Kiwi in v0.15.1) → 0.947 (engine flip in v0.16.0).

### 5. Evaluation coverage: 715 → 22,400 queries

New scripts:

* `examples/ablation/download_benchmarks.py` — pulls HotPotQA-dev
  (distractor), MuSiQue-Ans-dev, and 2WikiMultihopQA-dev from
  HuggingFace and converts them to the BEIR-style JSON that
  `run_ablation.py` already consumes.
* `examples/ablation/run_tier1_benchmarks.py` — standard IR
  metrics (MRR, R@5, R@10, Hit@10) for the three English
  multi-hop corpora.

Initial 500-query subset numbers (embedder-free, 2026-04-17):

| Dataset | Docs | MRR@10 | R@5 | Hit@10 |
|---------|-----:|-------:|----:|-------:|
| HotPotQA dev (distractor) | 66,635 | **0.784** | 0.585 | 91.8 % |
| 2WikiMultihopQA dev | 56,687 | **0.795** | 0.501 | 91.2 % |
| MuSiQue-Ans dev | 21,100 | 0.590 | 0.379 | 76.2 % |

MuSiQue is the honest outlier — R@5 0.379 vs HippoRAG2's 0.747.
2-4 hop chains that share no lexical overlap are exactly where an
embedder-free PPR pipeline underperforms an LLM-extracted entity
graph. Embedder path is tracked as v0.16.1.

### 6. Streaming invariance sharpens

Re-running `examples/ablation/streaming_experiment.py` on the
new default:

| Metric | v0.15.x (legacy engine) | v0.16.0 (evidence engine) |
|--------|-------------------------|---------------------------|
| Bit-wise identical top-10 | 51.5 % | **96.0 %** |
| Top-1 identical | 54.5 % | **100 %** |
| \|Δ MRR\| (batch vs streaming) | 0.0100 | **0.0000** |

The Top-K Set Invariance theorem in
[`docs/paper/theorem.md`](paper/theorem.md) holds more tightly.

---

## Deprecations

* `graph.search(engine="legacy")` — `DeprecationWarning`, removal
  scheduled for v0.17.0.
* `SynapticGraph(reranker=LLMReranker(...) | NoOpReranker(...))` —
  honoured only on the legacy engine; removal with legacy.
* `SynapticGraph(query_decomposer=...)` — same.

---

## Upgrade notes

### You called `graph.search(query)` with no kwargs

Before: ran the HybridSearch cascade.
After: runs the EvidenceSearch pipeline.

Retrieval numbers should **improve** on Korean and English alike.
If you specifically depend on the legacy shape
(`stages_used == "synonym"` branches, resonance-descending
strict order), pin it explicitly: `graph.search(query, engine="legacy")`.

### You called `sync_from_database(dsn)` on a large CDC graph

No API change. Under the hood the sync is several orders of
magnitude faster on tables with many updated rows (10 k row
test: 30 s → <100 ms). `tests/test_cdc_search_regression.py`
continues to guard that `mode="cdc"` produces the same top-k as
`mode="full"`.

### You depend on the Mem0-style `Reranker`/`QueryDecomposer` hooks

These remain available under `engine="legacy"` but are not wired
into EvidenceSearch. A migration story for query decomposition
through EvidenceSearch is planned (the `compare_search` compound
tool covers the main use cases today).

---

## Release checklist

- [x] Version bump in `pyproject.toml`, `src/synaptic/__init__.py`,
      `src/synaptic/mcp/__init__.py`.
- [x] CHANGELOG entry under `## [0.16.0] - 2026-04-17`.
- [x] README.md, README.ko.md, docs/comparison/synaptic_results.md
      updated with v0.16.0 numbers.
- [x] docs/paper/draft.md + theorem.md updated with v0.16.0
      streaming invariance and English benchmark numbers.
- [x] Figures regenerated
      (`ablation_bar.py`, `streaming_invariance.py`,
      `tier1_benchmark.py`).
- [x] 819 unit tests pass on Python 3.14 + macOS (local);
      `tests/test_cdc_search_regression.py` green.
- [ ] `uv build && uv publish` — **requires PyPI token**.
- [ ] GitHub release (`gh release create v0.16.0 --generate-notes`).
- [ ] Announce on r/LocalLLaMA, GeekNews, X.

---

## Next up — roadmap

* **v0.16.1** (2026 Q2) — PPR first-hit O(corpus) optimisation so
  HotPotQA-dev full (7,405 q) runs in under 15 min, plus an
  embedder rerun that closes the MuSiQue / 2Wiki gaps.
* **v0.17.0** — legacy HybridSearch removal; `extensions/` →
  `core/` rename with backward-compat re-exports.
* **v0.18.0** — observability (`search(explain=True)`,
  structured logs, Prometheus / OpenTelemetry metrics).

See the full list at [docs/ROADMAP.md](ROADMAP.md).
