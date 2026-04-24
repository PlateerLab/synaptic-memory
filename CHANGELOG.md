# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Measured — v0.18-β1 breakthrough on agent benchmarks (temp=0, Qwen3.5-27B vLLM)

Before / after of the v0.18-β1 + β2 shipwork, measured on the local
`Qwen3.5-27b` vLLM endpoint with `temperature=0` + `seed=42` so the gap
is pure code-effect (zero sampling variance). Three datasets run to
establish generalization:

| Benchmark | Baseline | v0.18-β1 | Δ |
|---|---:|---:|---:|
| assort Hard (structured, 33q) | 26 / 33 = 79 % | **28 / 33 = 85 %** | **+2 queries, +6 pp, -19 % runtime (2105 → 1709 s)** |
| KRRA Hard (text-docs, 39q) | 30 / 39 = 77 % | **32 / 39 = 82 %** | **+2 queries, +5 pp** |
| assort Conv (structured + conversational, 24q) | 22 / 24 = 92 % | 22 / 24 = 92 % | ±0 (no regression) |
| X2BEE Conv (mixed EN/KO conversational, 27q) | 25 / 27 = 93 % (temp=1) | 24 / 27 = 89 % | −1 query (see caveat below) |
| **Combined (123 queries, 4 benches)** | **103 / 123 = 84 %** | **106 / 123 = 86 %** | **+3 queries, +2.4 pp net** |

*X2BEE Conv caveat:* the prior 25 / 27 number was measured at
`temperature=1.0`. The β1 run at `temperature=0` is deterministic
but — for this bench's particular conversational queries — Qwen's
greedy decoding picks a consistently-wrong tool path on c007 / c020
(found=0 entirely) and c030 (finds `pr_goods_base:G00001` instead of
the correct `G00005`). Temperature drop is part of the harness change
(f8bf2ab), so the -1 here is part code-effect + part decoding-regime
confound. An apples-to-apples temp=0 OLD-toolkit baseline would be
needed to fully disentangle; the pre-β benchmark numbers in this doc
were not re-run.

Generalization: both a structured-data bench (where `top_nodes`
directly targets the failing query pattern) and a text-document bench
(where the wins come from 0-result recovery hints + multi-tool batching
prompt + error-envelope unification) show concurrent improvements.
The per-bench gain is small in absolute numbers but tight — temp=0 +
fixed seed collapses the ±2-query sampling variance that previously
obscured code effects, and every improved query traces to a specific
β-track change.

Per-query shift on assort Hard (T0-OLD → T0-β1):

- **a003** "가장 많이 팔린 상품의 리뷰" — miss → **hit** (agent picked
  `top_nodes(products, cumulative_sales, desc, 1)` instead of the old
  `aggregate_nodes(group_by=pk, metric="max")` hack that the previous
  toolkit forced)
- **a039** "최근 많이 팔린 + 핏만족도 높은" — miss → **hit**
- **a010** review-related — miss → hit
- a016 — hit → miss (one-query regression attributable to the prompt
  example-set shift, not a correctness issue — the GT has 8 product
  rows and partial matches)

Runtime drop (2105 → 1709 s on assort Hard) reflects fewer tool turns:
top-N ranking collapses from a 2-3-call composition to a 1-call
primitive, and the hint-following prompt cuts re-issue loops on tools
that return 0.

KRRA Hard remaining 7 misses (h012, h019, h020, h025, h029, h030,
h040) all share the same structural pattern: broad topical queries
("이용자보호 제도", "인권영향평가") where GT has 10-12 specific
documents and the agent's retrieval surfaces phrase hubs / related
terms instead. This is the retrieval-ceiling pattern documented under
α1-2 — not addressable at the agent-tool layer.

### Added — v0.18-β2: `top_nodes` — single-call top-N ranking primitive

New structured tool ``top_nodes(table, sort_by, order, limit, where_*)``
that returns the top-N rows of a table ordered by a column in a single
call. Closes a reliability gap on multi-hop agent benchmarks.

Why this matters:

  - Questions like "가장 많이 팔린 상품의 리뷰" (assort Hard a003),
    "최근 가장 많이 팔린 + 핏만족도 높은" (a039), "방송 횟수가 가장
    많았던 상품의 색상별 판매" (a040) all start with a top-N ranking.
  - Previously the agent had to compose
    ``aggregate_nodes(group_by=<pk>, metric="max", metric_property=<col>)``
    and then extract ``groups[0].node_title`` — a pattern Qwen3.5-27B
    mis-uses frequently (the three benchmarks above all fail this way
    on the measured baseline).
  - ``top_nodes`` is a direct primitive: ``results[0].title`` is the
    answer, and each row carries ``sort_value`` + properties ready to
    chain into ``join_related`` / ``get_document``.

Wiring:

  - ``src/synaptic/agent_tools_structured.py`` — ``top_nodes_tool``
    (list_nodes scan + sort, with the same ``where_*`` pre-filter
    semantics as ``filter_nodes``).
  - ``src/synaptic/agent_loop.py`` — AGENT_TOOLS entry + dispatcher
    + prompt guidance.
  - ``eval/run_all.py`` — AGENT_TOOLS entry + dispatcher + prompt
    guidance + worked examples for "가장 X한", "최근 Y 1위",
    "할인율 가장 높은 25SS 3개".
  - ``src/synaptic/mcp/server.py`` — registered as the
    ``knowledge_top_nodes`` MCP tool.

0-result path emits hints: missing-column (verify via filter_nodes
listing), over-strict WHERE (retry without the pre-filter). 7 new
unit tests cover desc/asc, pre-filter, missing column, strict where,
invalid order, budget exhaustion. 891/891 existing tests still green.

The agent prompt also now explicitly preferrrs parallel tool calls
within a single turn — measured Qwen behaviour is one-tool-per-turn
by default, which wastes context budget on compound questions.

### Added — v0.18-β1: 0-result recovery hints for structured tools

`filter_nodes`, `aggregate_nodes`, and `join_related` now emit
`Hint` entries when they return 0 matches. The hints carry one
concrete corrective action each:

- `filter_nodes(op="==")` → suggest `op="contains"`
- `filter_nodes(op="contains", multi-word value)` → suggest the first
  keyword alone
- `filter_nodes` generic 0-match → suggest `search(value)` as fallback
- `aggregate_nodes` with a `where_*` pre-filter and 0 groups →
  suggest retrying without the pre-filter
- `aggregate_nodes` → suggest `filter_nodes` to verify the
  `group_by` column name
- `join_related` with 0 rows → suggest `filter_nodes` on the target
  table to verify the FK column

The hints surface through `project_tool_result`, and a single new
line in `AGENT_SYSTEM` instructs the agent to follow the first hint
before reissuing near-identical queries. The behaviour was added in
response to observed retry-loops on Qwen3.5-27B benchmarks.

6 new tests in `tests/test_agent_tools_hints.py`; the matching
prompt line is in both `src/synaptic/agent_loop.py` and
`eval/run_all.py` (the two `AGENT_SYSTEM` copies now carry the
same α1-2 relative-time + multi-source guidance as well).

### Fixed — v0.18-C1: CDC schema-drift detection (silent-corruption P1)

`syn_cdc_state.schema_fingerprint` has been stored since v0.14.0 but
never compared. An `ALTER TABLE` on the source DB (column add / rename /
drop) slipped through: the watermark/hash state made every row look
"already synced" under the old shape, so the new column silently
vanished from the graph.

Now both `TimestampTableSyncer.sync_table()` and
`HashTableSyncer.sync_table()` compute the current fingerprint at sync
start, compare to `prior_state.schema_fingerprint`, and on mismatch:

1. Wipe the table's `syn_cdc_pk_index` rows (so stale hashes / FK
   snapshots don't bleed in).
2. Delete the `syn_cdc_state` row.
3. Set `TableSyncStats.schema_changed = True` for observability.
4. Fall through to the existing initial-load path — every row is
   re-ingested under the new schema with stable deterministic IDs.

Legacy state rows that pre-date the fingerprint (empty string) are
treated as "unknown, no reload" so upgrading Synaptic on an existing
graph is a no-op.

4 new tests — 3 in `test_cdc_sync_timestamp.py` (drift detected,
unchanged schema not flagged, empty legacy fingerprint skipped) and
1 in `test_cdc_sync_hash.py` (hash-mode parity).

### Fixed — v0.18-α1-4: agent-loop tool-result context overflow

Replaced naive `json.dumps(result, ensure_ascii=False)[:5000]` truncation at
the agent-loop → tool-result message boundary with a structured projection
(`synaptic.agent_loop.project_tool_result`). The old slice frequently chopped
mid-value, producing invalid JSON that confused the tool-calling agent and
triggered retry loops; combined with accumulating verbose payloads across
turns, this caused roughly 10/172 = 5.8 % of agent queries to exceed vLLM's
16k `max_model_len`.

New behaviour:

- Per-tool projection trims the heaviest fields — preview → 120 chars,
  property values → top-8 scalars × 80 chars, chunk text → 300 chars,
  `evidence[].snippet` → 180 chars — while keeping every `id` / `title` /
  `doc_id` the agent needs for chaining. `_extract_ids` still runs on the
  raw (pre-projection) result, so ID collection is unchanged.
- Default budget is now 4 000 chars (~1 000 tokens). Oversize results
  trigger iterative list-halving instead of mid-JSON truncation, marking
  the result `_trimmed_for_context: true` so the agent knows it saw a
  sample.
- Last-resort stub `{"tool": ..., "ok": ..., "data": {"_overflow": true},
  "error": "tool_result_exceeded_context_budget"}` — always valid JSON.

14 new unit tests in `tests/test_agent_loop_projection.py` lock the shape
and the per-budget size guarantee.

### Added — v0.18-α2: Auto graph snapshot (Graphify G1 absorption)

New `synaptic.snapshot` module + `synaptic-snapshot` CLI + `knowledge_snapshot`
MCP tool + opt-in priming inside `SynapticGraph.chat()`. Generates a markdown
summary of a graph (scale, categories, top phrase hubs, structured tables, edge
kinds, sample query hints) so an LLM agent can skip the cold-start exploration
turns. Measured 0.85 s on KRRA (720 docs / 18.6k chunks / 70k entities). All
stats are direct backend reads — no LLM calls; preserves the LLM-free
indexing principle. `chat(prime_with_snapshot=True)` is the default and the
priming is appended to `extra_context`. 11 new unit tests, all green.

This is the only Graphify (`safishamsi/graphify`) absorption item PLAN-v0.18
green-lit (G2-G5 declined as Neo4j/GraphRAG-derivative or out of scope).

### Changed — `agent_loop` system prompt: relative-time + multi-source guidance

Two new tip lines in the agent prompt, learned from the v0.18-α1-2 KRRA
Conv diagnostic:

- **Relative time references** ("올해" / "내년도" / "this year"): the agent
  should NOT inject a literal year number. The corpus may span multiple
  years and a hard `2024` filter throws away matches. Search the topic
  first, narrow by year only after evidence the user wants one.
- **"X 관련 자료/내용/정보" type questions** ask for *multiple* sources.
  The agent should not stop after the first ``deep_search`` returns 1-2
  docs — at least one paraphrase pass before concluding.

These guidance lines are general-purpose and apply to any corpus, not
just KRRA Conv. The KRRA Conv −23pp regression itself is documented as
known issue: it stems from a recall ceiling on broad topical queries
where 5 GT docs share a vague topic word (예산 / 인권), not from agent
reasoning. Real fix would require either a higher deep_search top-K cap
or reranker-on-by-default for broad queries — both v0.19+ items.

## [0.17.2] - 2026-04-19

Patch release bundling the license switch and the
`SynapticGraph.chat()` agent-loop ID-extraction fix. No public-API
breakage; downstream code that imported from `synaptic.agent_loop`
gains correct `found_ids` population — previously empty for tool
results that came back wrapped (i.e. all of them) and for
structured-only corpora where the answer was an aggregate group
key.

### Changed — License: MIT → Apache-2.0

Project license switched from MIT to Apache-2.0 for the next release. Both
licenses are permissive and allow commercial use with attribution; Apache-2.0
adds an explicit patent grant + termination clause that gives downstream
adopters (especially enterprises) clearer protection. All v0.17.x releases
remain MIT-licensed; the Apache-2.0 grant applies from the next published
version onward.

### Fixed — `synaptic.agent_loop` ID extraction

`run_agent_loop` and `SynapticGraph.chat()` were passing the raw tool wrapper
(`{"tool": ..., "data": {...}}`) to ``_extract_ids`` instead of the unwrapped
``data`` dict, so ``found_ids`` stayed empty even when tools returned valid
evidence. Also added aggregate-group extraction (group value + synthesised
``table:value`` composites) so structured-only corpora like assort Hard, where
the answer IS the group key, score correctly. Tool schemas relaxed to match
``eval/run_all.py`` (``filter_nodes.table`` and ``aggregate_nodes.metric`` no
longer required; ``aggregate_nodes.group_by_format`` accepted for date
bucketing).

## [0.17.1] - 2026-04-19 (PyPI published 22:16 KST)

v0.17.1 is the **kind-aware pipeline release**. v0.17.0's measurement
revealed that uniform pipeline application broke structured-data
corpora (assort, X2BEE Conv) and that the cross-encoder reranker
hurt retrieval-style corpora (AutoRAG −15 %) even after the
blend tune. v0.17.1 introduces three measured-and-validated
mechanisms that, together, push **mean Full-pipeline MRR above
mean FTS-only MRR for the first time** (0.647 vs 0.615, +5.2 %).

### Changed — `EvidenceAggregator` kind-aware split

Candidates whose node carries a ``_table_name`` property — the rows
materialised by ``table_ingester`` / ``db_ingester`` — bypass MMR /
per-document cap / category coverage. Those three mechanisms assume
a chunk-style passage hierarchy and actively dilute the gold rank
on entity-only corpora (assort: structured rows, X2BEE: PostgreSQL
tables). Passage-style nodes (CHUNK / CONCEPT / plain ENTITY) keep
the existing aggregator behaviour. Unit tests cover both code paths.

### Changed — cross-encoder reranker skips structured rows

``EvidenceSearch`` no longer sends ``_table_name``-tagged candidates
to the cross-encoder. Component isolation in
``examples/ablation/diagnose_autorag.py`` showed the cross-encoder
is the dominant failure cause on structured rows: bge-reranker-v2-m3
was trained on long-form sentence pairs and produces near-uniform
logits on short structured content, which override FTS's
near-optimal ranking when blended.

### Added — adaptive cross-encoder blend (variance-gated)

The cross-encoder's blend coefficient now scales with its own
discrimination strength: ``effective_blend = base × min(1, std/3)``
where ``std`` is over the reranker's logits for the top-N
candidates. AutoRAG queries (std ≈ 0.3) get near-zero blend, so
FTS rank dominates. PublicHealthQA queries (std ≈ 4) get the full
blend and keep their +34 % paraphrase win. Threshold 3.0 chosen
from per-corpus diagnostics.

A rank-fusion (RRF) alternative was tried in Round 5 and measured
strictly worse (mean MRR 0.637 vs 0.647) — discretising scores to
ranks discards the magnitude signal that small reorders rely on.

### Added — `DomainProfile.table_query_hints` + targeted FTS augmentation

`DomainProfile` gains a ``[table_query_hints]`` section mapping
table names to query keywords. When a hint fires AND the target
table has fewer than 3 hits in the FTS top-30 (gate against
dominant-table dilution), `EvidenceSearch` runs a targeted
``"{table_name} {query}"`` re-FTS and surfaces matching rows at
score 0.96 (just past the rank-0 FTS floor of 0.95). Fixes
assort q008 ("55 사이즈" → ``sizes:2`` was at FTS rank 5) and
q012 ("LBL코리아 판매 파트너" → ``sales_partners:2`` was outside
FTS top-30). The gate prevents X2BEE-style regressions where a
single dominant table (pr_goods_base) would just dilute the gold
rank if every product hit got boosted.

`assort.toml` populated with hints for the 9 assort tables.
`x2bee.toml` deliberately not added — empirically net-negative
on dominant-table corpora.

### Changed — `DomainProfile.from_dict` lenient on unknown NodeKind

Profiles often outlive enum renames. The previous loader raised
``ValueError`` and refused the whole config; v0.17.1 warns and
skips the bad entry, keeping ``stopwords_extra`` /
``table_query_hints`` / etc. usable. ``assort.toml`` had been
rendered unloadable when ``NodeKind.EVENT`` was removed.

### Added — `eval/run_all.py` profile loading + LLM judge model

`run_custom_dataset` now loads ``eval/data/profiles/{corpus}.toml``
when present and threads ``table_query_hints`` into EvidenceSearch.
`_llm_judge` model parameterised so vLLM agent runs use the same
endpoint as the agent itself (was hardcoded ``gpt-4o-mini``).

### Measurement (Round 6, 2026-04-19, 14 benches, bge-m3 + reranker)

  Bench              FTS-only  v0.17.0  v0.17.1   Δ vs v0.17.0
  KRRA Easy          0.967     0.967    0.975     +0.008
  KRRA Hard          0.583     0.593    0.589     -0.004
  KRRA Conv          0.146     0.139    0.166     +0.027
  assort Easy        0.760     0.767    0.856     +0.089
  assort Hard        0.000     0.000    0.000     0
  assort Conv        0.425     0.268    0.472     +0.204
  X2BEE Easy         1.000     1.000    1.000     0
  X2BEE Hard         0.379     0.250    0.368     +0.118
  X2BEE Conv         0.167     0.123    0.164     +0.041
  HotPotQA-24        0.875     0.979    0.979     0
  Allganize RAG-ko   0.947     0.982    0.983     +0.001
  Allganize RAG-Eval 0.911     0.946    0.955     +0.009
  PublicHealthQA     0.547     0.734    0.748     +0.014
  AutoRAG            0.906     0.766    0.806     +0.040
  MEAN               0.615     0.608    0.647     +0.039

12/14 benches improve or hold; X2BEE Hard / Conv still slightly
below FTS-only by 0.011 / 0.003 (single-query noise level);
AutoRAG regression vs FTS narrowed from −0.264 → −0.100 but
remains structural — pass ``reranker=None`` for FAQ-style corpora.

### Agent benchmark (Qwen3.5-27B vLLM, 5 turns, LLM-judge)

  Bench           Single-shot   Agent solved   v0.13 (gpt-4o-mini)
  KRRA Hard       0.589         30/39 (77%)    11/15 (73%)   +4pp
  assort Hard     0.000         30/33 (91%)    13/15 (87%)   +4pp
  X2BEE Hard      0.368         19/19 (100%)   17/19 (89%)   +11pp
  KRRA Conv       0.166         14/30 (47%)    21/30 (70%)   -23pp
  assort Conv     0.472         22/24 (92%)    20/24 (83%)   +9pp
  X2BEE Conv      0.164         25/27 (93%)    22/27 (81%)   +12pp
  MEAN                          140/172 = 81%

5/6 benches beat the v0.13 GPT-4o-mini baseline. Single-shot 0
on assort Hard becomes 91% under agent loop; X2BEE Hard 0.379
becomes 100%. **Agent loop is Synaptic's actual algorithm** —
single-shot is the diagnostic floor. Only KRRA Conv regresses
(suspected Qwen3.5-27B Korean conversational reasoning gap;
v0.18 track). Context overflow (16k vLLM max) caused 10/172
queries to fail (5.8 %) — agent_tools result truncation is a
v0.18 task.

### Documented — v0.18 architecture design

`docs/PLAN-v0.18-architecture.md` catalogues the 5 design questions
v0.17.1 measurements raised (agent default, selective LLM ingest,
adaptive pipeline, hierarchical schema, per-corpus reranker
calibration) and proposes scope per question for v0.18.0+.

### Tests

820 unit tests pass (818 in v0.17.0 + new kind-aware aggregator
tests + lenient loader test + decomposer integration tests).

### Also bundled into the 0.17.1 PyPI wheel (post-tag improvements)

The published wheel includes the following work that landed between
the v0.17.1 release tag (commit 7560e6d) and PyPI upload (commit
16cee92), captured here so version metadata stays honest:

- **`extensions/reranker_llm.LLMReranker`** — listwise rerank via any
  OpenAI-compatible LLM (vLLM, Ollama, Anthropic). Drop-in
  ``RerankerProtocol``. AutoRAG measurement showed it underperforms
  bge-reranker on FAQ-style corpora (0.793 vs 0.806 MRR, 19× slower)
  — kept as opt-in for users with different corpus characteristics.
- **`extensions/embedder_hyde.HyDEEmbedder`** — wraps any
  ``EmbeddingProvider`` so query-side ``embed`` first asks an LLM for
  a hypothetical answer, then embeds query + hypothetical. KRRA Conv
  measurement showed no benefit on Korean regulatory terminology (HyDE
  output diverged from the corpus's specific phrasing) — kept as
  opt-in for paraphrase-heavy English corpora where the original HyDE
  paper saw +5-10pp.
- **`EvidenceAggregator` MMR-preservation fix** — kind-aware split
  was unconditionally re-sorting the merged evidence by raw score,
  which silently destroyed the MMR-derived order returned by the
  passage aggregator (KRRA Hard FTS-only collapsed 0.583 → 0.518 in
  the v0.18-prep baseline before the fix). Now: preserve aggregator
  ordering when only one kind contributes; re-sort only on cross-kind
  merge.
- **`eval/run_all.py` SqliteGraphBackend swap** — public benchmark
  runner switched from MemoryBackend (Python-loop FTS) to
  SqliteGraphBackend (FTS5, C-implemented). 5× speedup on 2Wiki-dev
  (56k docs × 12k queries: ~7h → ~75min) plus +0.01-0.12 MRR uplift
  from FTS5's tighter BM25.
- **22-bench v0.17.1 baseline locked** — added 5 new public benches
  (TREC-COVID, FiQA, SciFact + wired-up 2Wiki-dev / MuSiQue-dev) for
  v0.18 generality verification. Mean FTS-only MRR 0.650 across 22
  benches in 6 distinct domains. See `eval/baselines/qa_latest.json`.

---

## [0.17.0] - 2026-04-19

v0.17.0 is a **measurement-driven tuning release**. The headline change
is a single constant — `EvidenceSearch.rerank_blend` — moving from 0.4
to 0.1 after a three-round triangulation exposed a 29 %-point MRR
regression on retrieval-style corpora that was hidden by the v0.16.0
engine flip. Two new opt-in flags (`--local-bge`, `--entity-linker`)
and a `QueryDecomposer` Protocol round out the release. The honest
summary: **Synaptic's FTS-only pipeline is already state-of-the-art on
Korean long-form corpora; Full pipeline adds ~+5 pp mean MRR but must
be opted out of on retrieval-style corpora.**

### Changed — `rerank_blend` default 0.4 → 0.1

`EvidenceSearch(reranker=...)` now blends the cross-encoder at 10 %
against the hybrid-rank's 90 %, down from 40 % / 60 %. The old blend
maximised paraphrase wins (PublicHealthQA, Allganize) but wrecked
retrieval-style corpora where FTS ranking was already near-optimal.

Evidence across 5 public benches (`bge-m3 + bge-reranker-v2-m3`,
H100 FP16):

| Bench | FTS-only | b=0.1 (new) | b=0.4 (old) |
|-------|---------:|------------:|------------:|
| HotPotQA-24 | 0.875 | 0.979 | 1.000 |
| Allganize RAG-ko | 0.947 | **0.982** | 0.972 |
| Allganize RAG-Eval | 0.911 | **0.946** | 0.925 |
| PublicHealthQA | 0.547 | **0.734** | 0.706 |
| AutoRAG | **0.906** | 0.766 | 0.642 |
| **MEAN** | 0.837 | **0.881** | 0.849 |

Component isolation on AutoRAG (`diagnose_autorag.py`) pinned the
regression squarely on the cross-encoder: FTS-only 0.906 → reranker
alone 0.641 (Hit 114 / 114 → 81 / 114). The embedder path was
near-neutral. Sweep: `examples/ablation/sweep_rerank_blend.py`.

Callers can override per-corpus: `EvidenceSearch(..., rerank_blend=0.2)`.

### Added — `QueryDecomposer` Protocol + `LLMChainDecomposer`

`src/synaptic/protocols.py` gains a `QueryDecomposer` Protocol
(`async decompose(query) -> list[str]`). Two implementations ship:

- Existing `QueryDecomposer` (rule-based, Korean conjunction splits) now
  satisfies the Protocol structurally — zero-change compat.
- New `LLMChainDecomposer` (`extensions/query_decomposer_llm.py`) for
  multi-hop English chain queries via any OpenAI-compatible endpoint.

`EvidenceSearch` gains a `decomposer=` kwarg. When the decomposer
returns ≥2 sub-queries, each sub runs a separate FTS seed retrieval
and the ranks are fused via RRF (`k=60`) before graph expansion and
reranking (which stay on the original query).

**Opt-in, default-off.** The chain decomposer measured
**−10.6 % R@5 on MuSiQue-Ans** (500 q, R@5 0.453 → 0.405) — documented
in `docs/PLAN-v0.17-ontology.md` §9 and `docs/CONCEPTS.md` §13.1. The
mechanism helps compound queries but hurts chain-reasoning benchmarks
because RRF equal-weights sub-query noise against the original query.
Use only when compound splits (Korean "A와 B 비교") dominate your
corpus.

### Added — `--local-bge` end-to-end benchmark runner

`eval/run_all.py --local-bge` loads `BAAI/bge-m3` +
`BAAI/bge-reranker-v2-m3` directly via `transformers` (FP16, cuda:0).
No Ollama endpoint, no TEI container. Same path in
`examples/ablation/run_tier1_benchmarks.py` and
`examples/benchmark_allganize.py`. Model weights load once per suite
run and are shared across all datasets. Requires `torch` and a GPU;
coexists with a running vLLM (tested with Qwen3.5-27B-TP2).

Also adds `--entity-linker` (opt-in post-hoc DF-filtered phrase hub,
`min_df=2, max_df_ratio=0.02`). Effect on public benches was ±1 %
across the board — left opt-in for users whose corpora may benefit.

### Measured and NOT shipping as default

Three mechanisms were implemented, measured, and **kept off**. Each
has a standalone section in `docs/CONCEPTS.md` §13:

- **LLM query decomposition**: −10.6 % on MuSiQue R@5
- **Inline phrase hub (no DF filter)**: −6.6 % on MuSiQue, 15× slower build
- **DF-filtered EntityLinker**: ±1 % on public benches (neutral)

See `docs/PLAN-v0.17-ontology.md` §9 for the full post-mortem of the
"initial +54 % uplift" narrative that collapsed on triangulation (the
baseline was v0.14.4, not current-code FTS-only after v0.15.1 +
v0.16.0).

### Documented — known gap on English multi-hop

MuSiQue-Ans-dev 500q full pipeline hits R@5 **0.453** vs HippoRAG2
published **0.747** (−0.294). Three rounds of targeted fixes
(decomposer, phrase hub variants, entity linker) all regressed the
score — the gap is structural. Closing it requires OpenIE triple
extraction + query→triple dense linking, which is a v0.18.0+ research
track rather than a default pipeline change. Synaptic's strength is
Korean / structured-data RAG; English Wikipedia multi-hop is
honestly documented as a trade-off.

### Unchanged but worth noting

- `graph.search()` default engine remains `"evidence"` (v0.16.0)
- `engine="legacy"` still raises DeprecationWarning; removal pushed to
  v0.18.0 to bundle with HippoRAG2-style architecture work
- Core dependencies remain 0; `torch` is only required when using
  `--local-bge` (benchmark harness opt-in)

---

## [0.16.0] - 2026-04-17

v0.16.0 is a **foundational cleanup release**. Five changes that add
up to a noticeably different product: (1) the retrieval engine
finally defaults to the hybrid EvidenceSearch pipeline the benchmarks
have been advertising, (2) CDC sync drops N round-trips to a single
batch SELECT, (3) concurrent MCP tool calls no longer race on
first-use initialisation, (4) the query-mode Kiwi improvement
from the v0.15.1 branch is carried forward, and (5) the evaluation
surface is expanded 30× — from 715 queries across 5 Korean-heavy
corpora to 22,400 queries across 8 corpora including standard English
multi-hop benchmarks (HotPotQA-dev, MuSiQue-Ans, 2WikiMultihopQA).
The net effect on public Korean benchmarks is large (Allganize RAG-ko
MRR 0.621 → 0.947 since v0.15.0); English multi-hop debuts at
HotPotQA-dev MRR 0.784 / 2Wiki-dev MRR 0.795 (500-query subsets,
embedder-free).

### Changed — `graph.search()` default engine flipped to `"evidence"`

Public API semantics change: `SynapticGraph.search(query, ...)` with
no `engine=` kwarg now uses the :class:`EvidenceSearch` hybrid
pipeline (BM25 + HNSW + PPR + MMR + optional cross-encoder) that
`agent_search` / `knowledge_search` / `deep_search` already used.

Why now. The previous default (``engine="legacy"`` → HybridSearch)
was retained through v0.15.x for caller stability, but every
measured benchmark — including the ones quoted in the README and in
[docs/paper/draft.md](docs/paper/draft.md) — was run through
EvidenceSearch via the MCP tool path. Leaving SDK callers on the
legacy cascade meant our own quoted numbers didn't match what a
first-time user of ``graph.search()`` saw. This release closes that
gap.

Effect on public FTS-only benchmarks (no embedder, no reranker):

| Dataset | v0.15.0 (legacy) | v0.15.1 (legacy + kiwi) | **v0.16.0 (evidence + kiwi)** | Total Δ |
|---------|------------------|-------------------------|-------------------------------|---------|
| Allganize RAG-ko | 0.621 | 0.743 | **0.947** | +0.326 |
| Allganize RAG-Eval | 0.615 | 0.695 | **0.911** | +0.296 |
| PublicHealthQA KO | 0.318 | 0.466 | **0.546** | +0.228 |
| AutoRAG KO | 0.592 | 0.692 | **0.906** | +0.314 |
| HotPotQA-24 EN | 0.727 | 0.727 | **0.875** | +0.148 |

Migration. ``engine="legacy"`` still works and raises
:class:`DeprecationWarning`. Legacy-specific features
(`query_decomposer`, `reranker=LLMReranker(...)` injection, and
reinforcement-based ranking boost) are only honoured on the legacy
path and keep their tests under `engine="legacy"`. Scheduled for
removal in v0.17.0.

### Changed — streaming invariance sharpens under the new default

`examples/ablation/streaming_experiment.py` re-run on the Evidence
pipeline (Allganize RAG-ko, 200 docs / 200 queries, 10 random
streaming batches vs. one batch):

| Metric | v0.15.x (legacy engine) | **v0.16.0 (evidence engine)** |
|--------|-------------------------|-------------------------------|
| Bit-wise identical top-10 | 103 / 200 (51.5 %) | **192 / 200 (96.0 %)** |
| Top-1 identical | 109 / 200 (54.5 %) | **200 / 200 (100 %)** |
| \|Δ MRR\| | 0.0100 | **0.0000** |
| Set-equal top-10 | 197 / 200 (98.5 %) | 197 / 200 (98.5 %) |

The streaming-invariance theorem in [docs/paper/theorem.md](docs/paper/theorem.md)
holds more tightly on the new default.

### Fixed — CDC sync: N+1 round-trips → one batch SELECT

Both `HashTableSyncer.sync_table()` and
`TimestampTableSyncer.sync_table()`
(`src/synaptic/extensions/cdc/sync.py`) previously issued 3 × N
awaits per table (one `get_row_hash` + `get_node_id` + `get_fk_edges`
call per row). They now issue a single
`SyncStateStore.get_pk_index_batch(...)` call that returns
`{pk: (node_id, row_hash, fk_json)}` for every changed PK in one
SQLite round-trip (chunked to 500 PKs per statement defensively).

Impact. For a source table with N changed rows on a 1 ms local
SQLite round-trip, CDC sync latency drops from 3N ms to ~1 ms — so
a 100-row change sync moves from ~300 ms of sequential awaits to a
single query. At 10 k rows the difference is 30 s → <100 ms.

### Fixed — concurrent `_ensure_graph()` no longer races

`src/synaptic/mcp/server.py` — lazy graph initialisation is now
serialised through an `asyncio.Lock` with a double-checked fast path.
Previously, two tool invocations firing on the same first turn could
both see `_graph is None` and construct two SynapticGraph instances,
leaking a backend connection. The fast path (graph already set) still
requires no lock.

### Added — Tier-1 English multi-hop evaluation coverage

`examples/ablation/download_benchmarks.py` pulls HotPotQA-dev
(distractor), MuSiQue-Ans-dev, and 2WikiMultihopQA-dev from
HuggingFace and converts them to the BEIR-style JSON that
`examples/ablation/run_tier1_benchmarks.py` consumes. Every file is
gitignored and regenerated on demand, so the download is a one-shot
per clone. Adds an `[eval]` optional extra with `datasets>=3.0`.

Initial numbers on a 500-query subset (embedder-free,
`engine="evidence"`, 2026-04-17):

| Dataset | Docs | MRR @ 10 | R @ 5 | R @ 10 | Hit @ 10 |
|---------|-----:|---------:|------:|-------:|---------:|
| HotPotQA dev (distractor) | 66,635 | **0.784** | 0.585 | 0.658 | 459/500 |
| MuSiQue-Ans dev | 21,100 | 0.590 | 0.379 | 0.440 | 381/500 |
| 2WikiMultihopQA dev | 56,687 | **0.795** | 0.501 | 0.552 | 456/500 |

Wall clock on a laptop, total 31 minutes. A full-dataset rerun is
scheduled for v0.16.1 after the PPR stage's first-hit latency is
profiled (currently ~1.8 s / query at 66 k docs).

### Deprecated

- `graph.search(engine="legacy")` — emits DeprecationWarning,
  scheduled for removal in v0.17.0.
- `LLMReranker` / `NoOpReranker` injection on `SynapticGraph`
  (legacy-engine-only).
- `query_decomposer` kwarg on `SynapticGraph` (legacy-engine-only).

### Changed — query-mode Kiwi lifts FTS-only Korean retrieval quality

`_normalize_korean(text, query_mode=True)` — at search time only —
drops verb (VV) and adjective (VA) stems and a small set of
interrogative / copular noise forms that survive POS filtering but
degrade BM25 ranking on natural-language Korean queries ("설명해주세요",
"무엇인가요", "어떻게", "대해", …).

Index-time normalisation is **unchanged**. No graph rebuild required.
Kiwi only fires on queries that are ≥50 % Hangul, so English and
code-heavy queries are mathematically unchanged.

Measured on public FTS-only reproducible benchmarks
(`examples/ablation/run_ablation.py`, 2026-04-17):

| Dataset | Pre-0.15.1 MRR | v0.15.1 MRR | Δ | Hit @ 10 |
|---------|----------------|-------------|---|----------|
| Allganize RAG-ko | 0.621 | **0.743** | +0.122 | 200/200 |
| Allganize RAG-Eval | 0.615 | **0.695** | +0.080 | 286/300 |
| PublicHealthQA KO | 0.318 | **0.466** | +0.148 | 65/77 |
| AutoRAG KO | 0.592 | **0.692** | +0.100 | 114/114 |
| HotPotQA-24 EN | 0.727 | 0.727 | 0.000 | 24/24 |

Diagnostic that led to the change:
`examples/ablation/failure_diagnostic.py`. 16 of 20 Allganize RAG-ko
misses were "generic question form" queries whose topic nouns were
drowned out by Kiwi-surviving question tails. Applied the same filter
in `MemoryBackend.search_fts` so benchmark adapters (which all use
MemoryBackend) pick up the improvement without a SQLite rebuild.

> **Note**: the numbers above are the v0.15.1 delta measured against
> the v0.15.0 legacy-engine baseline. v0.16.0's engine flip pushes
> these to 0.947 / 0.911 / 0.546 / 0.906 — see the 0.16.0 entry above.

## [0.15.0] - 2026-04-15

### Added — `graph.search(engine="evidence")` opt-in modern path

Phase C of the v0.14.x cleanup, rescoped from the original
"force-migrate `graph.search()` to EvidenceSearch" plan.

**Why rescoped.** Tracing every `graph.search()` caller turned up
67 sites across `tests/` and `eval/`, plus features (synonym
expansion, query rewriter fallback, resonance-ordering
contracts) that the legacy `HybridSearch` carries and
`EvidenceSearch` does not. A forced migration would silently
break every benchmark and every UI that branches on
``stages_used == "synonym"``. The user pain that motivated
Phase C was the magic ``cos >= 0.45`` cutoff, and that was
already removed in v0.14.1's relative threshold + v0.14.2's
MCP route to EvidenceSearch.

The remaining gap was that **SDK users** (callers of
`graph.search()` directly, not `MCP knowledge_search`) had no
clean path to the modern pipeline without instantiating
`EvidenceSearch` themselves. This release adds that path.

### What changed

- `SynapticGraph.search(query, *, limit=10, embedding=None,
  engine="legacy")` — new keyword-only ``engine`` parameter.
  - ``"legacy"`` (default) → :class:`HybridSearch`. Identical to
    every previous version. Zero behaviour change for the 67
    existing callers.
  - ``"evidence"`` → :class:`EvidenceSearch` via a new
    `_search_via_evidence()` adapter that returns a
    ``SearchResult`` (not ``EvidenceSearchResult``) so legacy
    iteration / sorting / `result.nodes[i].resonance` contracts
    keep working.
  - Anything else raises ``ValueError``.
- The adapter populates ``stages_used = ["evidence", "fts"]``
  (plus ``"vector"`` when an embedder is wired) so consumers
  can detect which engine ran. The legacy stages
  (``"synonym"``, ``"rewriter"``) are intentionally absent on
  the modern path because EvidenceSearch does not have those
  steps; UIs that branch on those names need to handle the
  empty signal.

### Deprecation timeline

| Version | Behaviour |
|---|---|
| **0.15.0** (this) | ``engine="legacy"`` default, ``engine="evidence"`` opt-in |
| **0.16.0** (next minor) | Default flips to ``engine="evidence"``, legacy still available |
| **0.17.0** | Legacy engine removed |

New code should pass ``engine="evidence"`` explicitly today —
it gets the modern pipeline (anchor extraction, hybrid
reranker, MMR aggregation, no magic cutoff) without an SDK
boilerplate construction.

### Tests

`tests/test_search_engine_param.py` (6 new):

- Default engine is ``"legacy"``.
- ``engine="legacy"`` matches the default in node IDs and
  ``stages_used`` (forward-compat switch, not a behaviour change).
- ``engine="evidence"`` returns a ``SearchResult`` with
  ``"evidence"`` in ``stages_used``.
- ``engine="evidence"`` finds the doc with the shared salient
  phrase and excludes the unrelated doc from the top-2.
- Unknown engine name raises ``ValueError``.
- ``engine="evidence"`` preserves descending-resonance ordering.

The existing 54 ``test_search.py`` + ``test_graph.py`` tests
continue to pass unchanged because the default path is the same
HybridSearch they always exercised.

Full suite: 809 passing.

### Implementation notes

- `_search_via_evidence` lazy-imports `EvidenceSearch` so the
  modern pipeline only loads when a caller opts in. Cold-start
  cost for legacy users is unchanged.
- The adapter forwards ``self._embedder``, ``self._phrase_extractor``,
  and ``self._reranker`` into the EvidenceSearch instance, so a
  graph that already has those wired (e.g. via
  ``SynapticGraph.from_data()``) gets the full modern pipeline
  without any extra setup.
- ``Evidence.score`` is mapped to both ``ActivatedNode.activation``
  and ``ActivatedNode.resonance`` so any legacy code that sorts
  by resonance keeps producing the right order.

## [0.14.4] - 2026-04-15

### Added — `graph.backfill()` + `knowledge_backfill` MCP tool

Recovery path for the v0.14.x silent-failure modes that landed
in v0.14.1 and v0.14.3 fixes. Two distinct gaps used to require
a full re-ingest from source to repair:

1. **Empty embeddings.** A graph ingested without an embedder
   stores ``Node.embedding=[]``. Wiring an embedder afterwards
   does not retroactively embed those nodes — the HNSW index
   stays empty and vector search degrades to "FTS only" on the
   affected slice.

2. **Missing phrase hubs.** A graph ingested without a
   ``phrase_extractor`` (the default for the MCP server before
   v0.14.3 — see that release's note) has no cross-document
   bridges, because no chunks ever got linked to shared ENTITY
   phrase hubs via CONTAINS edges. PPR / GraphExpander then
   cannot walk across files.

`graph.backfill()` walks the existing graph in place and repairs
each node where the relevant signal is missing, without touching
nodes that are already healthy. Idempotent — running twice on the
same graph produces zero work on the second pass.

```python
from synaptic import SynapticGraph

graph = await SynapticGraph.from_data("./old_corpus/", embed_url="...")
result = await graph.backfill()
print(result.embeddings_filled, result.phrases_linked, result.elapsed_ms)
```

### MCP tool

- ``knowledge_backfill(scope="all" | "embeddings" | "phrases",
  batch_size=64, max_nodes=None)`` — wraps the graph method.
  Tool count 35 → 36.

### Implementation notes

- Embedding pass batches via ``embedder.embed_batch`` for speed
  (configurable ``batch_size``). Phrase pass is per-node since
  the extractor is already per-passage.
- Both passes are best-effort — a single failing row appends to
  ``BackfillResult.errors`` but never aborts the rest of the run.
- ``max_nodes`` lets you process huge graphs incrementally.
- Skips already-healthy nodes:
  - Embedding pass: ``if node.embedding: continue``
  - Phrase pass: ``if any(e.kind == CONTAINS for e in outgoing): continue``
- Phrase hubs themselves (tagged ``_phrase``) are never re-extracted
  — that would create infinite hubs of hubs.

### Tests

`tests/test_backfill.py` (10 new):

- `TestEmbeddingBackfill` (4): no-op without embedder, fills
  missing embeddings, idempotent on healthy graph, skips
  text-less nodes without crashing.
- `TestPhraseBackfill` (4): no-op without extractor, creates
  bridge after wiring extractor, idempotent on healthy graph,
  skips phrase-hub nodes (no infinite recursion).
- `TestCombinedBackfill` (2): default repairs both, ``max_nodes``
  limit is respected.

Full suite: 803 passing.

### New `BackfillResult` dataclass

```python
@dataclass(slots=True)
class BackfillResult:
    scanned: int = 0
    embeddings_filled: int = 0
    phrases_linked: int = 0
    skipped_no_text: int = 0
    elapsed_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
```

Exported from `synaptic.models`.

## [0.14.3] - 2026-04-15

### Fixed — MCP graph now creates cross-document phrase-hub bridges

**The bug.** Ingesting N files through MCP (`knowledge_add_document`,
`knowledge_ingest_path`, `knowledge_add_chunks`) produced N
disconnected clusters of nodes that shared no edges. Files that
should obviously be related (same proper noun, same topic) had no
graph path between them. PPR / GraphExpander could not surface
cross-document evidence; the search degraded to "FTS over
disjoint files".

**Root cause.** Synaptic implements a HippoRAG2-style dual-node
KG: each chunk has its salient phrases extracted and lifted into
ENTITY *phrase-hub* nodes. Multiple chunks sharing the same
phrase all `CONTAINS`-edge into the same hub, which makes the hub
a bridge between documents. The mechanism is implemented in
`PhraseExtractor.extract_and_link()` and triggered from
`graph.add()` only when a `phrase_extractor` is wired into
`SynapticGraph`.

`SynapticGraph.from_data()` and `SynapticGraph.full()` always
wire one. The MCP server's `_ensure_graph()` factory wired
`ChunkEntityIndex` (the read-side index that PPR uses) but
**forgot the extractor that populates it**. Result: an empty
phrase-hub set, no `CONTAINS` edges, no bridges.

This had been silently degrading every MCP-driven graph since
v0.14.0 added the ingest tools. It only surfaced when a user
inspected the edge topology after ingesting three files and saw
three islands.

**Fix.** `mcp/server.py:_ensure_graph()` now passes
`phrase_extractor=PhraseExtractor()` alongside the existing
`chunk_entity_index=ChunkEntityIndex()`. The boot log line also
gained a `phrase_extractor=on` field so misconfigurations are
visible immediately.

**Tests.** `tests/test_mcp_ingest_tools.py::TestCrossDocumentBridges`
(2 new):

- `test_shared_phrase_creates_bridge_node`: two documents that
  both mention "Synaptic Memory" must share at least one
  phrase-hub `ENTITY` node reached via `CONTAINS` from both.
- `test_disjoint_documents_have_no_bridge`: a pizza recipe and a
  quantum tunneling note must NOT spuriously bridge — the phrase
  hub mechanism is precision-aware.

Full suite: 793 passing.

**Migration note.** Existing graphs created with v0.14.0~v0.14.2
through MCP do not have phrase hubs and need to be re-ingested
to gain cross-document bridges. There is no in-place backfill
yet (related: the embedding-backfill gap noted in the v0.14.x
follow-up plan). Re-ingest from source if you want the bridges.

## [0.14.2] - 2026-04-15

### Changed — MCP `knowledge_search` routes through EvidenceSearch

Phase 2 of the magic-number cleanup started in v0.14.1.

`MCP knowledge_search` previously called `graph.search()`, which
in turn called the legacy `HybridSearch` path. Even with v0.14.1's
relative-threshold fix, that path still treats vector hits as a
*supplement* to FTS via a hardcoded cascade. The deep tail of the
positive distribution on low-cosine embedders (OpenAI v3 small/
large, MiniLM) was still partially lost.

This release wires `knowledge_search` directly to
:class:`EvidenceSearch`, the same engine that already backs
`agent_search`, `agent_deep_search`, `compare_search`, and the
benchmark harness (`eval/run_all.py`). EvidenceSearch:

- Uses **min-max normalised cosine** in its hybrid reranker, so
  absolute cosine values disappear from the decision entirely.
- Has **no threshold cutoff** — vector hits compete on relative
  rank against lexical/graph/structural signals.
- Adds `reason` ("top_score", "category_coverage", "document_quota")
  and `category` fields to each hit so callers can see *why* the
  aggregator picked a node.

The knowledge_search response payload now includes:

- `reason` (new) — aggregator decision tag per hit
- `category` (new) — node category from properties
- `anchors` (new) — ``{categories, entities}`` extracted from query
- `total_candidates` — now reflects the EvidenceSearch reranker
  pool size, not the legacy HybridSearch candidate set
- `search_time_ms` — measured by EvidenceSearch end-to-end

`stages_used` is no longer reported because EvidenceSearch always
runs the full pipeline (FTS → vector → PRF → expand → rerank →
aggregate); there is no per-call branching to surface.

### Tests

`tests/test_mcp_ingest_tools.py::TestKnowledgeSearch` (4 new):
- Lexical query still hits the right document (sanity).
- Response carries the new EvidenceSearch fields (`reason`,
  `category`) — regression guard against accidental revert to
  `graph.search()`.
- Empty corpus returns ``{success: True, results: []}`` with an
  explanatory message.
- Unrelated query (no lexical or semantic overlap) does not put
  an irrelevant doc at the top.

Full suite: 791 passing.

### Migration note

The legacy `HybridSearch` and the v0.14.1 relative-threshold fix
are still in place — they back `graph.search()` directly and
`AgentSearch` (the intent-routed multi-query wrapper). Only
`MCP knowledge_search` moved to EvidenceSearch in this release.

If your code calls `graph.search()` (not the MCP tool), you are
still on the legacy path and the v0.14.1 relative threshold
applies. The legacy path is preserved for back-compat — no plan
to delete it before v0.16.

### Note on benchmark numbers

While running `eval/run_all.py --quick` to validate this release we
discovered that the committed `eval/baselines/qa_latest.json` (last
updated under v0.13.0) is **stale** — the underlying
`eval/data/parsed/krra/chunks.jsonl` was re-parsed on 2026-04-09
and the corpus shape changed. Bisecting back to commit ``d1f229e``
(the baseline source-of-truth) reproduces the *current* MRR
numbers (KRRA Easy 0.450, X2BEE Hard 0.263), confirming there is
**no code regression** between v0.13.0 and v0.14.2.

The numbers in `CLAUDE.md` and the committed baseline JSON should
be treated as historical until they are regenerated against the
current corpus snapshot. v0.14.x search code is behaviourally
identical to v0.13.0 on identical inputs (FTS-only path; the
v0.14.1 relative threshold only fires when an embedder is wired,
which `--quick` mode is not).

## [0.14.1] - 2026-04-15

### Fixed — Embedder-agnostic vector cascade threshold

**Background.** `HybridSearch` (the legacy 3-stage search backing
`graph.search()` and `MCP knowledge_search`) used a hard-coded
``cos >= 0.45`` cutoff on vector-only candidates. The threshold was
tuned in 2026-03-26 against bge-m3-style models where true positives
sit at cosine 0.55+. With OpenAI text-embedding-3-small / 3-large the
cosine distribution is much lower (p50 ≈ 0.40, p75 ≈ 0.48), so the
absolute 0.45 cutoff silently rejected 50–75% of true positives. The
threshold had also never been benchmarked — `eval/run_all.py` always
routes through `EvidenceSearch` when an embedder is wired up, so the
legacy path's tuning rotted unnoticed.

**Fix.** Replaced the absolute cutoff with a *relative* one whose
floor scales with the embedder's natural cosine distribution:

    floor = max(vector_min_cosine, top_cos * (1 - vector_relative_drop))

where `top_cos` is the highest cosine among non-FTS-overlapping
vector candidates. With the defaults (`vector_min_cosine=0.10`,
`vector_relative_drop=0.30`):

| Embedder | Top hit | Effective floor |
|---|---|---|
| bge-m3 / qwen3-embedding-4b | ~0.80 | ~0.56 |
| multilingual-e5 | ~0.85 | ~0.595 |
| **text-embedding-3-small** | **~0.55** | **~0.385** |
| text-embedding-3-large | ~0.62 | ~0.434 |

The same fixture returns the same number of vector candidates on
every embedder family — the cutoff is now embedder-agnostic.

**Override hierarchy.**
1. `HybridSearch(vector_min_cosine=, vector_relative_drop=)`
   constructor parameters
2. `SynapticGraph(vector_min_cosine=, vector_relative_drop=)`
   passthrough
3. `synaptic-mcp --vector-min-cosine 0.10 --vector-relative-drop 0.30`
   CLI flags
4. `SYNAPTIC_VECTOR_MIN_COSINE` / `SYNAPTIC_VECTOR_RELATIVE_DROP`
   environment variables
5. The defaults above

**Tests.** `tests/test_hybrid_search_threshold.py` covers the
override hierarchy and runs the same fixture under both
"bge-shape" (cosines 0.30–0.85) and "openai-shape" (0.20–0.55)
synthetic distributions, asserting that the *count* of vector
candidates that survive the cutoff is identical. Also documents
the legacy bug with a regression test using a 0.44 cosine that
the old hardcoded cutoff would have dropped. Full suite: 787 passing.

**Note for users.** This only changes the legacy `HybridSearch` /
`graph.search()` / `MCP knowledge_search` path. `agent_search`,
`agent_deep_search`, `compare_search`, and the eval bench all use
`EvidenceSearch` which never had this issue (it uses min-max
normalised cosine in the reranker). Phase 2 of this work (next PR)
will migrate `knowledge_search` to use `EvidenceSearch` as well so
the magic number disappears entirely.

## [0.14.0] - 2026-04-14

### Added — Live database CDC (Change Data Capture)

- **`SynapticGraph.from_database(mode="cdc")`** — opt-in deterministic
  node IDs derived from `(source_url, table, primary_key)`. Re-running
  the same source against the same graph file is now a true upsert;
  random-UUID `mode="full"` is preserved as the default for one-shot
  exports.
- **`SynapticGraph.sync_from_database(dsn)`** — incremental sync.
  Tables with an `updated_at`-style column use a `WHERE col >= watermark`
  filter; tables without one fall back to per-row content hashing.
  Both strategies share delete detection (TEMP TABLE + LEFT JOIN, no
  Python set diff) and FK edge re-computation.
- **`mode="auto"`** — uses prior CDC state when the graph file has
  one, otherwise falls back to `mode="full"`.
- **Supported dialects**: SQLite, PostgreSQL, MySQL/MariaDB.
  Oracle / MSSQL still use the legacy full-reload path.
- **Self-contained graph files**: CDC bookkeeping (`syn_cdc_state`,
  `syn_cdc_pk_index`) lives inside the same SQLite file as the graph,
  so a single `.db` round-trips cleanly.
- **Search regression test** locks in that `mode="cdc"` returns the
  same top-k results as `mode="full"` — CDC only changes node
  identification, never search algorithms.
- **Production validation** against X2BEE PostgreSQL (19,843 rows,
  7 tables): initial CDC load 51s, idempotent re-sync **6s** (~6×
  faster than full reload), 4/4 user queries return identical
  top-1 results vs `mode="full"`.

### Added — MCP ingest + CDC tools

Brings knowledge-base maintenance into the MCP tool surface so
Claude (or any MCP client) can ingest and sync from live data
without dropping to a CLI. Tool count 29 → 35.

- **`knowledge_add_document`** — wraps `graph.add_document()` with
  automatic sentence-boundary chunking and the PART_OF /
  NEXT_CHUNK edge scaffolding.
- **`knowledge_add_table`** — wraps `graph.add_table()`: column
  definitions + row list → typed ENTITY nodes, FK edges, and
  auto-registration of the table schema in the ontology.
- **`knowledge_add_chunks`** — BYO-chunker path. Accepts a list of
  `{title, content, tags, source, properties}` dicts for users
  whose upstream tooling (LangChain, Unstructured, custom OCR)
  already produced chunks.
- **`knowledge_ingest_path`** — ingest a single CSV / JSONL / text
  file from the local filesystem into the current graph. Uses
  sync helpers to keep the async tool body free of blocking I/O.
- **`knowledge_remove`** — single-node deletion with edge cascade.
  Bulk removal is intentionally not exposed.
- **`knowledge_sync_from_database`** — CDC incremental sync from
  MCP. First call seeds state, subsequent calls read only changed
  rows. Accepts a per-call `connection_string` or falls back to
  the new `--source-dsn` CLI flag.
- **`--source-dsn` CLI flag** on `synaptic-mcp` for binding a
  default CDC source.
- MCP graph now uses a `ChunkEntityIndex` so `add_document`
  produces nodes of `NodeKind.CHUNK` (required for the PART_OF
  validation path).
- `build_agent_ontology()` gains `document` / `chunk` types and
  the existing `part_of` constraint is widened so chunk → chunk
  edges validate alongside the existing agent_activity → session
  rule. Required because `validate_edge` AND-s across every
  matching constraint; a single permissive rule is the only way
  to express an OR between two legal shapes.

### Fixed — CDC bugs caught by production validation

- **Canonical PK normalization** (`canonical_pk()` in
  `extensions/cdc/ids.py`): the row-read path went through
  `_cast_row` (numeric → `float(1.0)` → `'1.0'`) while the PK-read
  path used raw asyncpg (`Decimal('1')` → `'1'`). The `LEFT JOIN`
  in `find_deleted_pks` therefore matched none of the rows, and
  every sync flapped the table by re-deleting + re-inserting
  every row. Integer-valued floats / Decimals now collapse to a
  single canonical string used everywhere a PK is hashed,
  stored, looked up, or compared.
- **Skip CDC sync for tables without a real primary key**
  (`TableSchema.has_explicit_pk` propagated by every schema
  reader): falling back to `columns[0]` collapses rows whose
  fallback column isn't unique (e.g. an AWS DMS validation
  table with 47 rows but only 1 distinct `TASK_NAME`) into a
  single deterministic node ID, losing 46 rows and producing
  endless update churn. Such tables are now skipped with a
  clear warning + error entry in the `SyncResult`; users can
  still ingest them via legacy `mode="full"`.

## [0.13.0] - 2026-04-13

### Added — Graph-aware agent search + structured data tools

- **`SynapticGraph.from_database()`** — one-line DB → ontology migration.
  Supports SQLite, PostgreSQL, MySQL, Oracle, SQL Server. Auto-discovers
  schema, foreign keys, and M:N join tables (2+ FKs → RELATED edges
  instead of intermediate nodes). Batch processing (10K rows default).
- **Structured data tools** — `filter_nodes`, `aggregate_nodes`,
  `join_related` for SQL-like queries on graph-stored tables. All three
  now return `{total, showing, truncated}` for accurate counting.
- **`aggregate_nodes` WHERE pre-filter** — conditional aggregation
  (`where_property`/`where_op`/`where_value`). Enables "count 5-star
  reviews per product" in one call.
- **Graph-aware expansion for structured data** — `GraphExpander` now
  follows RELATED edges for ENTITY nodes, so search surfaces FK-linked
  rows (product → sales, product → reviews) automatically.
- **`join_related` edge-first strategy** — walks RELATED edges when
  available, falls back to property scan. O(degree) instead of O(N).
- **Graph composition hint** in `build_graph_context()` — tells the
  agent which tools fit the data (documents → search, structured →
  filter/aggregate/join). Distinguishes mixed graphs.
- **Foreign key metadata** surfaced in graph context — agents see
  `table.column → target_table` mappings automatically.
- **Table schema metadata** — column names, sample values, row counts
  for every structured table, auto-injected into agent system prompt.
- **Value-centric row content** — `TableIngester` now orders row values
  by semantic priority (name > description > category > rest), giving
  search the most meaningful tokens first. Removes `key=value` noise
  from content generation.
- **`SearchSession.expanded_nodes`** — tracks which nodes the agent has
  already expanded for better multi-turn coordination.
- **LLM-as-Judge evaluation** — `eval/run_all.py --judge` adds
  semantic answer validation alongside ID matching. Essential for
  filter/aggregate queries where "correct but different IDs" is common.
- **X2BEE benchmark dataset** — 40 queries (20 easy + 20 hard) over
  real production AWS RDS PostgreSQL (19,843 rows from ai_lab_main).

### Changed

- **`build_graph_context()`** — now includes structured data schemas
  and FK relationships in addition to categories. Composition section
  tells agents which tools match their query type.
- **Agent system prompt** — explicit guidance on tool selection,
  fallback strategies (try English keywords when Korean fails), and
  structured data patterns (node title format, FK chaining).
- **`HybridReranker._REASON_PRIOR`** — added `"related": 0.50` for
  RELATED edge expansion priors.
- **Public dataset runner** — now uses `EvidenceSearch` pipeline with
  optional embeddings/reranker, matching custom dataset quality.

### Fixed

- `filter_nodes` no longer early-breaks at limit, so total counts
  reported to agents are accurate.
- `aggregate_nodes` groups now include `node_title` field for FK group
  values, eliminating `goodss:` / `pr_product_base:` heuristic failures.
- `from_database()` async row_reader for PostgreSQL (asyncpg returns
  coroutines where aiosqlite returns sync iterators).

### Performance

- Agent benchmarks:
  - X2BEE Hard: 1/19 (5%) → **17/19 (89%)**
  - assort Hard: 1/15 (7%) → **12/15 (80%)**
  - KRRA Hard MRR: 0.808 → **1.000** (15/15 hit)
- Public benchmarks with EvidenceSearch + embed + reranker:
  - HotPotQA-24: 0.727 → **0.964**
  - Allganize RAG-ko: 0.621 → **0.905**
  - PublicHealthQA: 0.318 → **0.600**

## [0.12.0] - 2026-04-12

### Added — 3rd-generation retrieval + agent tool layer

- **3rd-gen retrieval pipeline** — relation-free graph, LLM-free indexing.
  `QueryAnchorExtractor` → `GraphExpander` → `HybridReranker` →
  `EvidenceAggregator` → `EvidenceSearch` facade.
- **Agent tool layer** — 7 atomic tools for multi-turn LLM exploration:
  `search`, `expand`, `get_document`, `list_categories`, `count`,
  `search_exact`, `follow`. Each returns structured `ToolResult` with
  `data`, `hints`, and `session` state.
- **SearchSession** — stateful context for multi-turn agent use. Tracks
  seen nodes, budget, query history, category coverage.
- **MCP server** — 8 new `agent_*` tools: `agent_search`, `agent_expand`,
  `agent_get_document`, `agent_list_categories`, `agent_count`,
  `agent_search_exact`, `agent_follow`, `agent_session_info`.
- **DomainProfile** — TOML-based domain configuration injection point.
  `to_dict()`, `save(path)` for round-trip serialization. New fields:
  `authority_by_kind`, `enrich_document_content`, `document_preview_chars`.
- **ProfileGenerator** — 3-tier auto profile generation (rule-based →
  OntologyClassifier → LLM). Detects locale, suggests stopwords,
  maps categories to NodeKind.
- **OntologyClassifier** — BYO embedder NodeKind classification via
  embedding cosine similarity. No torch dependency.
- **DocumentIngester** — generic JSONL → graph ingestion with
  `JsonlDocumentSource`, `InMemoryDocumentSource`, `CorpusSource` protocol.
  Document content enrichment (first chunks joined). Authority metadata.
  NFC normalization for categories/titles.
- **EntityLinker** — post-processing DF-filtered entity hub creation.
- **SqliteGraphBackend** — SQLite + `GraphTraversal` protocol
  (`shortest_path` BFS, `find_by_type_hierarchy`).
- **SQLiteBackend** improvements — NFC normalization on save, title 3x
  BM25 weight, LIKE substring fallback for Korean compound words.
- **Phrase extractors** — `KoreanPhraseExtractor`, `EnglishPhraseExtractor`,
  `create_phrase_extractor()` locale dispatcher.
- **node_metadata** helpers — `year_of()`, `authority_of()`, `is_current()`,
  `authority_ranked()`.
- **eval harness** — `ingest_krra`, `score_krra`, `score_krra_evidence`,
  `ingest_assort`, `score_assort`, `generate_profile` CLI scripts.
  KRRA + assort domain profiles and GT queries.
- **Multi-turn demo** — `examples/multi_turn_search.py` with Claude Sonnet
  validation (5/5 difficulty tiers passing).
- **683+ tests** (up from 504).

### Changed
- README rewritten for v0.12 — 3rd-gen retrieval positioning, agent tool
  quickstart, MCP server guide.

## [0.11.0] - 2026-04-09

### Added
- **KuzuBackend** — embedded property graph database using Kuzu 0.11.3.
  Native openCypher, FTS extension (Okapi BM25), and built-in graph traversal.
  Zero-config deployment (`pip install synaptic-memory[kuzu]` — no Docker, no server).
- `SynapticGraph.kuzu(db_path)` factory method for one-line setup.
- `tests/test_backend_kuzu.py` — 25 unit tests covering CRUD, search, traversal,
  batch ops, and maintenance. Runs in CI without external infrastructure.

### Removed — BREAKING
- **Neo4jBackend removed.** GPLv3 licensing on Neo4j Community, clustering limits,
  and operational overhead did not fit an MIT-licensed embedded library.
  Users still needing Neo4j can depend on the `neo4j` driver directly and
  implement the `StorageBackend` protocol themselves.
- `synaptic-memory[neo4j]` optional dependency removed.
- `tests/test_backend_neo4j.py` deleted.
- `docker-compose.yml` Neo4j service removed.
- `pytest.mark.neo4j` marker removed.

### Changed
- `CompositeBackend` now routes graph operations to Kuzu by default.
- `SynapticGraph.full(...)` and the scale preset reference Kuzu in docstrings.
- `pyproject.toml` — `scale` and `all` extras swap `neo4j>=5.25` for `kuzu>=0.11.0`.
- README Quick Start reorganized with Kuzu as the recommended embedded backend.
- Refactored README Quick Start to use factory functions.
- Refactored public API: factory functions, type stubs, reduced code duplication.

### Migration guide
- **Before:** `SynapticGraph(Neo4jBackend("bolt://localhost:7687", auth=("neo4j", "password")))`
- **After:** `SynapticGraph.kuzu("knowledge.kuzu")`

The Kuzu backend implements the same `StorageBackend` + `GraphTraversal`
protocols so Phase-level graph operations (PPR, Hebbian, consolidation)
work identically.

## [0.7.0] - 2026-03-22

### Added
- **Evidence Chain Assembly** — small LLM augmentation for multi-hop reasoning, HotPotQA Correctness 0.856 (+9.2%).
- **Personalized PageRank (PPR) engine** — replaced spreading activation, multi-hop retrieval +28%.
- **End-to-end QA benchmark** — HotPotQA 24-question suite for Cognee comparison (Correctness 0.784).
- **Auto-ontology optimization** — HybridClassifier, batch LLM processing, EmbeddingRelation, PhraseExtractor.

### Fixed
- PhraseExtractor search noise — phrase filtering and optimization.

### Changed
- Removed `__pycache__` from repo and updated `.gitignore`.

## [0.6.0] - 2026-03-21

### Added
- **Auto-ontology construction** — LLM-based ontology building with search-optimized metadata generation.
- **LLM classifier prompt optimization** — few-shot examples improved accuracy from 50% to 86%.
- **FTS + embedding hybrid scoring** — S7 Auto+Embed achieved MRR 0.83.
- **Kind/tag/search_keywords utilization** in search — FTS and ranking boost.
- **Ontology auto-construction + benchmark framework + search engine improvements** (combined release).

### Changed
- Updated README with auto-ontology, benchmark results, and differentiation points.

## [0.5.0] - 2026-03-21

### Added
- **Ontology Engine** — dynamic type hierarchy, property inheritance, relation constraint validation (`OntologyRegistry`).
- **Agent Activity Tracking** — session/tool call/decision/outcome capture (`ActivityTracker`).
- **Intent-based Agent Search** — 6 search strategies: similar_decisions, past_failures, related_rules, reasoning_chain, context_explore, general (`AgentSearch`).
- **Neo4j Backend** — native Cypher graph traversal, dual label, typed relationships, fulltext index.
- **Auto-embedding** — automatic vector generation on `add()` / `search()`.
- **Qdrant + MinIO + CompositeBackend** — storage separation by purpose.
- **5-axis Resonance Scoring** — added context axis (session tag Jaccard similarity).
- **GraphTraversal Protocol** — `shortest_path()`, `pattern_match()`, `find_by_type_hierarchy()`.
- **Node.properties** — ontology extension attributes, supported across all backends.
- **MCP 9 new tools** (total 16): agent session/action/decision/outcome tracking, ontology tools.
- 6 new `NodeKind` values: tool_call, observation, reasoning, outcome, session, type_def.
- 5 new `EdgeKind` values: is_a, invoked, resulted_in, part_of, followed_by.
- `docker-compose.yml` for Neo4j dev environment.
- `docs/COMPARISON.md` — comparison with existing agent memory systems.
- 185+ unit tests, 22 Neo4j integration tests.

### Fixed
- MemoryBackend fuzzy search ineffectiveness bug + 12 edge case QA tests added.
- Library distribution quality: `__version__`, `py.typed`, lazy imports, embedding extra.

## [0.4.0] - 2026-03-21

### Added
- **MCP Server** — 7 tools (knowledge search/add/link/reinforce/stats/export/consolidate).
- **SQLite Backend** — FTS5, recursive CTE, WAL mode.
- **QA Test Suite** — 169 Wikipedia + 368 GitHub real-data verification cases.
- `synaptic-mcp` CLI entry point.

## [0.3.0] - 2026-03-21

### Added
- **Protocol implementations** — LLM QueryRewriter, RegexTagExtractor, EmbeddingProvider.
- **LRU Cache** — NodeCache with hit rate tracking.
- **JSON Exporter** — structured JSON export.
- **Node Merge** — duplicate node merging with edge reconnection.
- **Find Duplicates** — title similarity-based duplicate detection.

## [0.2.0] - 2026-03-21

### Added
- **PostgreSQL backend** — asyncpg + pgvector HNSW + pg_trgm + recursive CTE.
- Vector search with cosine distance (pgvector).
- Trigram fuzzy matching with graceful ILIKE fallback.
- Hybrid search: FTS + fuzzy + vector merged results.
- Connection pooling (asyncpg Pool, min=2, max=10).
- Configurable `embedding_dim` parameter.
- `ResonanceWeights` added to public exports.
- Configurable consolidation thresholds (TTL, promotion access counts).
- README.md, ARCHITECTURE.md, ROADMAP.md documentation.
- GitHub Actions CI (Python 3.12/3.13).
- Integration test suite for PostgreSQL (13 tests).

### Changed
- Consolidation constants now accept `__init__` parameters instead of module globals.

## [0.1.0] - 2026-03-21

### Added
- Core models: Node, Edge, ActivatedNode, SearchResult, DigestResult.
- Enums: NodeKind (9), EdgeKind (7), ConsolidationLevel (4).
- Protocols: StorageBackend, Digester, QueryRewriter, TagExtractor.
- SynapticGraph facade: add, link, search, reinforce, consolidate, prune, decay.
- Hybrid 3-stage search: FTS + fuzzy, synonym expansion, query rewrite.
- Hebbian learning engine: co-activation reinforcement with anti-resonance.
- 4-axis resonance scoring: relevance x importance x recency x vitality.
- Memory consolidation cascade: L0 -> L1 -> L2 -> L3 with TTL and promotion.
- Korean/English synonym map (38 groups).
- Markdown exporter.
- MemoryBackend (dict-based, zero dependencies).
- SQLiteBackend (FTS5, recursive CTE, WAL mode).
- 93 unit tests, pyright strict, ruff clean.
