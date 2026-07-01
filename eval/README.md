# eval/ — Independent Evaluation Harness

This directory is the standalone evaluation environment for Synaptic Memory.
It is **not** part of the library and is not shipped to PyPI.

## Purpose

- **Parity checks** — validate that new backends (e.g. Kuzu) match existing
  baselines (Memory) on real benchmarks.
- **RAG comparison** — run Synaptic Memory head-to-head against top-k RAG,
  XGEN's Fuseki-based GraphRAG, and other baselines on real document
  corpora (e.g. KRRA).
- **Regression monitoring** — every core phase (Kuzu swap, typed properties,
  ontology-aware retrieval, etc.) runs against the same query set so we can
  detect regressions quickly.

## Directory layout

```
eval/
├── README.md            # This file
├── scripts/
│   ├── parse_krra.py    # HWP/XLSX → documents.jsonl + chunks.jsonl
│   ├── ingest_krra.py   # JSONL → Kuzu graph (19,720 nodes, ~37K edges)
│   ├── score_krra.py    # Seed GT → graph.search() → MRR/nDCG/P@K/R@K
│   └── kuzu_parity.py   # Memory vs Kuzu parity on enterprise scenario
├── data/
│   ├── raw/             # Raw documents (gitignored)
│   │   └── krra/        # KRRA 마사회 corpus
│   ├── parsed/          # xgen-doc2chunk output (gitignored, NFD text)
│   │   └── krra/        # 1,110 docs + 18,600 chunks + errors
│   ├── queries/         # Hand-crafted ground truth (committed)
│   │   └── krra.json    # 20 seed queries (NFC, title keyword match)
│   └── krra_graph.kuzu  # Built graph (gitignored, ~240MB)
└── results/             # Benchmark run outputs (gitignored)
    └── krra_baseline_*.json
```

## KRRA benchmark pipeline

### One-time setup

```bash
# 1. Parse raw documents → JSONL (~3 min for 1,110 files)
uv run python eval/scripts/parse_krra.py

# 2. Ingest into Kuzu with NFC normalization (~8 min)
uv run python eval/scripts/ingest_krra.py

# 3. Score against seed GT
uv run python eval/scripts/score_krra.py
```

### OpenIE PoC on local 마사회 chunks

`eval/scripts/openie_mz_poc.py` is the v0.30 hybrid-graph smoke harness.
It rebuilds a chunk-preserving graph from `~/synaptic-eval/mz_chunks.jsonl`,
scores the baseline, applies the opt-in OpenIE layer to a copied DB, and
embeds any new OpenIE entity hubs before scoring again. The existing
`mz_full.db` is not modified.

```bash
# Verify the memory operating layer without an LLM or embedding server. This
# exercises retrieval/feedback ledgers, scoped reinforcement, Hebbian feedback
# edges, relation/edge scores, feedback-fed consolidation, edge provenance,
# pollution/growth signals, and the compact health report.
uv run --extra sqlite python eval/scripts/memory_operating_poc.py \
  --results ~/synaptic-eval/memory_operating_poc_results.json
```

The memory-operating smoke is deterministic and exits non-zero when any gate
fails. The result JSON records the individual gates and a compact
`summary.health` payload. Current gates cover:

- retrieval and feedback ledger roundtrips;
- implicit feedback staying weak and scope-local;
- task/test-style success promoting node and relation scores without double
  counting global scope;
- public `graph.reinforce()` flowing through the same feedback ledger path;
- Hebbian edge creation plus local/global edge `MemoryScore` updates;
- feedback counts feeding consolidation;
- source/model/prompt provenance staying on edge/event metadata;
- repeated failure, property conflict, supersession, drift-spike, and
  low-confidence relation signals;
- new-entity, new-relation, and relation-reinforced lifecycle signals in
  `memory_health(since=...)`;
- suspicious memory being flagged and optionally penalized, not deleted.

```bash
# Verify ingest/scoring without an LLM server
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py --skip-openie

# Run OpenIE when a local OpenAI-compatible Qwen server is available
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model /home/son/xgen-models/huggingface/Qwen3.6-27B \
  --openie-max-concurrency 4

# Re-score a bounded OpenIE smoke from cache only. This never calls an LLM:
# only chunks already covered by the cache are eligible for OpenIE replay.
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-source-limit 200 \
  --openie-max-chunks 5 \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-only \
  --openie-cache-missing-output ~/synaptic-eval/openie_cache_missing_200.jsonl \
  --llm-model Qwen3.6-27B \
  --relation-probe-limit 50 \
  --min-relation-expanded-lift 1 \
  --min-relation-evidence-lift 1 \
  --min-strong-relation-evidence-rate 0.5 \
  --min-openie-cache-coverage 0.02

# Audit a cache file before/after warming. This never calls an LLM.
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-audit \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --openie-cache-audit-bad-output ~/synaptic-eval/openie_cache_bad_rows_200.jsonl \
  --openie-cache-compact-output ~/synaptic-eval/openie_cache_mz_200_qwen.compact.jsonl \
  --results ~/synaptic-eval/openie_cache_audit_200_results.json

# Warm only the missing OpenIE cache rows exported above. This does not build
# DBs or run retrieval; it appends successful extractions to --openie-cache.
# Add --openie-cache-warm-dry-run first to count pending uncached rows without
# calling an LLM.
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --openie-cache-warm-input ~/synaptic-eval/openie_cache_missing_200.jsonl \
  --openie-cache ~/synaptic-eval/openie_cache_mz_200_qwen.jsonl \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-v4-flash \
  --llm-api-key-env DEEPSEEK_API_KEY \
  --openie-model-profile deepseek_v4_flash \
  --openie-cache-warm-limit 50 \
  --openie-cache-warm-total-chunks 200 \
  --openie-cache-warm-target-coverage 0.5 \
  --openie-cache-warm-pending-output ~/synaptic-eval/openie_cache_pending_200_50.jsonl \
  --openie-cache-warm-failure-output ~/synaptic-eval/openie_cache_failures_200_50.jsonl \
  --results ~/synaptic-eval/openie_cache_warm_200_results.json

# Scale OpenIE with DeepSeek Flash. Keep the API key only in the process
# environment; do not put it in code, cache files, docs, or DB metadata.
export DEEPSEEK_API_KEY=...
uv run --extra sqlite --extra embedding python eval/scripts/openie_mz_poc.py \
  --max-input-chunks 200 \
  --openie-max-chunks 50 \
  --llm-base-url https://api.deepseek.com/v1 \
  --llm-model deepseek-v4-flash \
  --llm-api-key-env DEEPSEEK_API_KEY \
  --openie-model-profile deepseek_v4_flash \
  --openie-max-concurrency 8 \
  --relation-probe-limit 100 \
  --min-relation-expanded-lift 10 \
  --min-relation-evidence-lift 5 \
  --min-strong-relation-evidence-rate 0.5
```

For a dependency-light ingest/search smoke without HTTP embedding, add
`--embed-base-url ''`.

The script writes `~/synaptic-eval/mz_openie_poc_results.json` and exits
non-zero when the default gates fail:

- OpenIE R@5 must not regress against the baseline DB.
- At least one query must be scoreable in both baseline and OpenIE DBs.
- In non-skipped OpenIE mode, at least one chunk must be selected, at
  least one non-MENTIONS OpenIE relation edge must be created, and
  extraction failures must be zero.
- OpenIE LLM extraction is bounded by `--openie-max-concurrency`; graph
  writes are applied in deterministic chunk order.
- Reasoning-heavy OpenAI-compatible models may need
  `--openie-max-output-tokens 4096` so final JSON is not truncated after
  internal reasoning tokens.
- `purge_openie_artifacts()` must logically restore the baseline graph
  fingerprint (`--verify-revertibility`, on by default).
- `--min-delta-r5` can raise the improvement gate above the default `0.0`.
- `relation_probe` compares graph expansion on/off for OpenIE relation
  targets. It records relation-level and strong/weak group breakdowns.
- `--min-relation-expanded-lift`, `--min-relation-evidence-lift`, and
  `--min-strong-relation-evidence-rate` turn relation-probe numbers into
  optional gates. Defaults are zero so existing no-regress checks still run.
- `--openie-cache-only` is the safe scale-smoke mode for cached extractions:
  it preserves model/provenance labels, never calls a remote LLM, and filters
  OpenIE replay to cache-covered chunks. The result JSON records
  `cache_checked_chunks`, `cache_eligible_chunks`, `cache_skipped_chunks`, and
  `cache_coverage_rate` so cache coverage problems are separate from extraction
  quality problems. Use `--openie-cache-missing-output` to export the skipped
  chunk rows as JSONL input for the next cache-warming run.
- `--min-openie-cache-coverage` turns cache coverage into an optional gate for
  scale smokes. Keep it low for early cache-only checks, then raise it after
  cache warming.
- `--openie-cache-audit` reads an OpenIE cache JSONL without calling an LLM and
  reports invalid JSON lines, invalid record shapes, duplicate keys, parse
  failures, empty extractions, and total extracted entity/triple counts. Use
  `--openie-cache-audit-bad-output` to export rows that need cleanup or retry.
  Use `--openie-cache-compact-output` to write a clean cache containing only
  parseable records, deduplicated by the latest value for each key.
- `--openie-cache-warm-input` consumes those missing rows, calls the configured
  OpenAI-compatible LLM, appends successful extractions to `--openie-cache`, and
  writes an `openie_cache_warm` summary to `--results`. Re-running the same
  input is idempotent: rows already present in the cache are counted under
  `rows_skipped_cached` and do not call the LLM again. Use
  `--openie-cache-warm-dry-run` to count `rows_pending` before spending tokens.
  `--openie-cache-warm-limit` caps pending uncached rows for the current batch;
  uncached rows beyond that batch are reported as `rows_deferred_limit`. Use
  `--openie-cache-warm-total-chunks` with a missing export to project coverage
  after the current batch succeeds. Add `--openie-cache-warm-target-coverage`
  to report the rows and batches still needed to reach a target cache coverage.
  Use
  `--openie-cache-warm-pending-output` to write the exact pending batch rows to
  JSONL for inspection. Use `--openie-cache-warm-failure-output` to write only
  failed rows as retry input.

### Graph structure (current — Day 1, structural only)

- **Category** (10, `CONCEPT`) — directory name
- **Document** (1,110, `ENTITY`, `content=""`) — title + metadata only
- **Chunk** (18,600, `CHUNK`) — actual text, 1000 chars, 200 overlap
- **Edges**: `PART_OF` (doc→cat), `CONTAINS` (doc→chunk), `NEXT_CHUNK` (sequential)

No entity extraction, no cross-doc linking, no embeddings yet — that's the
Track 🅑 ontology work.

## Baseline results (2026-04-12, Day 1)

**20 seed queries, k=10, FTS only, no embeddings:**

| Metric | NFD graph | NFC graph | Δ |
|--------|-----------|-----------|---|
| MRR | 0.525 | **0.650** | +23.8% |
| Mean P@10 | 0.186 | **0.392** | +110.8% |
| Mean R@10 | 0.417 | 0.453 | +8.6% |
| Mean nDCG@10 | 0.431 | **0.503** | +16.7% |
| Hit rate | 12/20 | 13/20 | +1 |
| Avg latency | 728ms | **585ms** | -20% |

## Known issues (Day 1 findings)

### 🔴 Library-level NFC/NFD bug
`graph.add()` and `graph.search()` do not normalize Unicode. Only
`phrase_extractor.py` does. macOS HFS+ stores Korean filenames as NFD, so
any corpus ingested from a Mac source silently fails substring search.

**Workaround in `ingest_krra.py`**: normalize title/content/category/source
to NFC at load time.

**Proper fix (Track 🅒)**: normalize at `graph.add()` entry (title, content,
tags, source, properties) and `graph.search()` entry (query).

### 🟡 Chunk granularity mismatch
Document nodes have `content=""` so FTS can only match their title.
Meanwhile chunk body matches from unrelated documents outrank the relevant
docs' chunks. 7 of 20 seed queries hit zero due to this, even on NFC graph.

Example: query `"인권영향평가 결과"` returns top chunks from `경영실적보고서`,
`ESG경영진단`, `시리즈 경주 시행 결과보고` — none of which have
"인권영향평가" in their title. The docs that DO have it in the title are
pushed out of top-10.

Fix options (Track 🅒 or Track 🅑 Phase 6):
1. Raise title weight in `search.py` FTS scoring
2. Duplicate title into `content` for Document nodes
3. Aggregate chunk scores into parent Document score (HippoRAG2 style)

### 🟡 parse_krra.py year=null for all 1,110 docs
`re.match(r"(\d{4})년도", filename)` fails against NFD filenames. Fix
`_extract_year`, `_extract_title`, `_extract_category` to normalize
`fpath.name` to NFC **before** regex — but leave `_doc_id()` computing
against raw NFD path to avoid breaking existing GT.

## Running the parity check (Memory vs Kuzu)

```bash
uv run python eval/scripts/kuzu_parity.py
```

Expected output: side-by-side MRR/nDCG/Recall/Latency for MemoryBackend
and KuzuBackend on the 15-query enterprise scenario. The two columns
should match within noise — anything else is a regression.
