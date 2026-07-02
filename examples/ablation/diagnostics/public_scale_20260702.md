# Public Large-Corpus Scale Smoke - 2026-07-02

## Datasets

| Dataset | Local artifact | Corpus | Queries | Smoke scope |
|---------|----------------|-------:|--------:|-------------|
| BEIR FiQA test | `tests/benchmark/data/fiqa.json` | 57,638 docs | 648 | 5-10 queries |
| BEIR TREC-COVID test | `tests/benchmark/data/trec_covid.json` | 171,332 docs | 50 | 10 queries |
| BEIR MS MARCO passage dev | `tests/benchmark/data/msmarco_passage.json` + `.corpus.jsonl` | 1M shard by default; 5M/full side-by-side shards from 8,841,823 source passages | validation qrels | manual large tier |

Mode: embedder-free `graph.search()` with `SqliteGraphBackend`.

## Commands

```bash
uv run --extra eval python examples/ablation/download_benchmarks.py --only fiqa
uv run --extra eval python examples/ablation/download_benchmarks.py --only trec_covid

PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only fiqa --subset 5 --corpus-limit 10000 --use-sqlite-graph
PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only fiqa --subset 10 --corpus-limit 25000 --use-sqlite-graph
PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only fiqa --subset 10 --use-sqlite-graph

PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only trec_covid --subset 10 --corpus-limit 50000 --use-sqlite-graph
PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only trec_covid --subset 10 --corpus-limit 100000 --use-sqlite-graph
PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only trec_covid --subset 10 --use-sqlite-graph

uv run --extra eval python examples/ablation/download_benchmarks.py --only msmarco_passage --large-corpus-limit 1000000
PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 50 --corpus-limit 1000000 --use-sqlite-graph

# Persistent 1M DB for repeat runs after the initial build:
PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 50 --corpus-limit 1000000 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_1m.db --overwrite-sqlite-db
PYTHONUNBUFFERED=1 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 50 --corpus-limit 1000000 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_1m.db --reuse-sqlite-db

# Full MS MARCO passage tier:
uv run --extra eval python examples/ablation/download_benchmarks.py --only msmarco_passage --large-scale-tier full
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 50 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_full.db --overwrite-sqlite-db --sqlite-fast-build
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 50 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_full.db --reuse-sqlite-db
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 50 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_full.db --reuse-sqlite-db --diagnose-raw-fts-limit 500
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 50 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_full.db --reuse-sqlite-db --diagnose-raw-fts-limit 500
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 200 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_full.db --reuse-sqlite-db
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 200 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_full.db --reuse-sqlite-db
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 500 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_full.db --reuse-sqlite-db
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500 uv run --extra sqlite python examples/ablation/run_tier1_benchmarks.py --only msmarco --subset 500 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_full.db --reuse-sqlite-db
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500 uv run python examples/ablation/run_agent_search_benchmarks.py --subset 50 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --sqlite-db-path tests/benchmark/data/msmarco_full.db --modes graph_search,deep_search,scripted_session --result-limit 20 --tool-limit 10 --read-top-k 0 --scripted-turns 2
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500 uv run python examples/ablation/run_agent_search_benchmarks.py --subset 10 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --sqlite-db-path tests/benchmark/data/msmarco_full.db --modes agent_search --result-limit 20 --tool-limit 10 --intent context_explore

# DeepSeek Flash agent-loop quality path.
# Put DEEPSEEK_API_KEY in shell env, the repo .env, or the parent workspace .env.
# Do not put the key in docs, commands, JSONL, or DBs.
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500 uv run python examples/ablation/run_agent_loop_benchmarks.py --llm-preset deepseek --subset 20 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --sqlite-db-path tests/benchmark/data/msmarco_full.db --max-turns 5 --llm-timeout 180 --preflight-timeout 15 --out-jsonl examples/ablation/diagnostics/agent_loop_deepseek_v4_flash_20.jsonl --resume

# Historical local Ollama fallback smoke when the H100/Qwen3.6 tunnel is down.
# This is functional/navigation-only and should not be used as the quality
# reference for the agent loop.
# Terminal 1:
ssh -N -L 18134:127.0.0.1:11434 go243
# Terminal 2:
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500 uv run python examples/ablation/run_agent_loop_benchmarks.py --subset 20 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --sqlite-db-path tests/benchmark/data/msmarco_full.db --llm-base-url http://127.0.0.1:18134/v1 --model qwen3:14b --api-key-env LLM_API_KEY --max-turns 3 --llm-timeout 180 --preflight-timeout 10 --allow-zero-tool-answer --out-jsonl examples/ablation/diagnostics/agent_loop_ollama_qwen3_14b_smoke.jsonl --resume
PYTHONUNBUFFERED=1 SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20 SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500 uv run python examples/ablation/run_agent_loop_benchmarks.py --subset 20 --msmarco-path tests/benchmark/data/msmarco_passage_full.json --corpus-limit 8841823 --sqlite-db-path tests/benchmark/data/msmarco_full.db --llm-base-url http://127.0.0.1:18134/v1 --model qwen3:14b --api-key-env LLM_API_KEY --max-turns 3 --llm-timeout 180 --preflight-timeout 10 --out-jsonl examples/ablation/diagnostics/agent_loop_ollama_qwen3_14b_force_first.jsonl
```

## FiQA Results

Before the SQLite batch FTS optimization:

| Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Build | Search |
|-----:|--------:|-------:|----:|-----:|-------:|------:|-------:|
| 10,000 | 5 | 0.425 | 0.300 | 0.400 | 3/5 | 13.2s | 0.1s |
| 25,000 | 10 | 0.353 | 0.333 | 0.383 | 5/10 | 101.7s | 0.6s |
| 57,638 | 10 | 0.202 | 0.233 | 0.333 | 5/10 | 577.4s | 1.5s |

After the SQLite batch FTS optimization:

| Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Build | Search |
|-----:|--------:|-------:|----:|-----:|-------:|------:|-------:|
| 10,000 | 5 | 0.425 | 0.300 | 0.400 | 3/5 | 3.2s | 0.1s |
| 25,000 | 10 | 0.353 | 0.333 | 0.383 | 5/10 | 9.3s | 0.6s |
| 57,638 | 10 | 0.202 | 0.233 | 0.333 | 5/10 | 58.4s | 1.4s |

## TREC-COVID Results

After the SQLite batch FTS optimization:

| Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Build | Search |
|-----:|--------:|-------:|----:|-----:|-------:|------:|-------:|
| 50,000 | 10 | 0.933 | 0.008 | 0.015 | 10/10 | 20.6s | 1.4s |
| 100,000 | 10 | 0.750 | 0.007 | 0.012 | 10/10 | 55.2s | 2.8s |
| 171,332 | 10 | 0.598 | 0.004 | 0.011 | 10/10 | 135.1s | 5.2s |

TREC-COVID has many relevant documents per query, so R@5/R@10 is naturally
small in this smoke even when Hit@10 is perfect.

## MS MARCO Passage Results

Manual large-tier shard from BEIR/MS MARCO passage validation:

| Mode | Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Build | Search |
|------|-----:|--------:|-------:|----:|-----:|-------:|------:|-------:|
| temp SQLite | 100,000 | 50 | 0.673 | 0.740 | 0.770 | 39/50 | 81.9s | 5.4s |
| temp SQLite | 1,000,000 | 50 | 0.462 | 0.543 | 0.580 | 30/50 | 1913.3s | 69.9s |
| persistent SQLite build | 1,000,000 | 50 | 0.462 | 0.543 | 0.580 | 30/50 | 2184.3s | 71.0s |
| persistent SQLite reuse | 1,000,000 | 50 | 0.462 | 0.543 | 0.580 | 30/50 | 0.0s | 70.1s |
| persistent SQLite reuse + English query filter | 1,000,000 | 50 | 0.479 | 0.553 | 0.600 | 31/50 | 0.0s | 9.1s |
| persistent SQLite reuse + tag-filtered anchors | 1,000,000 | 50 | 0.479 | 0.553 | 0.600 | 31/50 | 0.0s | 7.5s |
| persistent SQLite fast build | 5,000,000 | 50 | 0.334 | 0.407 | 0.473 | 24/50 | 288.7s | 40.8s |
| persistent SQLite reuse | 5,000,000 | 50 | 0.334 | 0.407 | 0.473 | 24/50 | 0.0s | 41.5s |
| persistent SQLite reuse + AND-first FTS threshold 20 | 5,000,000 | 50 | 0.334 | 0.407 | 0.473 | 24/50 | 0.0s | 37.8s |
| persistent SQLite full fast build + AND-first FTS threshold 20 | 8,841,823 | 50 | 0.212 | 0.347 | 0.393 | 20/50 | 642.2s | 71.6s |
| persistent SQLite full reuse + AND-first FTS threshold 20 | 8,841,823 | 50 | 0.212 | 0.347 | 0.393 | 20/50 | 0.0s | 68.7s |
| persistent SQLite full reuse + raw FTS@500 diagnostic | 8,841,823 | 50 | 0.212 | 0.347 | 0.393 | 20/50 | 0.0s | 69.7s |
| persistent SQLite full reuse + opt-in lexical rerank pool 500 | 8,841,823 | 50 | 0.234 | 0.353 | 0.433 | 22/50 | 0.0s | 70.4s |
| persistent SQLite full reuse + AND-first FTS threshold 20 (200-query check) | 8,841,823 | 200 | 0.250 | 0.379 | 0.476 | 97/200 | 0.0s | 249.9s |
| persistent SQLite full reuse + opt-in lexical rerank pool 500 (200-query check) | 8,841,823 | 200 | 0.301 | 0.436 | 0.506 | 103/200 | 0.0s | 259.0s |
| persistent SQLite full reuse + AND-first FTS threshold 20 (500-query check) | 8,841,823 | 500 | 0.219 | 0.344 | 0.446 | 229/500 | 0.0s | 546.0s |
| persistent SQLite full reuse + opt-in lexical rerank pool 500 (500-query check) | 8,841,823 | 500 | 0.264 | 0.399 | 0.483 | 248/500 | 0.0s | 577.3s |
| persistent SQLite full reuse + TEI reranker + FTS seed 200 + cross top 200 | 8,841,823 | 50 | 0.211 | 0.307 | 0.393 | 20/50 | 0.0s | 72.7s |

Raw FTS pool diagnostic for the full reuse run:

| Pool | MRR@10 | R@5 | R@10 | Hit@10 | Any@Pool | Raw FTS Time |
|-----:|-------:|----:|-----:|-------:|---------:|-------------:|
| 500 | 0.214 | 0.327 | 0.413 | 21/50 | 40/50 | 68.7s |

## MS MARCO Agent Tool-Surface Retrieval Results

LLM-free measurement of the deterministic agent-facing retrieval surfaces:

This is not the full agent loop that changes follow-up queries based on earlier
evidence. Reach@All counts queries where at least one relevant document appeared
anywhere in the returned evidence for that mode. Retrieval Ops/Q counts the
single `graph.search`/`agent_search` operation or the SearchSession tool calls
used by agent-tool modes. Do not interpret this table as the score for
LLM-planned agent exploration; use `run_agent_loop_benchmarks.py` for that.

| Mode | Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Reach@All | Search | Retrieval Ops/Q | Docs/Q |
|------|-----:|--------:|-------:|----:|-----:|-------:|----------:|-------:|----------------:|-------:|
| `graph_search` | 8,841,823 | 50 | 0.234 | 0.353 | 0.433 | 22/50 | 25/50 | 69.8s | 1.00 | 17.86 |
| `deep_search` | 8,841,823 | 50 | 0.226 | 0.333 | 0.433 | 22/50 | 22/50 | 203.4s | 3.00 | 10.00 |
| `scripted_session` | 8,841,823 | 50 | 0.226 | 0.333 | 0.433 | 22/50 | 24/50 | 413.7s | 4.00 | 19.76 |
| `agent_search` context-explore smoke (n=10) | 8,841,823 | 10 | 0.217 | 0.500 | 0.500 | 5/10 | 5/10 | 48.8s | 1.00 | 5.00 |

## MS MARCO LLM-Planned Agent Loop Smoke

This is the real `run_agent_loop()` path: the model can inspect earlier
evidence, change tool choice, and rewrite follow-up search targets. The H100
Qwen3.6 tunnel was unavailable during this run (`vllm-tunnel1/2` restarting
with `No route to host`), so this smoke used the `go243` Ollama fallback
`qwen3:14b`. Treat it as a functional navigation smoke, not a Qwen3.6 quality
reference.

The preferred quality path for follow-up runs is now DeepSeek Flash via
`--llm-preset deepseek`, with the API key supplied only through runtime env or a
gitignored `.env` file.

| Mode | Model | Docs | Queries | Reach | Mean turns | Mean calls | Mean first rel turn | Mean elapsed | Mean unique tools | Mean search targets | Mean rewrites | Multi-tool | Rewrites | Zero-tool |
|------|-------|-----:|--------:|------:|-----------:|-----------:|--------------------:|-------------:|------------------:|--------------------:|--------------:|-----------:|---------:|----------:|
| historical zero-tool allowed | `qwen3:14b` via Ollama | 8,841,823 | 20 | 6/20 | 2.50 | 1.90 | 1.17 | 41.3s | 1.90 | 1.85 | 1.20 | 12/20 | 14/20 | 2/20 |
| force-first-tool default | `qwen3:14b` via Ollama | 8,841,823 | 20 | 9/20 | 2.60 | 2.10 | 1.33 | 42.2s | 1.75 | 1.65 | 1.10 | 11/20 | 16/20 | 0/20 |

Historical per-query report: `examples/ablation/diagnostics/agent_loop_20260702_181702.md`.
Historical incremental rows: `examples/ablation/diagnostics/agent_loop_ollama_qwen3_14b_smoke.jsonl`.
Force-first per-query report: `examples/ablation/diagnostics/agent_loop_20260702_184227.md`.
Force-first incremental rows: `examples/ablation/diagnostics/agent_loop_ollama_qwen3_14b_force_first.jsonl`.

Observed failure pattern: the fallback model demonstrates real exploration
behavior, but quality is not yet a Qwen3.6-grade reference. It made no tool call
on 2/20 queries and accumulated 6 empty-result tool calls. The tool histogram
was `deep_search=17`, `search=12`, `get_document=9`, confirming the loop uses
the enhanced search surface and then changes targets/tools, but often moves to
weak follow-up targets on short MS MARCO web questions.

The historical run above allowed zero-tool answers. Later benchmark runs should
use the runner default, which forces at least one retrieval tool before accepting
a final answer; use `--allow-zero-tool-answer` only to reproduce this exact
baseline.

The force-first-tool rerun improved reach from 6/20 to 9/20 with no observed
regressions on this subset. The improved qids were `201376`, `1101278`, and
`165002`; the latter two were previous zero-tool failures that became
`deep_search` hits. Empty-result tool calls also fell from 6 to 4.

The local artifacts are gitignored:

- `tests/benchmark/data/msmarco_passage.json` - 511 KB manifest
- `tests/benchmark/data/msmarco_passage.corpus.jsonl` - 35 MB at 100k, 361 MB at 1M
- `tests/benchmark/data/msmarco_passage_5m.json` - 511 KB 5M manifest
- `tests/benchmark/data/msmarco_passage_5m.corpus.jsonl` - 1.8 GB, 5,000,000 rows
- `tests/benchmark/data/msmarco_passage_full.json` - 511 KB full manifest
- `tests/benchmark/data/msmarco_passage_full.corpus.jsonl` - 3.2 GB, 8,841,823 rows
- `tests/benchmark/data/msmarco_1m.db` - 1.2 GB persistent SQLite DB
- `tests/benchmark/data/msmarco_1m.db.tier1.json` - 535 byte reuse sidecar
- `tests/benchmark/data/msmarco_5m.db` - 6.0 GB persistent SQLite DB
- `tests/benchmark/data/msmarco_5m.db.tier1.json` - 541 byte reuse sidecar
- `tests/benchmark/data/msmarco_full.db` - 11 GB persistent SQLite DB
- `tests/benchmark/data/msmarco_full.db.tier1.json` - 499 byte reuse sidecar

## Interpretation

- Search latency remains usable at 171k docs: 5.2s over 10 queries.
- MS MARCO confirms the large-tier path on a web passage corpus: 100k docs,
  50 queries, 5.4s total search, and 0.673 MRR@10 without embeddings or reranking.
- MS MARCO 1M is now proven end-to-end, but it is a heavy manual tier:
  31.9 minutes build time and 69.9s total search for 50 queries.
- The main large-corpus bottleneck is still initial FTS/index build, not retrieval.
- Avoiding unnecessary FTS deletes for newly inserted nodes reduced full FiQA build time by about 9.9x.
- Raising benchmark ingest batches to 20k reduced full TREC-COVID build time by about 2.7x.
- `--corpus-limit` provides practical staged scale gates while preserving selected query gold docs.
- Long 1M runs should use ingest progress output; without it the build phase is
  too quiet for practical monitoring.
- Repeat 1M runs should use `--sqlite-db-path` + `--reuse-sqlite-db`; the first
  run still pays the materialization cost, but follow-up searches can skip the
  31.9 minute ingest/index phase after sidecar metadata validation.
- The persistent 1M DB is now built locally. A reuse run validates the sidecar,
  reports 1,000,000 docs, skips ingest, and preserves identical quality while
  reducing build time from 2184.3s to 0.0s.
- English query-term filtering removes high-frequency question glue
  (`how/is/the/of/to` etc.) before FTS5 `OR` matching. On the persistent 1M
  DB this reduced 50-query search time from 70.1s to 9.1s while improving
  MRR@10 from 0.462 to 0.479.
- QueryAnchor category loading now asks backends for nodes tagged `category`
  instead of materializing the first 500 `CONCEPT` rows and filtering in
  Python. On the persistent 1M MS MARCO DB, first anchor extraction dropped
  from roughly 1.7-2.0s to 0.218s, and the 50-query reuse smoke improved from
  9.1s to 7.5s with unchanged quality.
- 5M MS MARCO corpus data is now locally available as a side-by-side shard.
  Generate it with:
  `uv run --extra eval python examples/ablation/download_benchmarks.py --only msmarco_passage --large-corpus-limit 5000000 --large-output-suffix _5m`
  and run it with:
  `uv run python examples/ablation/run_tier1_benchmarks.py --only msmarco --msmarco-path tests/benchmark/data/msmarco_passage_5m.json --corpus-limit 5000000 --use-sqlite-graph --sqlite-db-path tests/benchmark/data/msmarco_5m.db --overwrite-sqlite-db --sqlite-fast-build`.
- `--sqlite-fast-build` applies relaxed SQLite durability PRAGMAs only for
  rebuildable benchmark DBs. A first 5M attempt without it reached 1M docs in
  1428.3s; with fast build, the full 5M ingest completed in 273.4s and the
  full build+checkpoint+search report completed in 288.7s build / 40.8s search.
- 5M FTS-only quality drops versus 1M because the candidate set has 5x more
  distractors and no embeddings/reranker: MRR@10 0.334 and Hit@10 24/50. The
  important result is that the pipeline now has a reproducible 5M persistent
  tier and can reuse it with 0.0s build time.
- 5M search is still FTS-bound: a 50-query timing pass measured 42.538s total,
  with 41.379s inside SQLite FTS (97.3%). Setting
  `SYNAPTIC_SQLITE_FTS_AND_FIRST_THRESHOLD=20` keeps the same 5M quality
  metrics while reducing reuse search from 41.5s to 37.8s.
- Full MS MARCO passage is now locally available and reproducible with
  `--large-scale-tier full`. The full shard preserves all 7,433 validation
  gold docs with 0 missing gold docs and writes exactly 8,841,823 JSONL rows.
- The full SQLite DB materializes 8,841,823 nodes and can be reused with
  sidecar validation. Fast build completed in 642.2s; follow-up search-only
  reuse completed in 68.7s over 50 queries.
- Full-scale FTS-only quality drops further than 5M (MRR@10 0.212, Hit@10
  20/50), confirming that the next large-corpus work should improve candidate
  recall and ranking rather than only making storage bigger.
- The raw FTS@500 diagnostic shows the headroom: official gold appears in the
  top-500 pool for 40/50 queries but only reaches the raw top-10 for 21/50 and
  the final EvidenceSearch top-10 for 20/50. This separates candidate recall
  from final ranking and gives future large-corpus experiments a concrete
  target.
- `SYNAPTIC_SQLITE_FTS_LEXICAL_RERANK_POOL=500` is the first opt-in ranking
  improvement on the full tier: it keeps the same persistent DB and lifts
  full reuse from MRR@10 0.212 / Hit@10 20/50 to MRR@10 0.234 / Hit@10 22/50
  with similar total search time.
- The 200-query full-tier check strengthens that signal: the same opt-in
  rerank pool lifts MRR@10 from 0.250 to 0.301 and Hit@10 from 97/200 to
  103/200, so the improvement is not limited to the initial 50-query smoke.
- The 500-query full-tier check keeps the same trend: MRR@10 improves from
  0.219 to 0.264 and Hit@10 from 229/500 to 248/500. Search cost increases
  from 546.0s to 577.3s, a roughly 5.7% latency increase for a roughly 20.5%
  MRR@10 lift.
- Agent tool-surface retrieval now has a reproducible full-corpus runner. On
  flat MS MARCO passages, deterministic `deep_search` and a two-turn scripted
  session do not improve top-10 quality over `graph.search`; the scripted
  session raises cumulative reach versus `deep_search` (22/50 to 24/50) but
  costs about 2x more. The true agent benchmark should use the LLM-planned
  `run_agent_loop()` path, where the agent can rewrite follow-up queries and
  change search targets based on earlier evidence.
- The first live `run_agent_loop()` full-corpus smoke ran through an Ollama
  `qwen3:14b` fallback while the H100/Qwen3.6 tunnel was down. The 20-query
  pass reached 6/20 MS MARCO gold documents and, importantly, recorded actual
  exploration behavior: query rewrites occurred on 14/20 queries and multiple
  tool types on 12/20 queries. This confirms the benchmark is measuring
  agent-driven follow-up search, not just a single retrieval call. The fallback
  model also exposed an agent-control quality issue: it skipped tools entirely
  on 2/20 queries and sometimes chased weak rewritten targets, so it remains a
  functional navigation smoke rather than a final quality reference.
- Enforcing one retrieval tool before accepting a benchmark answer removed the
  zero-tool failure mode on the same 20-query slice and lifted reach from 6/20
  to 9/20. This is now the benchmark runner default; applications can still keep
  the original behavior unless they opt into `force_first_tool=True`.
- TEI cross-reranking now handles large candidate pools without TEI batch-size
  errors by chunking requests, but the full 8.84M reranker smoke did not recover
  quality (MRR@10 0.211, Hit@10 20/50). The next target is better candidate
  generation and memory-aware scoring before reranking.

## Guard Policy

- `.github/workflows/public-scale.yml` runs weekly/manual FiQA 10k and TREC-COVID 50k staged smokes.
- FiQA 25k/full and TREC-COVID 100k/full remain manual checks because they are multi-minute runs and depend on ignored local benchmark data.
- MS MARCO passage is the manual large tier: the downloader writes metadata JSON plus a gitignored corpus JSONL shard so 100k/1M/5M/full scale can be tested without committing giant artifacts.
- If 5M/full quality needs to recover toward the 1M/100k tier, the next target
  is semantic candidate generation, memory-aware scoring, or graph/anchor
  recall before cross-reranking.

## Remote Guard Dispatch

After PR #101 merged, `Public Scale Guard` was manually dispatched on `main`:

- Run: https://github.com/PlateerLab/synaptic-memory/actions/runs/28560097957
- Result: success
- Duration: 53s
- FiQA 10k: build 1.2s, search 0.2s, MRR@10 0.425, Hit@10 3/5
- TREC-COVID 50k: build 8.2s, search 1.9s, MRR@10 0.933, Hit@10 10/10

The dispatch verifies that the scheduled/manual guard is visible on the default
branch, downloads ignored public benchmark JSONs, enforces thresholds, and
uploads both logs and markdown artifacts.
