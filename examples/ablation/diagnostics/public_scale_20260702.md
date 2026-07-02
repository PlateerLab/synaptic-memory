# Public Large-Corpus Scale Smoke - 2026-07-02

## Datasets

| Dataset | Local artifact | Corpus | Queries | Smoke scope |
|---------|----------------|-------:|--------:|-------------|
| BEIR FiQA test | `tests/benchmark/data/fiqa.json` | 57,638 docs | 648 | 5-10 queries |
| BEIR TREC-COVID test | `tests/benchmark/data/trec_covid.json` | 171,332 docs | 50 | 10 queries |
| BEIR MS MARCO passage dev | `tests/benchmark/data/msmarco_passage.json` + `.corpus.jsonl` | 1M shard by default from ~8.8M source passages | validation qrels | manual large tier |

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

| Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Build | Search |
|-----:|--------:|-------:|----:|-----:|-------:|------:|-------:|
| 100,000 | 50 | 0.673 | 0.740 | 0.770 | 39/50 | 81.9s | 5.4s |

The local artifacts are gitignored:

- `tests/benchmark/data/msmarco_passage.json` - 511 KB manifest
- `tests/benchmark/data/msmarco_passage.corpus.jsonl` - 35 MB corpus shard

## Interpretation

- Search latency remains usable at 171k docs: 5.2s over 10 queries.
- MS MARCO confirms the large-tier path on a web passage corpus: 100k docs,
  50 queries, 5.4s total search, and 0.673 MRR@10 without embeddings or reranking.
- The main large-corpus bottleneck is still initial FTS/index build, not retrieval.
- Avoiding unnecessary FTS deletes for newly inserted nodes reduced full FiQA build time by about 9.9x.
- Raising benchmark ingest batches to 20k reduced full TREC-COVID build time by about 2.7x.
- `--corpus-limit` provides practical staged scale gates while preserving selected query gold docs.

## Guard Policy

- `.github/workflows/public-scale.yml` runs weekly/manual FiQA 10k and TREC-COVID 50k staged smokes.
- FiQA 25k/full and TREC-COVID 100k/full remain manual checks because they are multi-minute runs and depend on ignored local benchmark data.
- MS MARCO passage is the manual large tier: the downloader writes metadata JSON plus a gitignored corpus JSONL shard so 100k/1M/8.8M-style scale can be tested without committing giant artifacts.
- If 100k+ docs becomes a required routine gate, the next target is faster initial FTS/index build.

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
