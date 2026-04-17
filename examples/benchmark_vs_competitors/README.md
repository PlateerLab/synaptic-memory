# Head-to-head RAG comparison harness

A common protocol and a set of adapters for running **the same corpus
and the same queries** through Synaptic, Mem0, Cognee, and HippoRAG2,
then comparing the numbers side-by-side.

This harness exists because self-reported benchmark numbers have a
credibility problem in the agent-memory space (see the LoCoMo-Zep
incident, 2025). The only honest comparison is one you can reproduce
— so the adapters here are deliberately thin, the input format is
BEIR-style, and the metrics are standard IR (MRR, Recall@k, hit
rate).

## What's here

```
benchmark_vs_competitors/
├── README.md                 # this file
├── protocol.py               # common corpus/query/result types + metrics
├── adapters/
│   ├── __init__.py
│   ├── base.py               # Adapter ABC — what each system implements
│   ├── synaptic.py           # FTS-only (no LLM)
│   ├── mem0.py               # Mem0 (LLM required — OpenAI / Anthropic / Ollama)
│   ├── cognee.py             # Cognee (LLM required)
│   └── hipporag.py           # HippoRAG2 (LLM required)
├── run_comparison.py         # driver — runs all adapters, prints table
└── results/                  # run outputs (gitignored)
```

## Run

### Synaptic only (no LLM, ~2s)

```bash
python examples/benchmark_vs_competitors/run_comparison.py --only synaptic
```

### Full comparison (needs API key)

```bash
# Mem0 + Cognee pick up OPENAI_API_KEY by default. Set LLM_PROVIDER=anthropic
# to route through Claude via LiteLLM (ANTHROPIC_API_KEY must be set).
export OPENAI_API_KEY=sk-...

# Run a small POC subset first — full runs can take 30+ minutes and
# cost a few dollars in API calls
python examples/benchmark_vs_competitors/run_comparison.py --subset 10

# Full run (all 200 Allganize RAG-ko queries)
python examples/benchmark_vs_competitors/run_comparison.py
```

The comparison table is written to
`results/comparison_<timestamp>.md` and also printed to stdout.

## Fairness notes

This harness tries to make the comparison fair, but exact parity is
impossible because the systems have different design philosophies:

* **Synaptic runs in FTS-only mode** (no LLM, no embedder) by default
  here. This is a deliberately conservative baseline — adding
  embedder + cross-encoder raises Synaptic's numbers (see
  [examples/benchmark_allganize.py](../benchmark_allganize.py)) but
  then we'd be comparing apples to heavily-infrastructure oranges.
* **Mem0 / Cognee / HippoRAG2 make LLM calls during indexing**
  (entity extraction, relation extraction, community summarization).
  That cost is reflected in the timing column.
* **BEIR-style corpora don't fit Mem0's conversational-memory
  model perfectly.** Mem0 is designed for "user says X → LLM
  remembers X for later conversations." We adapt by treating each
  corpus document as a memory with a unique `user_id`, then querying
  across all users. It works, but Mem0 is being used outside its
  primary use case.
* **All adapters share the same metrics computation** (see
  `protocol.py::score_run`) — there's no per-system metric fudging.

Published numbers from the systems' own papers and blog posts are
collected separately in [docs/comparison/published_numbers.md](../../docs/comparison/published_numbers.md).
Those are useful for context but should NOT be compared directly
to results from this harness (different corpora, different metric
definitions).

## Adding a new system

Implement `adapters.base.Adapter` — three methods:

* `async def build(self, corpus)` — ingest the corpus
* `async def search(self, query, k)` — return top-k doc_ids
* `async def close(self)` — release resources

Then add it to the `ADAPTERS` dict in `run_comparison.py`.
