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
│   └── kuzu_parity.py   # Memory vs Kuzu parity on enterprise scenario
├── data/
│   ├── raw/             # Raw documents (gitignored)
│   │   └── krra/        # KRRA 마사회 corpus
│   ├── parsed/          # Cached xgen-doc2chunk output (gitignored)
│   └── queries/         # Hand-crafted ground truth (committed)
└── results/             # Benchmark run outputs (gitignored)
```

## Running the parity check

```bash
uv run python eval/scripts/kuzu_parity.py
```

Expected output: side-by-side MRR/nDCG/Recall/Latency for MemoryBackend
and KuzuBackend on the 15-query enterprise scenario. The two columns
should match within noise — anything else is a regression.
