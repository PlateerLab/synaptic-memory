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
