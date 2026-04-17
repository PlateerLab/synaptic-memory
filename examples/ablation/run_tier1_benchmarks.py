"""Tier-1 English multi-hop benchmark runner.

Runs Synaptic's embedder-free retrieval pipeline over three standard
multi-hop corpora: HotPotQA-dev (full), MuSiQue-Ans-dev, and
2WikiMultiHopQA-dev. These are the datasets HippoRAG2, GraphRAG, and
the broader KG-RAG line use for head-to-head comparisons.

Prerequisite
------------
Download the datasets first::

    pip install datasets
    python examples/ablation/download_benchmarks.py

Usage
-----
::

    python examples/ablation/run_tier1_benchmarks.py
    # Or just one:
    python examples/ablation/run_tier1_benchmarks.py --only hotpotqa
    # Subset for quick smoke testing:
    python examples/ablation/run_tier1_benchmarks.py --subset 200

The JSON input files are gitignored (``tests/benchmark/data/*.json``);
the download script is the source of truth. This runner prints a
results table and writes ``examples/ablation/diagnostics/tier1_<ts>.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "tests" / "benchmark" / "data"
OUT_DIR = Path(__file__).parent / "diagnostics"

TOP_K = 10


@dataclass
class Dataset:
    name: str
    path: Path
    reference: str  # what prior published number to contextualise against


DATASETS = [
    Dataset(
        name="HotPotQA dev (full)",
        path=BENCH / "hotpotqa_full.json",
        reference="HippoRAG2: 56.7 % string accuracy",
    ),
    Dataset(
        name="MuSiQue-Ans dev",
        path=BENCH / "musique_dev.json",
        reference="HippoRAG2: F1 51.9, R@5 74.7 %",
    ),
    Dataset(
        name="2WikiMultihopQA dev",
        path=BENCH / "2wiki_dev.json",
        reference="HippoRAG2: R@5 90.4 %",
    ),
]


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, did in enumerate(retrieved):
        if did in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


@dataclass
class Report:
    name: str
    n_docs: int
    n_queries: int
    mrr: float
    recall_at_5: float
    recall_at_10: float
    hit_at_10: int
    build_sec: float
    search_sec: float
    reference: str


async def run_one(ds: Dataset, subset: int | None) -> Report:
    if not ds.path.exists():
        raise FileNotFoundError(
            f"{ds.path} missing. Run:  python examples/ablation/download_benchmarks.py"
        )
    with open(ds.path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = data["corpus"]
    queries_all = data["queries"]
    qrels = data["qrels"]

    query_items = list(queries_all.items())
    if subset is not None and subset < len(query_items):
        query_items = query_items[:subset]

    # Build the graph once for the whole dataset.
    t_build = time.perf_counter()
    backend = MemoryBackend()
    await backend.connect()
    graph = SynapticGraph(backend)
    for doc_id, doc in corpus.items():
        title = str(doc.get("title", "") or doc_id)
        text = str(doc.get("text", ""))
        if text or title:
            await graph.add(
                title=title,
                content=text,
                properties={"doc_id": doc_id},
            )
    build_sec = time.perf_counter() - t_build

    mrr_total = 0.0
    r5_total = 0.0
    r10_total = 0.0
    hit10 = 0

    t_search = time.perf_counter()
    for qid, qtext in query_items:
        rel = qrels.get(qid, {})
        relevant = set(rel.keys()) if isinstance(rel, dict) else set(map(str, rel))
        if not relevant:
            continue
        result = await graph.search(str(qtext), limit=TOP_K * 2)
        retrieved: list[str] = []
        for hit in result.nodes:
            did = (hit.node.properties or {}).get("doc_id", "")
            if did and did not in retrieved:
                retrieved.append(did)
        rr = _reciprocal_rank(retrieved[:TOP_K], relevant)
        mrr_total += rr
        r5_total += _recall_at_k(retrieved, relevant, 5)
        r10_total += _recall_at_k(retrieved, relevant, TOP_K)
        if rr > 0:
            hit10 += 1
    search_sec = time.perf_counter() - t_search

    n = max(len(query_items), 1)
    return Report(
        name=ds.name,
        n_docs=len(corpus),
        n_queries=len(query_items),
        mrr=mrr_total / n,
        recall_at_5=r5_total / n,
        recall_at_10=r10_total / n,
        hit_at_10=hit10,
        build_sec=build_sec,
        search_sec=search_sec,
        reference=ds.reference,
    )


def _emit_markdown(reports: list[Report], subset: int | None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"tier1_{stamp}.md"
    lines = [
        "# Tier-1 English multi-hop benchmark — Synaptic v0.16.0",
        "",
        f"- Run at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Subset: {subset if subset else 'full'}",
        "- Engine: `graph.search()` default (EvidenceSearch, no embedder, no cross-encoder)",
        "",
        "| Dataset | Docs | Queries | MRR@10 | R@5 | R@10 | Hit@10 | Build | Search |",
        "|---------|-----:|--------:|-------:|----:|-----:|-------:|------:|-------:|",
    ]
    for r in reports:
        lines.append(
            f"| {r.name} | {r.n_docs} | {r.n_queries} | "
            f"{r.mrr:.3f} | {r.recall_at_5:.3f} | {r.recall_at_10:.3f} | "
            f"{r.hit_at_10}/{r.n_queries} | {r.build_sec:.1f}s | {r.search_sec:.1f}s |"
        )
    lines.append("")
    lines.append("## Context")
    lines.append("")
    for r in reports:
        lines.append(f"- **{r.name}** — published baseline: {r.reference}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def amain(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        default=",".join(["hotpotqa", "musique", "2wiki"]),
        help="comma-separated dataset keys (hotpotqa | musique | 2wiki)",
    )
    p.add_argument("--subset", type=int, default=None)
    args = p.parse_args(argv)

    by_key = {
        "hotpotqa": DATASETS[0],
        "musique": DATASETS[1],
        "2wiki": DATASETS[2],
    }
    selected = [by_key[k.strip()] for k in args.only.split(",") if k.strip()]

    print("Tier-1 multi-hop English benchmarks — Synaptic v0.16.0 embedder-free")
    print()
    header = f"{'Dataset':<24} {'Docs':>7} {'Qs':>6} {'MRR@10':>8} {'R@5':>7} {'R@10':>7} {'Hit':>10} {'Build':>7} {'Search':>8}"
    print(header)
    print("-" * len(header))

    reports: list[Report] = []
    for ds in selected:
        try:
            r = await run_one(ds, args.subset)
        except FileNotFoundError as e:
            print(f"{ds.name:<24}  SKIP — {e}")
            continue
        reports.append(r)
        print(
            f"{r.name:<24} {r.n_docs:>7} {r.n_queries:>6} "
            f"{r.mrr:>8.3f} {r.recall_at_5:>7.3f} {r.recall_at_10:>7.3f} "
            f"{r.hit_at_10:>5}/{r.n_queries:<4} {r.build_sec:>6.1f}s {r.search_sec:>7.1f}s"
        )

    if reports:
        out = _emit_markdown(reports, args.subset)
        print()
        print(f"Markdown report → {out.relative_to(REPO_ROOT)}")
    return 0


def main() -> None:
    import sys as _sys

    _sys.exit(asyncio.run(amain(_sys.argv[1:])))


if __name__ == "__main__":
    main()
