"""Score KRRA with the 3rd-generation EvidenceSearch pipeline.

Runs the existing 20 seed queries from ``krra.json`` through the full
anchor → expand → rerank → aggregate pipeline and reports the same IR
metrics (MRR, P@K, R@K, nDCG) as ``score_krra.py``. Purpose: verify
the new pipeline doesn't regress on simple single-keyword queries
before using it for cross-category workloads.

Usage::

    uv run python eval/scripts/score_krra_evidence.py
    uv run python eval/scripts/score_krra_evidence.py --graph eval/data/krra_graph.sqlite
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.extensions.evidence_search import EvidenceSearch  # noqa: E402
from tests.benchmark.metrics import BenchmarkResult  # noqa: E402

DEFAULT_QUERIES = REPO_ROOT / "eval" / "data" / "queries" / "krra.json"
DEFAULT_SQLITE = REPO_ROOT / "eval" / "data" / "krra_graph.sqlite"
DEFAULT_KUZU = REPO_ROOT / "eval" / "data" / "krra_graph.kuzu"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

K = 10


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _evidence_to_doc_ids(evidence_list) -> list[str]:
    """Convert Evidence list → unique ordered doc_id list.

    Each evidence has a ``document_id`` property — from the chunk's
    ``properties['doc_id']``. We preserve rank order of first
    appearance so downstream metrics reflect "which doc surfaces
    first".
    """
    seen: list[str] = []
    for ev in evidence_list:
        doc_id = ev.document_id
        if not doc_id or doc_id in seen:
            continue
        seen.append(doc_id)
    return seen


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["sqlite", "kuzu"], default="sqlite")
    p.add_argument("--graph", type=Path, default=None)
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p.add_argument("--k", type=int, default=K)
    return p.parse_args()


async def _open_backend(backend_name: str, graph_path: Path):
    if backend_name == "sqlite":
        from synaptic.backends.sqlite_graph import SqliteGraphBackend
        backend = SqliteGraphBackend(str(graph_path))
        await backend.connect()
        return backend
    from synaptic.backends.kuzu import KuzuBackend
    backend = KuzuBackend(str(graph_path))
    await backend.connect()
    return backend


async def main() -> int:
    args = _parse_args()
    graph_path = args.graph or (DEFAULT_SQLITE if args.backend == "sqlite" else DEFAULT_KUZU)

    if not graph_path.exists():
        print(f"ERROR: graph not found: {graph_path}")
        return 1
    if not args.queries.exists():
        print(f"ERROR: queries not found: {args.queries}")
        return 1

    with open(args.queries, encoding="utf-8") as f:
        gt = json.load(f)
    queries = gt.get("queries", [])
    if not queries:
        print("ERROR: no queries in GT")
        return 1

    print(f"Loaded {len(queries)} GT queries from {_rel(args.queries)}")
    print(f"Backend: {args.backend}  Graph: {_rel(graph_path)}")

    backend = await _open_backend(args.backend, graph_path)

    # 3rd-gen pipeline — no phrase extractor (rule-based anchors only),
    # no query embedding (pure lexical + graph + structural)
    searcher = EvidenceSearch(backend=backend)

    bench = BenchmarkResult()
    skipped = 0

    for q in queries:
        qid = q["qid"]
        query_text = q["query"]
        relevant = set(q.get("relevant_docs", []))
        if not relevant:
            skipped += 1
            continue

        t0 = time.perf_counter()
        result = await searcher.search(
            query_text,
            k=args.k * 2,   # aggregator cap; then we cut to k at scoring
            fts_seed_limit=30,
            per_document_cap=2,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        retrieved = _evidence_to_doc_ids(result.evidence)[: args.k]
        bench.add(
            query_id=qid,
            query=query_text,
            retrieved=retrieved,
            relevant=relevant,
            k=args.k,
            description=q.get("description", ""),
            search_time_ms=elapsed_ms,
        )

    print(bench.report(k=args.k))
    if skipped:
        print(f"\n(Skipped {skipped} queries with empty relevant_docs)")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = int(time.time())
    out_path = RESULTS_DIR / f"krra_evidence_{args.backend}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "krra",
                "pipeline": "evidence_search_v1",
                "backend": args.backend,
                "timestamp": ts,
                "k": args.k,
                "summary": bench.summary(),
                "queries": bench.queries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults → {_rel(out_path)}")

    await backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
