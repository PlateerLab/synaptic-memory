"""Score Synaptic Memory on the assort structured dataset.

Loads ``eval/data/queries/assort.json``, runs ``graph.search()`` for each
query, maps hits to node titles (``table:pk`` format), and reports IR
metrics via ``tests.benchmark.metrics.BenchmarkResult``.

Unlike KRRA (text documents), assort data is relational — each node title
is ``{table_name}:{primary_key}``. The GT ``relevant_docs`` field uses this
same format so scoring maps naturally.

Usage::

    uv run python eval/scripts/score_assort.py
    uv run python eval/scripts/score_assort.py --backend kuzu
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

from synaptic.graph import SynapticGraph
from tests.benchmark.metrics import BenchmarkResult

DEFAULT_QUERIES = REPO_ROOT / "eval" / "data" / "queries" / "assort.json"
DEFAULT_SQLITE = REPO_ROOT / "eval" / "data" / "assort_graph.sqlite"
DEFAULT_KUZU = REPO_ROOT / "eval" / "data" / "assort_graph.kuzu"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

K = 10
SEARCH_POOL = 50


def _hits_to_titles(search_result: object, limit: int) -> list[str]:
    """Convert SearchResult nodes → unique ordered node titles.

    Node title format: ``{table_name}:{pk_value}`` (set by TableIngester).
    This is the same format used in GT ``relevant_docs``.
    """
    seen: list[str] = []
    nodes = getattr(search_result, "nodes", [])
    for hit in nodes:
        node = getattr(hit, "node", hit)
        title = getattr(node, "title", None)
        if not title or title in seen:
            continue
        seen.append(title)
        if len(seen) >= limit:
            break
    return seen


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", choices=["sqlite", "kuzu"], default="sqlite")
    p.add_argument("--graph", type=Path, default=None)
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
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
        print(f"ERROR: Graph not found at {graph_path}.")
        print(f"Run: uv run python eval/scripts/ingest_assort.py --backend {args.backend}")
        return 1
    if not args.queries.exists():
        print(f"ERROR: Queries not found at {args.queries}.")
        return 1

    with open(args.queries, encoding="utf-8") as f:
        gt = json.load(f)

    queries = gt.get("queries", [])
    if not queries:
        print("ERROR: No queries in GT file.")
        return 1

    print(f"Loaded {len(queries)} GT queries from {args.queries.relative_to(REPO_ROOT)}")
    print(f"Backend: {args.backend}  Graph: {graph_path.relative_to(REPO_ROOT)}")

    backend = await _open_backend(args.backend, graph_path)
    graph = SynapticGraph(backend)

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
        result = await graph.search(query_text, limit=SEARCH_POOL)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        retrieved = _hits_to_titles(result, limit=K)
        bench.add(
            query_id=qid,
            query=query_text,
            retrieved=retrieved,
            relevant=relevant,
            k=K,
            description=q.get("description", ""),
            search_time_ms=elapsed_ms,
        )

    print(bench.report(k=K))
    if skipped:
        print(f"\n(Skipped {skipped} queries with empty relevant_docs)")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = int(time.time())
    out_path = RESULTS_DIR / f"assort_baseline_{args.backend}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "assort",
                "backend": args.backend,
                "timestamp": ts,
                "k": K,
                "summary": bench.summary(),
                "queries": bench.queries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults → {out_path.relative_to(REPO_ROOT)}")

    await backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
