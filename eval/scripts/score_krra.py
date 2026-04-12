"""Score Synaptic Memory KRRA retrieval against seed GT queries.

Loads ``eval/data/queries/krra.json``, runs ``graph.search()`` for each
query on the chosen backend, maps chunk/document hits back to doc_ids,
and reports IR metrics (MRR, nDCG, P@K, R@K) via
``tests.benchmark.metrics.BenchmarkResult``.

The script is backend-agnostic — pass ``--backend sqlite`` (default) or
``--backend kuzu`` depending on which backend you ingested into. Graph
path defaults follow the same convention as ``ingest_krra.py``.

Usage::

    # Default: SQLite graph (new v0.12 default)
    uv run python eval/scripts/score_krra.py

    # Explicit backend + path
    uv run python eval/scripts/score_krra.py --backend kuzu
    uv run python eval/scripts/score_krra.py \\
        --backend sqlite --graph eval/data/krra_graph.sqlite

Prerequisites:
    - ``eval/data/parsed/krra/`` exists (from parse_krra.py)
    - A graph file ingested via ``ingest_krra.py`` with matching backend
    - ``eval/data/queries/krra.json`` exists (seed GT)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.graph import SynapticGraph  # noqa: E402
from tests.benchmark.metrics import BenchmarkResult  # noqa: E402

DEFAULT_QUERIES = REPO_ROOT / "eval" / "data" / "queries" / "krra.json"
DEFAULT_SQLITE_GRAPH = REPO_ROOT / "eval" / "data" / "krra_graph.sqlite"
DEFAULT_KUZU_GRAPH = REPO_ROOT / "eval" / "data" / "krra_graph.kuzu"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

K = 10
SEARCH_POOL = 50  # raw hits before dedup to doc_ids


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if s else s


def _hits_to_doc_ids(search_result: object, limit: int) -> list[str]:
    """Convert a ``SearchResult``'s nodes → unique ordered doc_ids.

    Each hit is either a Document node or a Chunk node; both carry a
    ``doc_id`` property set during ingestion. We preserve the rank order
    of first appearance so MRR/nDCG reflect "which document surfaces
    first".
    """
    seen: list[str] = []
    nodes = getattr(search_result, "nodes", [])
    for hit in nodes:
        node = getattr(hit, "node", hit)
        props = getattr(node, "properties", None) or {}
        doc_id = props.get("doc_id")
        if not doc_id:
            continue
        if doc_id in seen:
            continue
        seen.append(doc_id)
        if len(seen) >= limit:
            break
    return seen


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backend",
        choices=["sqlite", "kuzu"],
        default="sqlite",
        help="Graph backend to score against (default: sqlite)",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Graph file path. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES,
        help="GT queries JSON path",
    )
    return parser.parse_args()


async def _open_backend(backend_name: str, graph_path: Path):
    if backend_name == "sqlite":
        from synaptic.backends.sqlite_graph import SqliteGraphBackend

        backend = SqliteGraphBackend(str(graph_path))
        await backend.connect()
        return backend
    if backend_name == "kuzu":
        from synaptic.backends.kuzu import KuzuBackend

        backend = KuzuBackend(str(graph_path))
        await backend.connect()
        return backend
    msg = f"Unknown backend: {backend_name}"
    raise ValueError(msg)


async def main() -> int:
    args = _parse_args()

    graph_path = args.graph or (
        DEFAULT_SQLITE_GRAPH if args.backend == "sqlite" else DEFAULT_KUZU_GRAPH
    )

    if not graph_path.exists():
        print(f"ERROR: Graph not found at {graph_path}.")
        print(f"Run: uv run python eval/scripts/ingest_krra.py --backend {args.backend}")
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

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    print(f"Loaded {len(queries)} GT queries from {_rel(args.queries)}")
    print(f"Backend: {args.backend}  Graph: {_rel(graph_path)}")

    backend = await _open_backend(args.backend, graph_path)
    graph = SynapticGraph(backend)

    bench = BenchmarkResult()
    skipped = 0

    for q in queries:
        qid = q["qid"]
        query_text = _nfc(q["query"])
        relevant = {doc_id for doc_id in q.get("relevant_docs", [])}

        if not relevant:
            skipped += 1
            continue

        t0 = time.perf_counter()
        result = await graph.search(query_text, limit=SEARCH_POOL)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        retrieved = _hits_to_doc_ids(result, limit=K)
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
    out_path = RESULTS_DIR / f"krra_baseline_{args.backend}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "krra",
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
    print(f"\nResults → {_rel(out_path)}")

    await backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
