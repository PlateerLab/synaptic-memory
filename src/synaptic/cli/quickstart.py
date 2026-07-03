"""``synaptic-quickstart`` CLI — build and query a tiny graph immediately."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from synaptic import __version__
from synaptic.models import NodeKind
from synaptic.options import DEFAULT_EMBED_MODEL, SynapticPreset

SAMPLE_PRODUCTS: tuple[dict[str, str], ...] = (
    {
        "name": "LunaBook Air 14",
        "category": "laptop",
        "description": "Lightweight laptop with long battery life for business travel.",
    },
    {
        "name": "Gochu Ramyun Kit",
        "category": "food",
        "description": "Spicy Korean noodles with gochujang broth and dried vegetables.",
    },
    {
        "name": "Cica Recovery Mask",
        "category": "skincare",
        "description": "Facial skincare sheet mask for calming sensitive skin.",
    },
    {
        "name": "Aurora Desk Lamp",
        "category": "office",
        "description": "Adjustable LED desk lamp with warm light and USB-C charging.",
    },
    {
        "name": "TrailBottle Steel",
        "category": "outdoor",
        "description": "Insulated stainless bottle that keeps drinks cold during hikes.",
    },
)

DEFAULT_QUERIES = (
    "laptop with long battery",
    "spicy Korean noodles",
    "facial skincare mask",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synaptic-quickstart",
        description="Build a tiny Synaptic Memory graph and run a few searches.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Persist the quickstart graph to this SQLite file. Requires the sqlite extra.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Persist to ./quickstart.db instead of using the zero-dependency in-memory path.",
    )
    parser.add_argument(
        "--preset",
        choices=[preset.value for preset in SynapticPreset],
        default=SynapticPreset.LOCAL.value,
        help="Build preset for endpoint wiring (default: local).",
    )
    parser.add_argument(
        "--embed-url",
        default=None,
        help="OpenAI-compatible embedding base URL. Overrides preset/env.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help=f"Embedding model name (default: {DEFAULT_EMBED_MODEL}).",
    )
    parser.add_argument(
        "--rerank-url",
        default=None,
        help="Optional cross-encoder reranker base URL. Overrides preset/env.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=None,
        help="Query to run. Repeat for multiple queries. Defaults to three sample queries.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON for smoke tests.",
    )
    parser.add_argument("--version", action="version", version=f"synaptic-quickstart {__version__}")
    return parser


async def _build_memory_graph() -> Any:
    from synaptic import SynapticGraph

    graph = SynapticGraph.memory()
    await graph.connect()
    for product in SAMPLE_PRODUCTS:
        await graph.add(
            product["name"],
            product["description"],
            kind=NodeKind.ENTITY,
            source="synaptic-quickstart",
            properties=dict(product),
        )
    return graph


async def _build_sqlite_graph(args: argparse.Namespace, db_path: Path) -> Any:
    from synaptic import SynapticGraph

    with tempfile.TemporaryDirectory(prefix="synaptic-quickstart-") as tmpdir:
        csv_path = Path(tmpdir) / "products.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["name", "category", "description"])
            writer.writeheader()
            writer.writerows(SAMPLE_PRODUCTS)
        return await SynapticGraph.from_data(
            str(csv_path),
            db=str(db_path),
            preset=args.preset,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            rerank_url=args.rerank_url,
        )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    db_path = args.db or (Path.cwd() / "quickstart.db" if args.keep_db else None)
    graph = (
        await _build_sqlite_graph(args, db_path)
        if db_path is not None
        else await _build_memory_graph()
    )

    try:
        stats = await graph.stats()
        queries = args.queries or list(DEFAULT_QUERIES)
        query_results: list[dict[str, Any]] = []
        for query in queries:
            result = await graph.search(query, limit=3)
            hits = []
            for activated in result.nodes[:3]:
                node = activated.node
                hits.append(
                    {
                        "id": node.id,
                        "title": node.properties.get("name") or node.title,
                        "score": round(float(activated.activation), 6),
                        "category": node.properties.get("category", ""),
                    }
                )
            query_results.append({"query": query, "hits": hits})

        return {
            "backend": "sqlite" if db_path is not None else "memory",
            "db": str(db_path) if db_path is not None else "",
            "node_count": int(stats.get("total_nodes", 0)),
            "queries": query_results,
        }
    finally:
        await graph.close()


def _print_text(payload: dict[str, Any]) -> None:
    backend = payload["backend"]
    db = f" ({payload['db']})" if payload.get("db") else ""
    print(f"Synaptic quickstart ready: {payload['node_count']} nodes via {backend}{db}\n")
    for item in payload["queries"]:
        print(f"Query: {item['query']!r}")
        for idx, hit in enumerate(item["hits"], 1):
            category = f" [{hit['category']}]" if hit.get("category") else ""
            print(f"  {idx}. {hit['title']}{category}  score={hit['score']:.3f}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = asyncio.run(_run(args))
    except ModuleNotFoundError as exc:
        if exc.name == "aiosqlite":
            print(
                "synaptic-quickstart --db requires the sqlite extra.\n"
                'Install with: pip install "synaptic-memory[sqlite]"',
                file=sys.stderr,
            )
            return 2
        raise

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _print_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
