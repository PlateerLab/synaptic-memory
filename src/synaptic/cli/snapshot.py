"""``synaptic-snapshot`` CLI — generate a markdown summary of a graph.

Usage::

    synaptic-snapshot path/to/graph.sqlite
    synaptic-snapshot path/to/graph.sqlite --output report.md
    synaptic-snapshot path/to/graph.sqlite --max-entities 20000

The output is the same markdown the ``knowledge_snapshot`` MCP tool and
``graph.chat()``'s priming path emit. Use it to preview "what does my
graph look like" without spinning up an agent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from synaptic import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synaptic-snapshot",
        description=(
            "Generate a markdown snapshot of a Synaptic Memory graph — "
            "scale, categories, top phrase hubs, structured tables, edge "
            "kinds, and sample query hints."
        ),
    )
    p.add_argument("db", help="Path to the SQLite graph file (or :memory: for ephemeral)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the markdown to this file. Default: stdout.",
    )
    p.add_argument(
        "--max-entities",
        type=int,
        default=5_000,
        help="Cap on entity scan (default 5000). Higher = more accurate phrase-hub ranking, slower.",
    )
    p.add_argument(
        "--top-phrase-hubs",
        type=int,
        default=15,
        help="Number of phrase hubs to surface (default 15).",
    )
    p.add_argument(
        "--top-categories",
        type=int,
        default=30,
        help="Number of categories to list (default 30).",
    )
    p.add_argument(
        "--no-sample-queries",
        action="store_true",
        help="Omit the sample-queries section.",
    )
    p.add_argument(
        "--title",
        default="Knowledge Graph Snapshot",
        help="H1 heading title for the report.",
    )
    p.add_argument("--version", action="version", version=f"synaptic-snapshot {__version__}")
    return p


async def _run(args: argparse.Namespace) -> str:
    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.snapshot import generate_snapshot

    backend = SqliteGraphBackend(args.db)
    await backend.connect()
    try:
        return await generate_snapshot(
            backend,
            max_entities_scanned=args.max_entities,
            top_n_phrase_hubs=args.top_phrase_hubs,
            top_n_categories=args.top_categories,
            include_sample_queries=not args.no_sample_queries,
            title=args.title,
        )
    finally:
        # Best-effort close — some backends raise during shutdown.
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not Path(args.db).exists() and args.db != ":memory:":
        print(f"error: graph file not found: {args.db}", file=sys.stderr)
        return 2

    md = asyncio.run(_run(args))

    if args.output is None:
        sys.stdout.write(md)
    else:
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote snapshot ({len(md)} chars) to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
