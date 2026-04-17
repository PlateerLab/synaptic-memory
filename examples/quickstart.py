"""Synaptic Memory — 5-minute quickstart.

Copy-paste this file, run it, and you have a working knowledge graph with
hybrid (lexical + vector + graph) retrieval over a small product catalog.

Prerequisites
-------------
Install with the recommended extras (one command)::

    pip install "synaptic-memory[sqlite,korean,vector]"

Or with uv::

    uv pip install "synaptic-memory[sqlite,korean,vector]"

Run::

    python examples/quickstart.py

What it does
------------
1. Ingests ``examples/data/products.csv`` (10 rows) into a SQLite-backed
   knowledge graph at ``quickstart.db``.
2. Runs three searches showing hybrid retrieval — FTS + usearch HNSW
   vector search + graph-aware reranking.
3. Prints the top result for each query with its activation score.

No LLM is called at any point — indexing is free, search is local.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from synaptic import SynapticGraph

DATA_PATH = Path(__file__).parent / "data" / "products.csv"
DB_PATH = "quickstart.db"


async def main() -> None:
    # Build the graph. Auto-detects CSV, generates a DomainProfile,
    # ingests each row as an ENTITY node, and indexes it.
    graph = await SynapticGraph.from_data(str(DATA_PATH), db=DB_PATH)

    try:
        stats = await graph.stats()
        print(f"Ingested: {stats.get('total_nodes', 0)} nodes\n")

        queries = [
            "laptop with long battery",
            "spicy Korean noodles",
            "facial skincare mask",
        ]

        for query in queries:
            print(f"Query: {query!r}")
            # engine="evidence" selects the 3rd-gen retrieval pipeline
            # (BM25 + HNSW + PPR + cross-encoder + MMR). Default flips
            # to "evidence" in v0.16.0.
            result = await graph.search(query, limit=3, engine="evidence")

            for i, activated in enumerate(result.nodes[:3], 1):
                node = activated.node
                name = node.properties.get("name", node.title or node.id)
                print(f"  {i}. {name:<30}  score={activated.activation:.3f}")
            print()
    finally:
        # Close the backend connection cleanly so aiosqlite's worker
        # thread exits before the event loop does.
        close = getattr(graph._backend, "close", None)
        if close is not None:
            await close()


if __name__ == "__main__":
    asyncio.run(main())
