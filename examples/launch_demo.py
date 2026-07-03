"""Launch video demo for Synaptic Memory.

This script is intentionally small and dependency-light so it can be recorded
as a 60-second terminal demo:

    uv run python examples/launch_demo.py

It demonstrates the launch message:
- mixed documents + structured rows
- deterministic, LLM-free default indexing
- search with event recording
- feedback + memory health metadata
"""

from __future__ import annotations

import argparse
import asyncio
import time

from synaptic import FeedbackSignal, MemoryScope, NodeKind, SynapticGraph
from synaptic.extensions.table_ingester import TableIngester


def _pause(enabled: bool, seconds: float = 0.9) -> None:
    if enabled:
        time.sleep(seconds)


def _section(title: str) -> None:
    print(f"\n# {title}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run without recording-friendly pauses.",
    )
    args = parser.parse_args()
    pause = not args.no_pause

    print("Synaptic Memory")
    print("Graph memory for RAG agents: docs + SQL rows + feedback")
    print("Default path: no LLM calls at indexing time")
    _pause(pause, 1.1)

    graph = SynapticGraph.memory()
    await graph.connect()

    try:
        _section("1. Build a tiny mixed graph")
        await graph.add(
            "Refund Policy",
            "Refund exceptions require manager approval. VIP customers may receive "
            "expedited handling when the original order is delayed.",
            kind=NodeKind.RULE,
            source="policy/refunds.md",
            properties={"source": "policy/refunds.md", "page": "1"},
        )
        await graph.add(
            "Shipping Policy",
            "Delayed orders over 7 days should be escalated before a refund is denied.",
            kind=NodeKind.RULE,
            source="policy/shipping.md",
            properties={"source": "policy/shipping.md", "page": "2"},
        )

        await TableIngester().ingest(
            graph,
            "ticket",
            columns=[
                {"name": "id", "type": "str"},
                {"name": "customer", "type": "str"},
                {"name": "tier", "type": "str"},
                {"name": "issue", "type": "str"},
                {"name": "days_delayed", "type": "int"},
            ],
            rows=[
                {
                    "id": "T-1001",
                    "customer": "Acme Korea",
                    "tier": "VIP",
                    "issue": "refund exception for delayed shipment",
                    "days_delayed": "9",
                },
                {
                    "id": "T-1002",
                    "customer": "Blue Shop",
                    "tier": "standard",
                    "issue": "address change request",
                    "days_delayed": "1",
                },
            ],
            primary_key="id",
            source_url="demo://support",
        )

        stats = await graph.stats()
        print(f"Created graph: {stats.get('total_nodes', 0)} nodes")
        print("Sources: 2 policy docs + 2 support ticket rows")
        _pause(pause)

        _section("2. Search and record retrieval")
        scope = MemoryScope(workspace_id="launch-demo", user_id="support-agent")
        result = await graph.search(
            "VIP refund exception delayed shipment",
            limit=4,
            record=True,
            scope=scope,
        )
        print(f"event_id: {result.event_id}")
        for idx, activated in enumerate(result.nodes[:4], 1):
            node = activated.node
            table = node.properties.get("_table_name", "document")
            source = node.properties.get("source", node.source or table)
            print(f"{idx}. {node.title:<28} score={activated.activation:.3f} source={source}")
        _pause(pause)

        _section("3. Feed back that the evidence helped")
        top_node_ids = [activated.node.id for activated in result.nodes[:2]]
        await graph.record_feedback(
            event_id=result.event_id,
            signal=FeedbackSignal.EXPLICIT_POSITIVE,
            success=True,
            node_ids=top_node_ids,
            scope=scope,
        )
        print("Recorded explicit positive feedback for the top evidence.")
        _pause(pause)

        _section("4. Inspect memory health metadata")
        health = await graph.memory_health(scope=scope, persist_signals=False)
        print(f"memory_events:   {health.memory_events}")
        print(f"retrieval_events:{health.retrieval_events}")
        print(f"memory_scores:   {health.memory_score_count}")
        print(f"health_signals:  {health.signal_count}")
        print("\nNo raw provenance was appended to Node.content.")
        print("Swap the backend to PostgreSQL/Kuzu/Qdrant when the corpus grows.")
    finally:
        await graph.close()


if __name__ == "__main__":
    asyncio.run(main())
