"""Tests for the auto graph snapshot module."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.models import ConsolidationLevel, Edge, EdgeKind, Node, NodeKind
from synaptic.snapshot import (
    SnapshotStats,
    collect_stats,
    generate_snapshot,
    render_markdown,
)


def test_render_empty_stats_still_produces_valid_markdown():
    """No data → header + Scale section only, no crash, no empty headings."""
    md = render_markdown(SnapshotStats())
    assert md.startswith("# Knowledge Graph Snapshot")
    assert "## Scale" in md
    # Sections that have nothing to show should be omitted
    assert "## Categories" not in md
    assert "## Tables" not in md
    assert "## Top phrase hubs" not in md


def test_render_with_categories_and_hubs():
    stats = SnapshotStats(
        n_documents=5,
        n_chunks=50,
        n_entities_phrase=10,
        n_categories=2,
        categories=[("Cat A", 3), ("Cat B", 2)],
        top_phrase_hubs=[("entity X", 7), ("entity Y", 4)],
        edges_by_kind={"contains": 12, "mentions": 8},
        n_edges_total=20,
    )
    md = render_markdown(stats)
    assert "**Documents**: 5" in md
    assert "## Categories" in md
    assert "Cat A (3 docs)" in md
    assert "## Top phrase hubs" in md
    assert "entity X (7 mentions)" in md
    assert "## Edge types (sampled)" in md
    assert "``contains`` (12)" in md


def test_render_omits_sample_queries_when_disabled():
    stats = SnapshotStats(
        tables={"products": 100},
        top_phrase_hubs=[("widget", 5)],
        categories=[("Cat A", 2)],
    )
    md = render_markdown(stats, include_sample_queries=False)
    assert "## Sample queries" not in md


def test_render_uses_custom_title():
    md = render_markdown(SnapshotStats(), title="My Corpus Overview")
    assert md.startswith("# My Corpus Overview")


def test_render_table_section_when_tables_present():
    stats = SnapshotStats(
        n_entities_structured=200,
        tables={"products": 100, "reviews": 80},
    )
    md = render_markdown(stats)
    assert "## Tables" in md
    assert "``products`` (100 rows)" in md
    assert "``reviews`` (80 rows)" in md


@pytest.mark.asyncio
async def test_collect_stats_on_empty_backend():
    """Empty graph → all-zero stats, no crash."""
    backend = MemoryBackend()
    await backend.connect()
    stats = await collect_stats(backend)
    assert stats.n_documents == 0
    assert stats.n_chunks == 0
    assert stats.n_categories == 0
    assert stats.n_entities_phrase == 0
    assert not stats.top_phrase_hubs
    assert not stats.tables


@pytest.mark.asyncio
async def test_collect_stats_with_documents_and_chunks():
    backend = MemoryBackend()
    await backend.connect()
    for i in range(3):
        await backend.save_node(
            Node(
                id=f"rule_{i}",
                kind=NodeKind.RULE,
                title=f"Rule {i}",
                content="...",
                level=ConsolidationLevel.L0_RAW,
            )
        )
    for i in range(5):
        await backend.save_node(
            Node(
                id=f"chunk_{i}",
                kind=NodeKind.CHUNK,
                title=f"Chunk {i}",
                content="text",
                level=ConsolidationLevel.L0_RAW,
            )
        )
    stats = await collect_stats(backend)
    assert stats.n_documents == 3
    assert stats.n_chunks == 5


@pytest.mark.asyncio
async def test_collect_stats_categories_with_doc_counts():
    backend = MemoryBackend()
    await backend.connect()
    cat = Node(
        id="cat_alpha",
        kind=NodeKind.CONCEPT,
        title="Alpha Category",
        tags=["category"],
        level=ConsolidationLevel.L0_RAW,
    )
    await backend.save_node(cat)
    for i in range(4):
        d = Node(
            id=f"doc_{i}",
            kind=NodeKind.RULE,
            title=f"Doc {i}",
            level=ConsolidationLevel.L0_RAW,
        )
        await backend.save_node(d)
        await backend.save_edge(Edge(source_id=d.id, target_id=cat.id, kind=EdgeKind.PART_OF))
    stats = await collect_stats(backend)
    assert stats.n_categories == 1
    assert stats.categories == [("Alpha Category", 4)]


@pytest.mark.asyncio
async def test_collect_stats_separates_phrase_from_structured_entities():
    """Entities with ``_table_name`` property → structured rows.
    Entities without → phrase hubs."""
    backend = MemoryBackend()
    await backend.connect()
    for i in range(3):
        await backend.save_node(
            Node(
                id=f"row_{i}",
                kind=NodeKind.ENTITY,
                title=f"Row {i}",
                properties={"_table_name": "products", "name": f"item_{i}"},
                level=ConsolidationLevel.L0_RAW,
            )
        )
    for i in range(2):
        await backend.save_node(
            Node(
                id=f"phrase_{i}",
                kind=NodeKind.ENTITY,
                title=f"phrase {i}",
                level=ConsolidationLevel.L0_RAW,
            )
        )
    stats = await collect_stats(backend)
    assert stats.n_entities_structured == 3
    assert stats.n_entities_phrase == 2
    assert stats.tables == {"products": 3}


@pytest.mark.asyncio
async def test_collect_stats_top_phrase_hubs_ranked_by_mentions():
    backend = MemoryBackend()
    await backend.connect()
    chunk = Node(
        id="chunk_a",
        kind=NodeKind.CHUNK,
        title="Chunk A",
        level=ConsolidationLevel.L0_RAW,
    )
    await backend.save_node(chunk)

    # hub_high gets 3 mentions, hub_low gets 1
    high = Node(
        id="phrase_high",
        kind=NodeKind.ENTITY,
        title="popular term",
        level=ConsolidationLevel.L0_RAW,
    )
    low = Node(
        id="phrase_low",
        kind=NodeKind.ENTITY,
        title="rare term",
        level=ConsolidationLevel.L0_RAW,
    )
    await backend.save_node(high)
    await backend.save_node(low)
    for _ in range(3):
        await backend.save_edge(Edge(source_id=chunk.id, target_id=high.id, kind=EdgeKind.MENTIONS))
    await backend.save_edge(Edge(source_id=chunk.id, target_id=low.id, kind=EdgeKind.MENTIONS))

    stats = await collect_stats(backend)
    assert stats.top_phrase_hubs is not None
    assert stats.top_phrase_hubs[0] == ("popular term", 3)


@pytest.mark.asyncio
async def test_generate_snapshot_end_to_end_returns_nonempty_markdown():
    backend = MemoryBackend()
    await backend.connect()
    await backend.save_node(
        Node(
            id="rule_1",
            kind=NodeKind.RULE,
            title="Sample doc",
            level=ConsolidationLevel.L0_RAW,
        )
    )
    md = await generate_snapshot(backend)
    assert md.startswith("# Knowledge Graph Snapshot")
    assert "**Documents**: 1" in md
