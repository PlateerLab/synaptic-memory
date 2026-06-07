"""Connectivity backbone — bridge a fragmented graph into one navigable component.

Connectivity is a deterministic graph property, so this is unit-testable without
the GPU/noise floor: a synthetic corpus split into disconnected clusters must
come out as a single component with the minimal number of high-quality bridges.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.connectivity import _cosine, bridge_components
from synaptic.models import ConsolidationLevel, Edge, EdgeKind, Node, NodeKind


def test_cosine():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector → 0, no div0


def _node(nid: str, emb: list[float]) -> Node:
    return Node(
        id=nid,
        kind=NodeKind.CHUNK,
        title=nid,
        content=nid,
        embedding=emb,
        level=ConsolidationLevel.L0_RAW,
    )


async def _fragmented_backend() -> MemoryBackend:
    """Three internally-connected clusters, mutually disconnected."""
    b = MemoryBackend()
    await b.connect()
    clusters = {
        "a": [1.0, 0.1, 0.1],
        "b": [0.1, 1.0, 0.1],
        "c": [0.1, 0.1, 1.0],
    }
    for name, base in clusters.items():
        n1, n2 = f"{name}1", f"{name}2"
        await b.save_node(_node(n1, base))
        await b.save_node(_node(n2, [v + 0.01 for v in base]))
        # intra-cluster edge → each cluster is one component
        await b.save_edge(
            Edge(id=f"e_{name}", source_id=n1, target_id=n2, kind=EdgeKind.RELATED, weight=1.0)
        )
    return b


@pytest.mark.asyncio
async def test_bridges_fragmented_graph_into_one_component():
    b = await _fragmented_backend()
    stats = await bridge_components(b, k=10)

    assert stats.nodes == 6
    assert stats.components_before == 3
    assert stats.components_after == 1  # fully navigable
    assert stats.isolated_after == 0
    # a spanning forest over 3 components needs exactly 2 bridges — minimal.
    assert stats.bridges_added == 2


@pytest.mark.asyncio
async def test_connects_isolated_singletons():
    b = MemoryBackend()
    await b.connect()
    # 1 mainland pair + 3 isolated singletons (no edges at all)
    await b.save_node(_node("m1", [1.0, 0.0, 0.0]))
    await b.save_node(_node("m2", [0.99, 0.01, 0.0]))
    await b.save_edge(
        Edge(id="em", source_id="m1", target_id="m2", kind=EdgeKind.RELATED, weight=1.0)
    )
    for i, emb in enumerate(([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0])):
        await b.save_node(_node(f"iso{i}", emb))

    stats = await bridge_components(b, k=10)
    assert stats.isolated_before == 3
    assert stats.components_after == 1
    assert stats.isolated_after == 0


@pytest.mark.asyncio
async def test_bridges_are_persisted_as_edges():
    b = await _fragmented_backend()
    await bridge_components(b, k=10)
    # every node must now be reachable: total RELATED edges = 3 intra + 2 bridges
    all_edges = []
    for nid in ("a1", "a2", "b1", "b2", "c1", "c2"):
        all_edges += await b.get_edges(nid, direction="outgoing")
    bridge_edges = [e for e in all_edges if e.id.startswith("bridge_")]
    assert len(bridge_edges) == 2
    assert all(e.kind == EdgeKind.RELATED for e in bridge_edges)


@pytest.mark.asyncio
async def test_lexical_bridge_for_non_embedded_node():
    # An entity-style node with no embedding but a title that lexically matches
    # the corpus bridges via FTS (the MENTIONS-style link), not vector.
    b = await _fragmented_backend()
    # title "a1" exactly matches existing node a1's FTS → lexical bridge
    await b.save_node(
        Node(
            id="ent",
            kind=NodeKind.ENTITY,
            title="a1",
            content="a1",
            embedding=[],
            level=ConsolidationLevel.L0_RAW,
        )
    )
    stats = await bridge_components(b, k=10)
    assert stats.bridges_lexical >= 1  # entity bridged by FTS, no embedding
    assert stats.skipped_no_signal == 0


@pytest.mark.asyncio
async def test_skips_node_with_no_signal_and_idempotent():
    b = await _fragmented_backend()
    # truly unbridgeable: no embedding AND no text
    await b.save_node(
        Node(
            id="ghost",
            kind=NodeKind.ENTITY,
            title="",
            content="",
            embedding=[],
            level=ConsolidationLevel.L0_RAW,
        )
    )
    s1 = await bridge_components(b, k=10)
    assert s1.skipped_no_signal == 1
    # re-running adds no new bridges (stable ids, already connected)
    s2 = await bridge_components(b, k=10)
    assert s2.bridges_added == 0


@pytest.mark.asyncio
async def test_dry_run_diagnoses_without_mutating():
    b = await _fragmented_backend()
    stats = await bridge_components(b, dry_run=True)
    # reports the fragmentation...
    assert stats.components_before == 3
    assert stats.components_after == 3  # unchanged — diagnosis only
    assert stats.bridges_added == 0
    # ...and wrote no bridge edges
    edges = []
    for nid in ("a1", "a2", "b1", "b2", "c1", "c2"):
        edges += await b.get_edges(nid, direction="outgoing")
    assert not any(e.id.startswith("bridge_") for e in edges)


@pytest.mark.asyncio
async def test_empty_graph_is_safe():
    b = MemoryBackend()
    await b.connect()
    stats = await bridge_components(b, k=10)
    assert stats.nodes == 0
    assert stats.bridges_added == 0
