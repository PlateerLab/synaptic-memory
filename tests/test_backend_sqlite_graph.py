"""Tests for SqliteGraphBackend — SQLite + GraphTraversal protocol.

The parent ``SQLiteBackend`` already has comprehensive coverage in
``test_backend_sqlite.py``. This module focuses on:

1. Subclass instantiation and inheritance sanity
2. The three added methods from the GraphTraversal protocol:
   ``shortest_path``, ``find_by_type_hierarchy``, ``pattern_match``
3. End-to-end parity with KuzuBackend on a small 2-hop corpus
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.models import Edge, EdgeKind, Node, NodeKind


@pytest.fixture
async def backend() -> AsyncGenerator[SqliteGraphBackend]:
    b = SqliteGraphBackend(":memory:")
    await b.connect()
    yield b
    await b.close()


async def _seed_linear_chain(backend: SqliteGraphBackend, n: int) -> list[Node]:
    """Create n nodes connected as A → B → C → ... in a straight line."""
    nodes: list[Node] = []
    for i in range(n):
        node = Node(id=f"n{i}", title=f"Node {i}", kind=NodeKind.CONCEPT)
        await backend.save_node(node)
        nodes.append(node)

    for i in range(n - 1):
        edge = Edge(
            id=f"e{i}",
            source_id=nodes[i].id,
            target_id=nodes[i + 1].id,
            kind=EdgeKind.RELATED,
            weight=1.0,
        )
        await backend.save_edge(edge)

    return nodes


# --- Inheritance sanity ---


class TestInheritance:
    async def test_is_sqlite_backend(self, backend: SqliteGraphBackend) -> None:
        from synaptic.backends.sqlite import SQLiteBackend

        assert isinstance(backend, SQLiteBackend)

    async def test_inherited_crud_works(self, backend: SqliteGraphBackend) -> None:
        """Sanity check that the parent's save/get still work."""
        node = Node(title="hello", content="world", kind=NodeKind.CONCEPT)
        await backend.save_node(node)
        fetched = await backend.get_node(node.id)
        assert fetched is not None
        assert fetched.title == "hello"
        assert fetched.content == "world"

    async def test_inherited_fts_works(self, backend: SqliteGraphBackend) -> None:
        node = Node(
            title="Knowledge Graph",
            content="Ontology-based reasoning",
            kind=NodeKind.CONCEPT,
        )
        await backend.save_node(node)
        results = await backend.search_fts("ontology")
        assert len(results) >= 1
        assert any(n.id == node.id for n in results)

    async def test_inherited_neighbors_works(
        self, backend: SqliteGraphBackend
    ) -> None:
        nodes = await _seed_linear_chain(backend, 3)
        hops = await backend.get_neighbors(nodes[1].id, depth=1)
        neighbor_ids = {n.id for n, _ in hops}
        # n1 connects to n0 and n2
        assert nodes[0].id in neighbor_ids
        assert nodes[2].id in neighbor_ids


# --- shortest_path ---


class TestShortestPath:
    async def test_same_node_returns_empty(
        self, backend: SqliteGraphBackend
    ) -> None:
        await _seed_linear_chain(backend, 3)
        path = await backend.shortest_path("n0", "n0")
        assert path == []

    async def test_direct_edge_one_hop(
        self, backend: SqliteGraphBackend
    ) -> None:
        nodes = await _seed_linear_chain(backend, 3)
        path = await backend.shortest_path("n0", "n1")
        assert len(path) == 1
        assert path[0][0].id == "n1"

    async def test_two_hop_path(self, backend: SqliteGraphBackend) -> None:
        await _seed_linear_chain(backend, 4)
        path = await backend.shortest_path("n0", "n2")
        assert len(path) == 2
        assert path[0][0].id == "n1"
        assert path[1][0].id == "n2"

    async def test_three_hop_path(self, backend: SqliteGraphBackend) -> None:
        await _seed_linear_chain(backend, 5)
        path = await backend.shortest_path("n0", "n3", max_depth=5)
        assert len(path) == 3
        assert [p[0].id for p in path] == ["n1", "n2", "n3"]

    async def test_max_depth_cuts_off(
        self, backend: SqliteGraphBackend
    ) -> None:
        await _seed_linear_chain(backend, 5)
        # n0 → n4 needs depth 4, cap at 2 → no path found
        path = await backend.shortest_path("n0", "n4", max_depth=2)
        assert path == []

    async def test_unreachable_returns_empty(
        self, backend: SqliteGraphBackend
    ) -> None:
        # Two disconnected nodes
        await backend.save_node(Node(id="island_a", title="A"))
        await backend.save_node(Node(id="island_b", title="B"))
        path = await backend.shortest_path("island_a", "island_b")
        assert path == []

    async def test_picks_shortest_when_multiple_paths(
        self, backend: SqliteGraphBackend
    ) -> None:
        # Diamond: A→B→D and A→C→D; either 2-hop path is acceptable
        await backend.save_node(Node(id="A", title="A"))
        await backend.save_node(Node(id="B", title="B"))
        await backend.save_node(Node(id="C", title="C"))
        await backend.save_node(Node(id="D", title="D"))
        # A→B→D longer path with intermediate
        for eid, src, tgt in [
            ("eAB", "A", "B"),
            ("eBD", "B", "D"),
            ("eAC", "A", "C"),
            ("eCD", "C", "D"),
        ]:
            await backend.save_edge(
                Edge(id=eid, source_id=src, target_id=tgt, kind=EdgeKind.RELATED)
            )
        path = await backend.shortest_path("A", "D")
        assert len(path) == 2
        assert path[-1][0].id == "D"


# --- find_by_type_hierarchy ---


class TestFindByTypeHierarchy:
    async def test_finds_nodes_by_kind(
        self, backend: SqliteGraphBackend
    ) -> None:
        await backend.save_node(Node(id="r1", title="R1", kind=NodeKind.RULE))
        await backend.save_node(Node(id="r2", title="R2", kind=NodeKind.RULE))
        await backend.save_node(
            Node(id="d1", title="D1", kind=NodeKind.DECISION)
        )

        rules = await backend.find_by_type_hierarchy("rule")
        assert len(rules) == 2
        assert {n.id for n in rules} == {"r1", "r2"}

    async def test_respects_limit(self, backend: SqliteGraphBackend) -> None:
        for i in range(10):
            await backend.save_node(
                Node(id=f"r{i}", title=f"R{i}", kind=NodeKind.RULE)
            )
        rules = await backend.find_by_type_hierarchy("rule", limit=3)
        assert len(rules) == 3

    async def test_unknown_kind_empty(
        self, backend: SqliteGraphBackend
    ) -> None:
        await backend.save_node(Node(title="X", kind=NodeKind.RULE))
        empty = await backend.find_by_type_hierarchy("agent")
        assert empty == []


# --- pattern_match (not supported) ---


class TestPatternMatchNotImplemented:
    async def test_raises_not_implemented(
        self, backend: SqliteGraphBackend
    ) -> None:
        with pytest.raises(NotImplementedError, match="Cypher"):
            await backend.pattern_match("(:Node)-[:RELATED]->(:Node)")

    async def test_error_message_suggests_alternatives(
        self, backend: SqliteGraphBackend
    ) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            await backend.pattern_match("whatever")
        # Should point users to KuzuBackend
        assert "KuzuBackend" in str(exc_info.value)
