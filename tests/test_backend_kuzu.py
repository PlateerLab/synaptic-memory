"""Tests for Kuzu embedded graph backend.

Unlike previous Neo4j tests, these run in CI without external infrastructure —
Kuzu is an embedded library installed via ``pip install kuzu``.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from synaptic.models import ConsolidationLevel, Edge, EdgeKind, Node, NodeKind

try:
    from synaptic.backends.kuzu import KuzuBackend

    HAS_KUZU = True
except ImportError:
    HAS_KUZU = False

pytestmark = [
    pytest.mark.kuzu,
    pytest.mark.skipif(not HAS_KUZU, reason="kuzu not installed"),
]


@pytest.fixture
async def kuzu() -> AsyncGenerator[KuzuBackend]:
    with tempfile.TemporaryDirectory(prefix="kuzu-test-") as tmp:
        db_path = Path(tmp) / "test.kuzu"
        b = KuzuBackend(str(db_path))
        await b.connect()
        try:
            yield b
        finally:
            await b.close()


class TestKuzuNodes:
    async def test_save_and_get(self, kuzu: KuzuBackend) -> None:
        node = Node(
            id="n1",
            title="Deploy pipeline",
            content="Helm + ArgoCD",
            kind=NodeKind.LESSON,
            tags=["devops", "infra"],
        )
        await kuzu.save_node(node)
        fetched = await kuzu.get_node("n1")
        assert fetched is not None
        assert fetched.title == "Deploy pipeline"
        assert fetched.content == "Helm + ArgoCD"
        assert fetched.tags == ["devops", "infra"]
        assert str(fetched.kind) == "lesson"

    async def test_get_nonexistent(self, kuzu: KuzuBackend) -> None:
        assert await kuzu.get_node("missing") is None

    async def test_update(self, kuzu: KuzuBackend) -> None:
        node = Node(id="u1", title="Original", content="v1")
        await kuzu.save_node(node)
        node.title = "Updated"
        node.success_count = 5
        node.vitality = 0.5
        await kuzu.update_node(node)
        fetched = await kuzu.get_node("u1")
        assert fetched is not None
        assert fetched.title == "Updated"
        assert fetched.success_count == 5
        assert fetched.vitality == pytest.approx(0.5)

    async def test_upsert_via_save(self, kuzu: KuzuBackend) -> None:
        """save_node should update an existing node rather than failing."""
        node = Node(id="upsert", title="v1")
        await kuzu.save_node(node)
        node.title = "v2"
        await kuzu.save_node(node)  # upsert
        fetched = await kuzu.get_node("upsert")
        assert fetched is not None
        assert fetched.title == "v2"

    async def test_delete(self, kuzu: KuzuBackend) -> None:
        node = Node(id="d1", title="Temporary")
        await kuzu.save_node(node)
        await kuzu.delete_node("d1")
        assert await kuzu.get_node("d1") is None

    async def test_list_filter_by_kind(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="l1", title="Lesson A", kind=NodeKind.LESSON))
        await kuzu.save_node(Node(id="r1", title="Rule A", kind=NodeKind.RULE))
        await kuzu.save_node(Node(id="l2", title="Lesson B", kind=NodeKind.LESSON))

        lessons = await kuzu.list_nodes(kind=NodeKind.LESSON)
        assert len(lessons) == 2
        assert all(str(n.kind) == "lesson" for n in lessons)

    async def test_properties_roundtrip(self, kuzu: KuzuBackend) -> None:
        node = Node(
            id="p1",
            title="With props",
            properties={"department": "infra", "priority": "high"},
        )
        await kuzu.save_node(node)
        fetched = await kuzu.get_node("p1")
        assert fetched is not None
        assert fetched.properties == {"department": "infra", "priority": "high"}

    async def test_consolidation_level_roundtrip(self, kuzu: KuzuBackend) -> None:
        node = Node(id="cl1", title="Monthly", level=ConsolidationLevel.L2_MONTHLY)
        await kuzu.save_node(node)
        fetched = await kuzu.get_node("cl1")
        assert fetched is not None
        assert fetched.level == ConsolidationLevel.L2_MONTHLY


class TestKuzuEdges:
    async def test_save_and_get(self, kuzu: KuzuBackend) -> None:
        n1 = Node(id="s1", title="Source")
        n2 = Node(id="t1", title="Target")
        await kuzu.save_node(n1)
        await kuzu.save_node(n2)

        edge = Edge(
            id="e1",
            source_id="s1",
            target_id="t1",
            kind=EdgeKind.RELATED,
            weight=0.75,
        )
        await kuzu.save_edge(edge)

        out_edges = await kuzu.get_edges("s1", direction="outgoing")
        assert len(out_edges) == 1
        assert out_edges[0].target_id == "t1"
        assert out_edges[0].weight == pytest.approx(0.75)

    async def test_edge_directions(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="a"))
        await kuzu.save_node(Node(id="b"))
        await kuzu.save_edge(Edge(id="ab", source_id="a", target_id="b", kind=EdgeKind.RELATED))
        await kuzu.save_edge(Edge(id="ba", source_id="b", target_id="a", kind=EdgeKind.RELATED))

        out_a = await kuzu.get_edges("a", direction="outgoing")
        in_a = await kuzu.get_edges("a", direction="incoming")
        both_a = await kuzu.get_edges("a", direction="both")
        assert len(out_a) == 1
        assert len(in_a) == 1
        assert len(both_a) == 2

    async def test_update_edge_weight(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="u_a"))
        await kuzu.save_node(Node(id="u_b"))
        edge = Edge(id="u_e", source_id="u_a", target_id="u_b", weight=0.5)
        await kuzu.save_edge(edge)

        edge.weight = 0.9
        await kuzu.update_edge(edge)
        edges = await kuzu.get_edges("u_a")
        assert edges[0].weight == pytest.approx(0.9)

    async def test_delete_edge(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="de_a"))
        await kuzu.save_node(Node(id="de_b"))
        await kuzu.save_edge(
            Edge(id="de_e", source_id="de_a", target_id="de_b", kind=EdgeKind.RELATED)
        )
        await kuzu.delete_edge("de_e")
        assert await kuzu.get_edges("de_a") == []

    async def test_delete_node_cascades_edges(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="cas_a"))
        await kuzu.save_node(Node(id="cas_b"))
        await kuzu.save_edge(
            Edge(id="cas_e", source_id="cas_a", target_id="cas_b", kind=EdgeKind.RELATED)
        )
        await kuzu.delete_node("cas_a")
        # Incoming edge to cas_b should be gone
        assert await kuzu.get_edges("cas_b", direction="incoming") == []


class TestKuzuSearch:
    async def test_fts_finds_content(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(
            Node(id="fts1", title="Deployment pipeline", content="ArgoCD and Helm charts")
        )
        await kuzu.save_node(
            Node(id="fts2", title="Incident postmortem", content="Database overload")
        )
        results = await kuzu.search_fts("deployment")
        assert any(n.id == "fts1" for n in results)

    async def test_fuzzy_case_insensitive(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="fz1", title="UPPERCASE TITLE", content="mixed Content"))
        results = await kuzu.search_fuzzy("uppercase")
        assert any(n.id == "fz1" for n in results)

    async def test_vector_returns_empty(self, kuzu: KuzuBackend) -> None:
        """Vector search is not implemented at the Kuzu backend level."""
        results = await kuzu.search_vector([0.1, 0.2, 0.3])
        assert results == []


class TestKuzuTraversal:
    async def test_neighbors_depth1(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="t_a"))
        await kuzu.save_node(Node(id="t_b"))
        await kuzu.save_node(Node(id="t_c"))
        await kuzu.save_edge(Edge(id="ab", source_id="t_a", target_id="t_b", kind=EdgeKind.RELATED))
        await kuzu.save_edge(Edge(id="bc", source_id="t_b", target_id="t_c", kind=EdgeKind.RELATED))

        neighbors = await kuzu.get_neighbors("t_a", depth=1)
        neighbor_ids = {n.id for n, _ in neighbors}
        assert neighbor_ids == {"t_b"}

    async def test_neighbors_depth2(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="d2_a"))
        await kuzu.save_node(Node(id="d2_b"))
        await kuzu.save_node(Node(id="d2_c"))
        await kuzu.save_edge(
            Edge(id="d2_ab", source_id="d2_a", target_id="d2_b", kind=EdgeKind.RELATED)
        )
        await kuzu.save_edge(
            Edge(id="d2_bc", source_id="d2_b", target_id="d2_c", kind=EdgeKind.RELATED)
        )

        neighbors = await kuzu.get_neighbors("d2_a", depth=2)
        neighbor_ids = {n.id for n, _ in neighbors}
        assert neighbor_ids == {"d2_b", "d2_c"}

    async def test_shortest_path(self, kuzu: KuzuBackend) -> None:
        # Build: a -> b -> c -> d, plus shortcut a -> d
        for nid in ("sp_a", "sp_b", "sp_c", "sp_d"):
            await kuzu.save_node(Node(id=nid))
        await kuzu.save_edge(
            Edge(id="sp_ab", source_id="sp_a", target_id="sp_b", kind=EdgeKind.RELATED)
        )
        await kuzu.save_edge(
            Edge(id="sp_bc", source_id="sp_b", target_id="sp_c", kind=EdgeKind.RELATED)
        )
        await kuzu.save_edge(
            Edge(id="sp_cd", source_id="sp_c", target_id="sp_d", kind=EdgeKind.RELATED)
        )

        path = await kuzu.shortest_path("sp_a", "sp_d", max_depth=5)
        assert len(path) == 3
        path_ids = [n.id for n, _ in path]
        assert path_ids == ["sp_b", "sp_c", "sp_d"]

    async def test_shortest_path_self(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="self"))
        path = await kuzu.shortest_path("self", "self")
        assert path == []

    async def test_shortest_path_unreachable(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="iso_a"))
        await kuzu.save_node(Node(id="iso_b"))
        path = await kuzu.shortest_path("iso_a", "iso_b", max_depth=3)
        assert path == []


class TestKuzuMaintenance:
    async def test_prune_edges(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="pr_a"))
        await kuzu.save_node(Node(id="pr_b"))
        await kuzu.save_edge(Edge(id="heavy", source_id="pr_a", target_id="pr_b", weight=0.9))
        await kuzu.save_edge(Edge(id="light", source_id="pr_a", target_id="pr_b", weight=0.05))

        pruned = await kuzu.prune_edges(weight_below=0.1)
        assert pruned == 1

        edges = await kuzu.get_edges("pr_a")
        assert len(edges) == 1
        assert edges[0].id == "heavy"

    async def test_decay_vitality(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="v1", vitality=1.0))
        await kuzu.save_node(Node(id="v2", vitality=1.0))

        decayed = await kuzu.decay_vitality(factor=0.5)
        assert decayed == 2

        n1 = await kuzu.get_node("v1")
        assert n1 is not None
        assert n1.vitality == pytest.approx(0.5)


class TestKuzuBatch:
    async def test_save_nodes_batch(self, kuzu: KuzuBackend) -> None:
        nodes = [Node(id=f"b_{i}", title=f"Node {i}") for i in range(5)]
        await kuzu.save_nodes_batch(nodes)
        for i in range(5):
            fetched = await kuzu.get_node(f"b_{i}")
            assert fetched is not None
            assert fetched.title == f"Node {i}"

    async def test_save_edges_batch(self, kuzu: KuzuBackend) -> None:
        await kuzu.save_node(Node(id="eb_src"))
        await kuzu.save_node(Node(id="eb_tgt"))
        edges = [
            Edge(
                id=f"eb_{i}",
                source_id="eb_src",
                target_id="eb_tgt",
                kind=EdgeKind.RELATED,
                weight=float(i) / 10,
            )
            for i in range(3)
        ]
        await kuzu.save_edges_batch(edges)
        out = await kuzu.get_edges("eb_src")
        assert len(out) == 3
