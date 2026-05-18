"""Tests for SQLite backend."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from synaptic.backends.sqlite import SQLiteBackend
from synaptic.models import Edge, EdgeKind, Node, NodeKind


@pytest.fixture
async def sqlite() -> AsyncGenerator[SQLiteBackend]:
    b = SQLiteBackend(":memory:")
    await b.connect()
    yield b
    await b.close()


class TestSQLiteNodes:
    async def test_save_and_get(self, sqlite: SQLiteBackend) -> None:
        node = Node(title="Test", content="Content", kind=NodeKind.LESSON, tags=["a", "b"])
        await sqlite.save_node(node)
        fetched = await sqlite.get_node(node.id)
        assert fetched is not None
        assert fetched.title == "Test"
        assert fetched.tags == ["a", "b"]

    async def test_update(self, sqlite: SQLiteBackend) -> None:
        node = Node(title="Original")
        await sqlite.save_node(node)
        node.title = "Updated"
        node.success_count = 5
        await sqlite.update_node(node)
        fetched = await sqlite.get_node(node.id)
        assert fetched is not None
        assert fetched.title == "Updated"
        assert fetched.success_count == 5

    async def test_delete(self, sqlite: SQLiteBackend) -> None:
        node = Node(title="ToDelete")
        await sqlite.save_node(node)
        await sqlite.delete_node(node.id)
        assert await sqlite.get_node(node.id) is None

    async def test_list_filter(self, sqlite: SQLiteBackend) -> None:
        await sqlite.save_node(Node(title="A", kind=NodeKind.LESSON))
        await sqlite.save_node(Node(title="B", kind=NodeKind.RULE))

        lessons = await sqlite.list_nodes(kind=NodeKind.LESSON)
        assert len(lessons) == 1
        assert lessons[0].kind == NodeKind.LESSON


class TestSQLiteEdges:
    async def test_save_and_get(self, sqlite: SQLiteBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        await sqlite.save_node(n1)
        await sqlite.save_node(n2)
        edge = Edge(source_id=n1.id, target_id=n2.id, kind=EdgeKind.CAUSED)
        await sqlite.save_edge(edge)

        edges = await sqlite.get_edges(n1.id, direction="outgoing")
        assert len(edges) == 1

    async def test_direction_both(self, sqlite: SQLiteBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        await sqlite.save_node(n1)
        await sqlite.save_node(n2)
        await sqlite.save_edge(Edge(source_id=n1.id, target_id=n2.id))

        both = await sqlite.get_edges(n1.id, direction="both")
        assert len(both) == 1


class TestSQLiteSearch:
    async def test_fts(self, sqlite: SQLiteBackend) -> None:
        await sqlite.save_node(Node(title="Python programming", content="Learn Python basics"))
        await sqlite.save_node(Node(title="Java guide", content="Java for beginners"))

        results = await sqlite.search_fts("Python")
        assert len(results) == 1

    async def test_fts_korean(self, sqlite: SQLiteBackend) -> None:
        await sqlite.save_node(Node(title="배포 자동화", content="CI/CD 파이프라인 구현"))
        results = await sqlite.search_fts("배포")
        assert len(results) == 1

    async def test_fuzzy_like(self, sqlite: SQLiteBackend) -> None:
        await sqlite.save_node(Node(title="Performance tuning", content="Optimize database"))
        results = await sqlite.search_fuzzy("Performance")
        assert len(results) >= 1


class TestSQLiteTraversal:
    async def test_neighbors(self, sqlite: SQLiteBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        n3 = Node(title="C")
        await sqlite.save_node(n1)
        await sqlite.save_node(n2)
        await sqlite.save_node(n3)
        await sqlite.save_edge(Edge(source_id=n1.id, target_id=n2.id))
        await sqlite.save_edge(Edge(source_id=n2.id, target_id=n3.id))

        neighbors = await sqlite.get_neighbors(n1.id, depth=2)
        ids = {n.id for n, _ in neighbors}
        assert n2.id in ids


class TestSQLiteMaintenance:
    async def test_prune(self, sqlite: SQLiteBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        await sqlite.save_node(n1)
        await sqlite.save_node(n2)
        await sqlite.save_edge(Edge(source_id=n1.id, target_id=n2.id, weight=0.05))

        pruned = await sqlite.prune_edges(weight_below=0.1)
        assert pruned == 1

    async def test_decay(self, sqlite: SQLiteBackend) -> None:
        await sqlite.save_node(Node(title="Test", vitality=1.0))
        count = await sqlite.decay_vitality(factor=0.9)
        assert count == 1


class TestSQLiteVectorSearch:
    async def test_search_vector_returns_by_cosine_similarity(self, sqlite: SQLiteBackend) -> None:
        for i, emb in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.9, 0.1, 0]]):
            await sqlite.save_node(Node(id=f"v{i}", title=f"vec {i}", embedding=emb))
        results = await sqlite.search_vector([1, 0, 0], limit=3)
        assert results[0].id == "v0"
        assert results[1].id == "v3"

    async def test_search_vector_returns_empty_when_no_embeddings(
        self, sqlite: SQLiteBackend
    ) -> None:
        await sqlite.save_node(Node(id="noEmb", title="no embedding"))
        results = await sqlite.search_vector([1, 0, 0], limit=5)
        assert results == []

    async def test_search_vector_with_empty_query_returns_empty(
        self, sqlite: SQLiteBackend
    ) -> None:
        await sqlite.save_node(Node(id="e1", title="has emb", embedding=[1, 0]))
        results = await sqlite.search_vector([], limit=5)
        assert results == []

    async def test_embedding_persisted_across_reload(self, sqlite: SQLiteBackend) -> None:
        await sqlite.save_node(Node(id="persist", title="persist test", embedding=[0.5, 0.5, 0.5]))
        loaded = await sqlite.get_node("persist")
        assert loaded is not None
        assert loaded.embedding == [0.5, 0.5, 0.5]


class TestFindNodesByProperty:
    async def test_finds_matching_property(self, sqlite: SQLiteBackend) -> None:
        await sqlite.save_node(
            Node(id="a", kind=NodeKind.CHUNK, properties={"doc_id": "h1", "law": "X"})
        )
        await sqlite.save_node(
            Node(id="b", kind=NodeKind.CHUNK, properties={"doc_id": "h1", "law": "X"})
        )
        await sqlite.save_node(
            Node(id="c", kind=NodeKind.CHUNK, properties={"doc_id": "h2"})
        )
        hits = await sqlite.find_nodes_by_property("doc_id", "h1")
        assert {n.id for n in hits} == {"a", "b"}

    async def test_no_match_returns_empty(self, sqlite: SQLiteBackend) -> None:
        await sqlite.save_node(Node(id="a", properties={"doc_id": "h1"}))
        assert await sqlite.find_nodes_by_property("doc_id", "nope") == []

    async def test_respects_limit(self, sqlite: SQLiteBackend) -> None:
        for i in range(5):
            await sqlite.save_node(Node(id=f"n{i}", properties={"k": "v"}))
        assert len(await sqlite.find_nodes_by_property("k", "v", limit=3)) == 3
