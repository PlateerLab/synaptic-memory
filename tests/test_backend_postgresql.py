"""Tests for PostgreSQL backend. Requires a running PostgreSQL server.

Run with: uv run pytest tests/test_backend_postgresql.py -v -m integration
Skipped in CI unless PG_DSN is set.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest

from synaptic.backends.postgresql import PostgreSQLBackend
from synaptic.models import Edge, EdgeKind, Node, NodeKind

PG_DSN = os.environ.get("PG_DSN", "postgresql://ailab:ailab123@localhost:5432/plateerag")

pytestmark = pytest.mark.integration

_PREFIX = "test_"


@pytest.fixture
async def pg() -> AsyncGenerator[PostgreSQLBackend]:
    backend = PostgreSQLBackend(PG_DSN, embedding_dim=4)
    await backend.connect()

    await backend.execute_raw(
        "DELETE FROM syn_edges WHERE source_id LIKE 'test_%' OR target_id LIKE 'test_%'"
    )
    await backend.execute_raw("DELETE FROM syn_nodes WHERE id LIKE 'test_%'")

    yield backend

    await backend.execute_raw(
        "DELETE FROM syn_edges WHERE source_id LIKE 'test_%' OR target_id LIKE 'test_%'"
    )
    await backend.execute_raw("DELETE FROM syn_nodes WHERE id LIKE 'test_%'")
    await backend.close()


def _tid(suffix: str) -> str:
    return f"{_PREFIX}{suffix}"


class TestPostgreSQLNodes:
    async def test_save_and_get(self, pg: PostgreSQLBackend) -> None:
        node = Node(id=_tid("n1"), title="Test Node", content="Hello", kind=NodeKind.LESSON)
        await pg.save_node(node)
        fetched = await pg.get_node(_tid("n1"))
        assert fetched is not None
        assert fetched.title == "Test Node"
        assert fetched.kind == NodeKind.LESSON

    async def test_update(self, pg: PostgreSQLBackend) -> None:
        node = Node(id=_tid("n2"), title="Original")
        await pg.save_node(node)
        node.title = "Updated"
        node.success_count = 5
        await pg.update_node(node)
        fetched = await pg.get_node(_tid("n2"))
        assert fetched is not None
        assert fetched.title == "Updated"
        assert fetched.success_count == 5

    async def test_delete(self, pg: PostgreSQLBackend) -> None:
        node = Node(id=_tid("n3"), title="ToDelete")
        await pg.save_node(node)
        await pg.delete_node(_tid("n3"))
        assert await pg.get_node(_tid("n3")) is None

    async def test_list_by_kind(self, pg: PostgreSQLBackend) -> None:
        await pg.save_node(Node(id=_tid("n4"), title="A", kind=NodeKind.LESSON))
        await pg.save_node(Node(id=_tid("n5"), title="B", kind=NodeKind.RULE))
        lessons = await pg.list_nodes(kind=NodeKind.LESSON)
        ids = [n.id for n in lessons]
        assert _tid("n4") in ids

    async def test_tags_roundtrip(self, pg: PostgreSQLBackend) -> None:
        node = Node(id=_tid("n6"), title="Tagged", tags=["deploy", "ci/cd"])
        await pg.save_node(node)
        fetched = await pg.get_node(_tid("n6"))
        assert fetched is not None
        assert fetched.tags == ["deploy", "ci/cd"]


class TestPostgreSQLEdges:
    async def test_save_and_get(self, pg: PostgreSQLBackend) -> None:
        await pg.save_node(Node(id=_tid("ea"), title="A"))
        await pg.save_node(Node(id=_tid("eb"), title="B"))
        edge = Edge(
            id=_tid("e1"),
            source_id=_tid("ea"),
            target_id=_tid("eb"),
            kind=EdgeKind.CAUSED,
        )
        await pg.save_edge(edge)
        edges = await pg.get_edges(_tid("ea"), direction="outgoing")
        assert len(edges) >= 1
        assert any(e.kind == EdgeKind.CAUSED for e in edges)


class TestPostgreSQLSearch:
    async def test_fts(self, pg: PostgreSQLBackend) -> None:
        await pg.save_node(
            Node(id=_tid("s1"), title="Python programming", content="Learn Python basics")
        )
        await pg.save_node(Node(id=_tid("s2"), title="Java guide", content="Java for beginners"))
        results = await pg.search_fts("Python")
        titles = [n.title for n in results]
        assert "Python programming" in titles

    async def test_fuzzy(self, pg: PostgreSQLBackend) -> None:
        await pg.save_node(
            Node(id=_tid("s3"), title="Performance tuning", content="Optimize queries")
        )
        results = await pg.search_fuzzy("Performance", threshold=0.1)
        assert len(results) >= 1

    async def test_vector(self, pg: PostgreSQLBackend) -> None:
        node = Node(
            id=_tid("s4"),
            title="Vector test",
            content="Embedding search",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        await pg.save_node(node)
        results = await pg.search_vector([0.1, 0.2, 0.3, 0.4], limit=5)
        assert len(results) >= 1
        assert results[0].title == "Vector test"

    async def test_hybrid(self, pg: PostgreSQLBackend) -> None:
        await pg.save_node(
            Node(id=_tid("s5"), title="배포 자동화", content="CI/CD 파이프라인 구현")
        )
        results = await pg.search_hybrid("배포", limit=5)
        assert len(results) >= 1
        node, score = results[0]
        assert "배포" in node.title
        assert score > 0


class TestPostgreSQLTraversal:
    async def test_neighbors(self, pg: PostgreSQLBackend) -> None:
        await pg.save_node(Node(id=_tid("ta"), title="A"))
        await pg.save_node(Node(id=_tid("tb"), title="B"))
        await pg.save_node(Node(id=_tid("tc"), title="C"))
        await pg.save_edge(
            Edge(
                id=_tid("te1"),
                source_id=_tid("ta"),
                target_id=_tid("tb"),
            )
        )
        await pg.save_edge(
            Edge(
                id=_tid("te2"),
                source_id=_tid("tb"),
                target_id=_tid("tc"),
            )
        )
        neighbors = await pg.get_neighbors(_tid("ta"), depth=2)
        ids = {n.id for n, _ in neighbors}
        assert _tid("tb") in ids


class TestPostgreSQLMaintenance:
    async def test_prune(self, pg: PostgreSQLBackend) -> None:
        await pg.save_node(Node(id=_tid("pa"), title="A"))
        await pg.save_node(Node(id=_tid("pb"), title="B"))
        await pg.save_edge(
            Edge(
                id=_tid("pe"),
                source_id=_tid("pa"),
                target_id=_tid("pb"),
                weight=0.05,
            )
        )
        pruned = await pg.prune_edges(weight_below=0.1)
        assert pruned >= 1

    async def test_decay(self, pg: PostgreSQLBackend) -> None:
        await pg.save_node(Node(id=_tid("da"), title="Decay", vitality=1.0))
        count = await pg.decay_vitality(factor=0.9)
        assert count >= 1
